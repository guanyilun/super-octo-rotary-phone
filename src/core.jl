using TensorOperations, LinearAlgebra, Optim, FiniteDiff, LoopVectorization, PyCall, Debugger, NumericalIntegration
@pyimport healpy as hp

# constants
const h_over_k = 0.04799243073366221
const GHz = 1.
const Tcmb = 2.726

# unit conversion
KRJ_to_KCMB(ν) = ν/Tcmb*h_over_k |> x-> (exp.(x) .- 1).^2 / (exp.(x).*x.^2)  # everything in K_CMB

# seds: @> is faster than cmb., sync., etc.
cmb(ν) = @. ν*0+1
sync(ν, β; ν₀=20*GHz) = @. (ν/ν₀)^β * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)
dust(ν, βd, Td; ν₀=150*GHz) = @. (exp(ν₀/Td*h_over_k)-1) / (exp(ν/Td*h_over_k)-1)*(ν/ν₀)^(1+βd) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)

mixing_matrix(comps, ν) = (folder = fold(comps); pars -> folder(pars) |> pars-> hcat([c(ν,p...) for (c, p) in zip(comps, pars)]...))
mixing_matrix(comps, ν; folder) = (pars -> folder(pars) |> pars-> hcat([c(ν,p...) for (c, p) in zip(comps, pars)]...))

𝔣LᵀA(N⁻¹, A) = [svd!(N⁻¹[:,i].^0.5 .* A) for i = 1:size(N⁻¹,2)]
function 𝔣logL(LᵀA, Lᵀd)
    s = 0
    @tturbo for k in eachindex(axes(Lᵀd,2)), j in eachindex(axes(LᵀA.U,2))
        sjk = 0
        for i in eachindex(axes(LᵀA.U,1)); sjk += LᵀA.U[i,j] * Lᵀd[i,k] end
        s += sjk^2/2
    end
    s
end
lnlike(LᵀA, Lᵀd) = try sum([𝔣logL(LᵀA[i], view(Lᵀd,:,i,:)) for i=1:size(Lᵀd,2)]) catch; -Inf end

# build to-be-minimized function
function build_target(comps, ν, N⁻¹, Lᵀd; mm=nothing, use_jac=false)
    mm = ifelse(mm == nothing, mixing_matrix(comps, ν; folder=fold(comps)), mm)
    f(pars) = try 𝔣LᵀA(N⁻¹, mm(pars)) |> LᵀA -> -lnlike(LᵀA, Lᵀd) catch; -Inf end
    if use_jac; g!(storage, pars) = FiniteDiff.finite_difference_jacobian(f, pars) |> res->storage[:]=res
    else g! = ()->() end # do nothing
    f, g!
end

# main interface
function compsep(comps, ν, N⁻¹, d; mask::Union{BitArray{1},Nothing}=nothing, x₀=[-3.,1.54,20.],
                 use_jac=false, algo=BFGS(), options=Optim.Options(f_abstol=1))
    if mask isa BitArray{1}; d = d[:,:,mask] end
    Lᵀd = N⁻¹.^0.5 .* d
    f, g! = build_target(comps, ν, N⁻¹, Lᵀd, use_jac=use_jac)
    res = use_jac ? optimize(f, g!, x₀, algo, options) : optimize(f, x₀, algo, options)
    res
end

function compsep(comps, ν, N⁻¹, d, nside; mask::Union{BitArray{1},Nothing}=nothing, x₀=[-3.,1.54,20.],
                 use_jac=false, algo=BFGS(), options=Optim.Options(f_abstol=1))
    if nside == 0; return compsep(comps, ν, N⁻¹, d; mask=mask, x₀=x₀, use_jac=use_jac, algo=algo, options=options) end
    masks = build_masks(nside, obs, mask=mask)
    res = Array{Any}(undef, length(masks))
    Threads.@threads for i in 1:length(masks); res[i] = compsep(comps, ν, N⁻¹, d; mask=masks[i], x₀=x₀, use_jac=use_jac, algo=algo, options=options) end
    res
end

# for bandpass integration
struct SimplePassband
    lo::AbstractFloat
    hi::AbstractFloat
    SimplePassband(center, width) = new(center-width/2, center+width/2)
end

# to work with bandpass integration: note that this can be ~100 times slower
function mixing_matrix(comps, bands::Vector{SimplePassband}; npoints=10, method=TrapezoidalEvenFast())
    folder = fold(comps)
    function mm(pars)
        pars = folder(pars)
        A = zeros(Float64, length(bands), length(comps))
        for i = 1:length(comps), j = 1:length(bands)
            ν = LinRange(bands[j].lo, bands[j].hi, npoints)
            A[j,i] = integrate(ν, comps[i](ν,pars[i]...), method) / (bands[j].hi-bands[j].lo)
        end
        A
    end
end

# utility functions
function build_masks(nside, obs; mask::Union{BitArray{1},Nothing}=nothing) where T
    npix = hp.nside2npix(nside)
    patch_ids = hp.ud_grade(collect(1:npix), hp.npix2nside(size(obs,3)))
    if mask isa Nothing; return [(patch_ids .== i) for i = 1:npix]
    else; return [(patch_ids .== i) .& mask for i = 1:npix] end
end
parse_sigs(comps; nskip=1) = map(c->methods(c)[1].nargs-1-nskip, comps) |> cumsum |> x->[1;x[1:end-1].+1;;x] |> x->map((b,e)->range(b,e),x[:,1],x[:,2])
fold(comps; nskip=1) = (sigs=parse_sigs(comps, nskip=nskip); params -> (params |> p -> map(sl->p[sl], sigs)))