# @pyimport healpy as hp

# constants
const h_over_k = 0.04799243073366221
const GHz = 1.
const Tcmb = 2.726

# unit conversion
KRJ_to_KCMB(ν) = ν/Tcmb*h_over_k |> x-> (exp.(x) .- 1).^2 / (exp.(x).*x.^2)  # everything in K_CMB

# seds: @. is faster than cmb., sync., etc.
cmb(ν) = @. ν*0+1
sync(ν, β; ν₀=20*GHz) = @. (ν/ν₀)^β * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)
dust(ν, βd, Td; ν₀=150*GHz) = @. (exp(ν₀/Td*h_over_k)-1) / (exp(ν/Td*h_over_k)-1)*(ν/ν₀)^(1+βd) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)

# mixing matrices, sometime it's fast to pass in a folding function, for some reason, which is the second option
mixing_matrix(comps, ν) = (folder = fold(comps); pars -> folder(pars) |> pars-> hcat([c(ν,p...) for (c, p) in zip(comps, pars)]...))
mixing_matrix(comps, ν, folder) = (pars -> folder(pars) |> pars-> hcat([c(ν,p...) for (c, p) in zip(comps, pars)]...))

# linear algebra
𝔣LᵀA(N⁻¹, A) = [svd!(N⁻¹[:,i].^0.5 .* A) for i = 1:size(N⁻¹,2)]
# the next is just a matrix multiplication written out explictly to save memory allocs
# leverage compiler optimization
function 𝔣logL(LᵀA, Lᵀd)
    s = 0
    @tturbo for k in eachindex(axes(Lᵀd,2)), j in eachindex(axes(LᵀA.U,2))
        sjk = 0
        for i in eachindex(axes(LᵀA.U,1)); sjk += LᵀA.U[i,j] * Lᵀd[i,k] end
        s += sjk^2/2
    end
    s
end
# loop over stokes
lnlike(LᵀA, Lᵀd) = try sum([𝔣logL(LᵀA[i], view(Lᵀd,:,i,:)) for i=1:size(Lᵀd,2)]) catch; -Inf end
# get ml signal
function 𝔣s(LᵀA, Lᵀd)
    out=zeros(Float64,size(LᵀA[1].U,2),size(Lᵀd)[2:end]...)
    for i = 1:length(LᵀA)
        view(out,:,i,:) .= Octavian.matmul(LᵀA[i].V*(LᵀA[i].U'./LᵀA[i].S), view(Lᵀd,:,i,:))
    end
    out
end

function build_target(comps, ν, N⁻¹, Lᵀd; mm=nothing, use_jac=false)
    mm = isnothing(mm) ? mixing_matrix(comps, ν) : mm
    f(pars) = try 𝔣LᵀA(N⁻¹, mm(pars)) |> LᵀA -> -lnlike(LᵀA, Lᵀd) catch; -Inf end
    if use_jac; g!(storage, pars) = FiniteDiff.finite_difference_jacobian(f, pars) |> res->storage[:]=res
    else g! = ()->() end # do nothing
    f, g!
end

# main interface
function compsep(comps, ν, N⁻¹, d; mask::Union{BitArray{1},Nothing} = nothing, x₀ = [-3.0, 1.54, 20.0],
    use_jac = false, algo = BFGS(), options = Optim.Options(f_abstol = 1))
    !isnothing(mask) && (d = d[:, :, mask])
    Lᵀd = N⁻¹ .^ 0.5 .* d
    mm = mixing_matrix(comps, ν)
    f, g! = build_target(comps, ν, N⁻¹, Lᵀd; mm=mm, use_jac = use_jac)
    res = use_jac ? optimize(f, g!, x₀, algo, options) : optimize(f, x₀, algo, options)

    # postprocess results
    out = Dict()
    out["res"] = res
    # store bestfit parameter
    out["params"] = Optim.minimizer(res)
    # recover mixing matrix
    A = mm(out["params"])
    out["A"] = A
    out["s"] = 𝔣s(𝔣LᵀA(N⁻¹,A), Lᵀd)
    out
end

# add support for multi-resolution search
function compsep(comps, ν, N⁻¹, d, nside; mask::Union{BitArray{1},Nothing} = nothing, x₀ = [-3.0, 1.54, 20.0],
    use_jac = false, algo = BFGS(), options = Optim.Options(f_abstol = 1))
    (nside == 0) && (compsep(comps, ν, N⁻¹, d; mask=mask, x₀=x₀, use_jac=use_jac, algo=algo, options=options))
    masks = build_masks(nside, obs, mask = mask)
    res = Array{Any}(undef, length(masks))
    Threads.@threads for i in 1:length(masks)
        res[i] = compsep(comps, ν, N⁻¹, d; mask = masks[i], x₀ = x₀, use_jac = use_jac, algo = algo, options = options)
    end
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
        @inbounds for i in eachindex(comps), j in eachindex(bands)
            ν = LinRange(bands[j].lo, bands[j].hi, npoints)
            A[j,i] = integrate(ν, comps[i](ν,pars[i]...), method) / (bands[j].hi-bands[j].lo)
        end
        A
    end
end

# utility functions
# function build_masks(nside, obs; mask::Union{BitArray{1},Nothing}=nothing) where T
    # npix = hp.nside2npix(nside)
    # patch_ids = hp.ud_grade(collect(1:npix), hp.npix2nside(size(obs,3)))
    # isnothing(mask) && (return [(patch_ids .== i) for i = 1:npix])
    # return [(patch_ids .== i) .& mask for i = 1:npix]
# end
parse_sigs(comps; nskip=1) = map(c->methods(c)[1].nargs-1-nskip, comps) |> cumsum |> x->[1;x[1:end-1].+1;;x] |> x->map((b,e)->range(b,e),x[:,1],x[:,2])
fold(comps; nskip=1) = (sigs=parse_sigs(comps, nskip=nskip); params -> (params |> p -> map(sl->p[sl], sigs)))
