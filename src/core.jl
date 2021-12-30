using TensorOperations, LinearAlgebra, Optim, FiniteDiff, LoopVectorization

# constants
const h_over_k = 0.04799243073366221
const GHz = 1.
const Tcmb = 2.726

# unit conversion
KRJ_to_KCMB(ν) = ν/Tcmb*h_over_k |> x-> (exp.(x) .- 1).^2 / (exp.(x).*x.^2)  # everything in K_CMB

# seds
cmb(ν) = @. ν*0+1
sync(ν, β; ν₀=20*GHz) = @. (ν/ν₀)^β * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)
dust(ν, βd, Td; ν₀=150*GHz) = @. (exp(ν₀/Td*h_over_k)-1) / (exp(ν/Td*h_over_k)-1)*(ν/ν₀)^(1+βd) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)

mixing_matrix(comps, ν; folder) = pars -> folder(pars) |> pars-> hcat([c(ν,p...) for (c, p) in zip(comps, pars)]...)

𝔣LᵀA(N⁻¹, A) = [svd!(N⁻¹[:,i].^0.5 .* A) for i = 1:size(N⁻¹,2)]
function 𝔣logL(LᵀA, Lᵀd)
    s = 0
    @tturbo for k in eachindex(axes(Lᵀd,2)), j in eachindex(axes(LᵀA.U,2))
        sjk = 0
        for i in eachindex(axes(LᵀA.U,1))
            sjk += LᵀA.U[i,j] * Lᵀd[i,k]
        end
        s += sjk^2/2
    end
    s
end
lnlike(LᵀA, Lᵀd) = try sum([𝔣logL(LᵀA[i], view(Lᵀd,:,i,:)) for i=1:size(Lᵀd,2)]) catch; -Inf end

# build to-be-minimized function
function build_target(comps, ν, N⁻¹, Lᵀd; mm=nothing)
    mm = ifelse(mm == nothing, mixing_matrix(comps, ν; folder=fold(comps)), mm)
    f(pars) = try 𝔣LᵀA(N⁻¹, mm(pars)) |> LᵀA -> -lnlike(LᵀA, Lᵀd) catch; -Inf end
    g!(storage, pars) = FiniteDiff.finite_difference_jacobian(f, pars) |> res->storage[:]=res
    f, g!
end

function compsep(comps, ν, N⁻¹, d; x₀=[-3.,1.54,20.], use_jac=false, algo=BFGS())
    Lᵀd = N⁻¹.^0.5 .* d
    f, g! = build_target(comps, ν, N⁻¹, Lᵀd)
    use_jac ? optimize(f, g!, x₀, algo) : optimize(f, x₀, algo)
end

# utility functions
parse_sigs(comps; nskip=1) = map(c->methods(c)[1].nargs-1-nskip, comps) |> cumsum |> x->[1;x[1:end-1].+1;;x] |> x->map((b,e)->range(b,e),x[:,1],x[:,2])
fold(comps; nskip=1) = (sigs=parse_sigs(comps, nskip=nskip); params -> (params |> p -> map(sl->p[sl], sigs)))
unfold(params) = params |> x -> vcat(x...)
