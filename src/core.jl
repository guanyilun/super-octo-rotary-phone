using TensorOperations, LinearAlgebra, Optim

# constants
h_over_k = 0.04799243073366221
GHz = 1.
Tcmb = 2.726

# unit conversion
KRJ_to_KCMB(ν) = ν/Tcmb*h_over_k |> x-> (exp.(x) .- 1).^2 / (exp.(x).*x.^2)  # everything in K_CMB

cmb(ν) = @. ν*0+1
sync(ν, β; running=0, νₚ=70*GHz, ν₀=20*GHz) = @. (ν/ν₀)^(β + running*log(ν/νₚ)) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)
dust(ν, βd, Td; ν₀=150*GHz) = @. (exp(ν₀/Td*h_over_k)-1) / (exp(ν/Td*h_over_k)-1)*(ν/ν₀)^(1+βd) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)

mixing_matrix(comps, ν; folder) = pars -> folder(pars) |> pars-> hcat([c(ν,p...) for (c, p) in zip(comps, pars)]...)

𝔣LᵀA(N⁻¹, A) = N⁻¹.^(1/2) |> L -> svd(L.*A)
𝔣logL(LᵀA, Lᵀd) = LᵀA.U' * Lᵀd |> Uᵀd -> sum(Uᵀd.^2)/2
lnlike(A, N⁻¹, Lᵀd) = try sum(𝔣logL(𝔣LᵀA(N⁻¹[:,i],A), Lᵀd[:,i,:]) for i=1:size(obs,2)) catch; -Inf end

# build to-be-minimized function
function build_target(comps, ν, N⁻¹, Lᵀd; folder)
    pars -> -lnlike(mixing_matrix(comps, ν; folder=folder)(pars), N⁻¹, Lᵀd)
end

function compsep(comps, ν, N⁻¹, d; x₀=[-3.,1.54,20.])
    Lᵀd = N⁻¹.^(1/2) .* d
    folder = fold(comps)
    f = build_target(comps, ν, N⁻¹, Lᵀd; folder=folder)
    optimize(f, x₀, BFGS())
end

# utility functions
parse_sigs(comps) = map(c->methods(c)[1].nargs-2, comps) |> cumsum |> x->[1;x[1:end-1].+1;;x] |> x->map((b,e)->range(b,e),x[:,1],x[:,2])
fold(comps) = (sigs=parse_sigs(comps); params -> (params |> p -> map(sl->p[sl], sigs)))
unfold(params) = params |> x -> vcat(x...)