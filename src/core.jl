using TensorOperations, LinearAlgebra, Optim

# constants
h_over_k = 0.04799243073366221
GHz = 1.  # matched with h_over_k
Tcmb = 2.726

# unit conversion
KRJ_to_KCMB(ν) = @. (exp(ν/Tcmb*h_over_k)-1)^2 / (exp(h_over_k*ν/Tcmb)*(h_over_k*ν/Tcmb)^2)  # everything in K_CMB

cmb(ν) = @. ν*0+1
sync(ν, β; running=0, νₚ=70*GHz, ν₀=20*GHz) = @. (ν/ν₀)^(β + running*log(ν/νₚ)) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)
dust(ν, βd, Td; ν₀=150*GHz) = @. (exp(ν₀/Td*h_over_k)-1) / (exp(ν/Td*h_over_k)-1)*(ν/ν₀)^(1+βd) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)

mixing_matrix(comps, ν) = pars -> fold(pars, comps) |> pars->hcat([c(ν, p...) for (c, p) in zip(comps, pars)]...)

𝔣LᵀA(N⁻¹, A) = N⁻¹.^(1/2) |> L -> svd(L.*A)
𝔣div(LᵀA)  = LᵀA.V * Diagonal(LᵀA.S.^2) * LᵀA.V
𝔣div⁻¹(LᵀA) = LᵀA.V * Diagonal(LᵀA.S.^-2) * LᵀA.V
𝔣logL(LᵀA, Lᵀd) = LᵀA.U' * Lᵀd |> Uᵀd -> sum(Uᵀd.^2)/2
lnlike(A, N⁻¹, Lᵀd) = try sum(𝔣logL(𝔣LᵀA(N⁻¹[:,i],A), Lᵀd[:,i,:]) for i=1:size(obs,2)) catch; -Inf end

# build to-be-minimized function
function build_target(comps, ν, N⁻¹, N⁻¹d)
    A = mixing_matrix(comps, ν)
    pars -> A(pars) |> A->-lnlike(A, N⁻¹, N⁻¹d)
end

function compsep(comps, ν, N⁻¹, d; x₀=[-3.,1.54,20.])
    Lᵀd = N⁻¹.^(1/2) .* d 
    target = build_target(comps, ν, N⁻¹, Lᵀd)
    optimize(target, x₀, LBFGS())
end

# utility functions
parse_sigs(comps) = map(comp->methods(comp)[1].nargs-2, comps)
fold(params, comps) = params |> copy |> p -> map(n->[popfirst!(p) for _ in 1:n], parse_sigs(comps))
unfold(params) = params |> x -> vcat(x...)