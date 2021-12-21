using TensorOperations
using Optim

# constants
h_over_k = 0.04799243073366221
GHz = 1.
Tcmb = 2.726

# units
KRJ_to_KCMB(ν) = @. (exp(ν/Tcmb*h_over_k)-1)^2 / (exp(h_over_k*ν/Tcmb)*(h_over_k*ν/Tcmb)^2)

# seds
cmb(ν) = @. ν*0+1
sync(ν, β; running=0, νₚ=70*GHz, ν₀=20*GHz) = @. (ν/ν₀)^(β + running*log(ν/νₚ)) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)
dust(ν, βd, Td; ν₀=150*GHz) = @. (exp(ν₀/Td*h_over_k)-1) / (exp(ν/Td*h_over_k)-1)*(ν/ν₀)^(1+βd) * KRJ_to_KCMB(ν) / KRJ_to_KCMB(ν₀)

mixing_matrix(comps, ν, pars) = hcat([c(ν, par...) for (c, par) in zip(comps, pars)]...)

# log-likelihood
function lnlike(A, N⁻¹, N⁻¹d)
    @tensoropt rhs[i,k,l] := A'[i,j] * N⁻¹d[j,k,l]
    try @tensoropt 0.5 * rhs[i,k,l] * inv(A' * N⁻¹(A))[i,j] * rhs[j,k,l] catch e; -Inf end
end

# utility functions
parse_sigs(comps) = map(comp->methods(comp)[1].nargs-2, comps)
fold(params, sigs) = params |> copy |> p -> map(n->[popfirst!(p) for _ in 1:n], sigs)
unfold(params) = params |> x -> vcat(x...)

function build_target(comps, freqs, N⁻¹, N⁻¹d)
    sigs = parse_sigs(comps)
    pars -> fold(pars, sigs) |> pars->mixing_matrix(comps, freqs, pars) |> A->-lnlike(A, N⁻¹, N⁻¹d)
end

function compsep(comps, freqs, N⁻¹, N⁻¹d; x₀=[-3.,1.54,20.])
    target = build_target(comps, freqs, N⁻¹, N⁻¹d)
    optimize(target, x₀, LBFGS()) |> Optim.minimizer
end