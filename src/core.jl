using TensorOperations

# constants
h_over_k = 0.04799243073366221
GHz = 1

# seds
cmb(ν) = @. ν*0+1
sync(ν, β; running=0, νₚ=30*GHz) = @. (ν/ν₀)^(β + running*log(ν/νₚ))
dust(ν, βd, Td; ν₀=150*GHz) = @. (exp(ν₀/Td*h_over_k)-1) / (exp(ν/Td*h_over_k)-1)*(ν/m.ν₀)^(1+βd)

# mixing matrix
mixing_matrix(comps, ν, pars) = hcat([c(ν, par...) for (c, par) in zip(comps, pars)]...)

# log-likelihood
function lnlike(A, N⁻¹, d)
    @tensoropt rhs[i,k,l] := A'[i,j] * N⁻¹(d)[j,k,l]
    s = rhs .* 0 .+ 1
    @tensoropt As[i,k,l] := A[i,j] * s[j,k,l]
    @tensoropt div[i,k,l] := A'[i,j] * N⁻¹(As)[j,k,l]
    s[:] = rhs ./ div
    @tensoropt -0.5 * rhs[i,j,k] * s[i,j,k]
end

# utility functions
parse_sigs(comps) = map(comp->methods(comp)[1].nargs-2, comps)
fold(params, sigs) = params |> copy |> p -> map(n->[popfirst!(p) for _ in 1:n], sigs)
unfold(params) = params |> x -> vcat(x...)
