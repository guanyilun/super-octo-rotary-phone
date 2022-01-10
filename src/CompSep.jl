module CompSep

using TensorOperations
using LinearAlgebra
using Optim
using FiniteDiff
using LoopVectorization
using NumericalIntegration
using Octavian

include("core.jl")
export cmb, sync, dust, mixing_matrix, compsep

end
