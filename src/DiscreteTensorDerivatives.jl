"""
    DiscreteTensorDerivatives

Gradient, divergence, curl and Laplacian of scalar, vector and tensor fields
on scattered points, computed as weighted sums of differences over each
point's nearest neighbours.
"""
module DiscreteTensorDerivatives

using LinearAlgebra
using NearestNeighbors

include("coefficients.jl")   # weights a_k from a single point's neighbour offsets
include("stencils.jl")       # StencilSet: neighbours and weights for every point
include("gradient.jl")       # ∇⊗Y for fields of any rank
include("divergence.jl")     # ∇·Y
include("curl.jl")           # ∇×Y (3D only)
include("laplacian.jl")      # ∇²Y

export StencilSet, minimum_neighbours, gradient, divergence, curl, laplacian

end