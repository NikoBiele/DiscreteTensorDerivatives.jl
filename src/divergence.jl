"""
    divergence(stencils::StencilSet, values::AbstractArray) -> Array
    divergence(stencils::StencilSet, values::AbstractArray, i::Integer)

Estimate the divergence `∇·Y = Σ_k a_k Δx_k · ΔY_k` of a vector or tensor
field `Y`, contracting the derivative with the field's first index.

`values` uses the same layout as in `gradient`, with points along the last
dimension, and its first dimension must equal the spatial dimension `N`.
For a vector field (`N × P`), the result is a length-`P` vector. For a
matrix field (`N × m × P`), it is an `m × P` matrix with
`result[j, p] = Σ_a ∂Y_aj/∂x_a` at point `p`. The three-argument method
returns the value at point `i` alone: a scalar for a vector field, and an
array of size `(dims[2:end]...)` otherwise.

The divergence is a contraction of the gradient and has the same accuracy:
exact for linear fields with `order = 1` stencils, and also for quadratic
fields with `order = 2` stencils.
"""
function divergence(stencils::StencilSet, values::AbstractArray{<:Real})
    N, npoints = size(stencils.points)               # dimension and number of points
    fieldsize = _fieldsize(values, npoints)          # size of the field value at one point
    _check_divergence(fieldsize, N)                  # first field index must be spatial
    restsize = fieldsize[2:end]                      # field dimensions that survive the contraction
    V = reshape(values, :, npoints)                  # one column per point, components flattened
    R = float(promote_type(eltype(stencils.points), eltype(values)))  # result element type
    result = Array{R}(undef, prod(restsize), npoints)
    for i in 1:npoints
        result[:, i] = _divergence_flat(stencils, V, i, N)
    end
    return reshape(result, restsize..., npoints)     # a plain vector for vector fields
end

function divergence(stencils::StencilSet, values::AbstractArray{<:Real}, i::Integer)
    N, npoints = size(stencils.points)               # dimension and number of points
    fieldsize = _fieldsize(values, npoints)          # size of the field value at one point
    _check_divergence(fieldsize, N)                  # first field index must be spatial
    restsize = fieldsize[2:end]                      # field dimensions that survive the contraction
    V = reshape(values, :, npoints)                  # one column per point, components flattened
    d = _divergence_flat(stencils, V, i, N)          # contracted gradient at point i
    return isempty(restsize) ? d[1] : reshape(d, restsize...)  # scalar for vector fields
end

# The divergence needs at least a vector field whose first dimension is spatial
function _check_divergence(fieldsize::Tuple, N::Integer)
    (!isempty(fieldsize) && fieldsize[1] == N) || throw(DimensionMismatch(
        "divergence needs a field whose first dimension is the spatial dimension $N, " *
        "but one point's value has size $fieldsize"))
    return nothing
end

# Σ_a ∂Y_a.../∂x_a at point i: the gradient with its first two indices contracted
function _divergence_flat(stencils::StencilSet, V::AbstractMatrix, i::Integer, N::Integer)
    G = _gradient_flat(stencils, V, i)               # N × components
    G3 = reshape(G, N, N, :)                         # [derivative a, first field index j, remaining]
    return [sum(G3[a, a, r] for a in 1:N) for r in 1:size(G3, 3)]  # contract a with j
end