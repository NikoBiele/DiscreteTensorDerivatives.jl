"""
    gradient(stencils::StencilSet, values::AbstractArray) -> Array
    gradient(stencils::StencilSet, values::AbstractArray, i::Integer) -> Array

Estimate the gradient `∇⊗Y = Σ_k a_k Δx_k ⊗ ΔY_k` of a scalar, vector or
tensor field `Y` sampled at the stencil points.

`values` holds one field value per point along its last dimension, so it is
a length-`P` vector for a scalar field, an `n × P` matrix for a vector
field, an `n × m × P` array for a matrix field, and so on.

The result puts the derivative index first and the point index last. For a
field with values of size `(dims...)`, it has size `(N, dims..., P)`, with
`result[a, j..., p] = ∂Y_j/∂x_a` at point `p`. For a vector field this is
the transpose of the usual Jacobian. The three-argument method returns the
gradient at point `i` alone, with size `(N, dims...)`.

The estimate is exact for linear fields with `order = 1` stencils, and also
for quadratic fields with `order = 2` stencils. For smooth fields, it
converges at first and second order in the stencil size, respectively.
"""
function gradient(stencils::StencilSet, values::AbstractArray{<:Real})
    N, npoints = size(stencils.points)               # dimension and number of points
    fieldsize = _fieldsize(values, npoints)          # size of the field value at one point
    V = reshape(values, :, npoints)                  # one column per point, components flattened
    R = float(promote_type(eltype(stencils.points), eltype(values)))  # result element type
    result = Array{R}(undef, N, size(V, 1), npoints)
    for i in 1:npoints
        result[:, :, i] = _gradient_flat(stencils, V, i)
    end
    return reshape(result, N, fieldsize..., npoints)  # restore the field's own dimensions
end

function gradient(stencils::StencilSet, values::AbstractArray{<:Real}, i::Integer)
    N, npoints = size(stencils.points)               # dimension and number of points
    fieldsize = _fieldsize(values, npoints)          # size of the field value at one point
    V = reshape(values, :, npoints)                  # one column per point, components flattened
    return reshape(_gradient_flat(stencils, V, i), N, fieldsize...)
end

# Size of one point's field value, after checking that the last dimension counts the points
function _fieldsize(values::AbstractArray, npoints::Integer)
    size(values, ndims(values)) == npoints || throw(DimensionMismatch(
        "the last dimension of values has length $(size(values, ndims(values))), " *
        "but there are $npoints points"))
    return size(values)[1:end-1]
end

# Σ_k a_k Δx_k ΔY_kᵀ at point i, for flattened field values V (components × points)
function _gradient_flat(stencils::StencilSet, V::AbstractMatrix, i::Integer)
    nbrs = stencils.neighbours[i]                         # neighbour indices of point i
    dX = stencils.points[:, nbrs] .- stencils.points[:, i]  # offsets Δx_k, N × k
    dY = V[:, nbrs] .- V[:, i]                            # value differences ΔY_k, components × k
    return dX * (stencils.weights[i] .* transpose(dY))    # N × components
end