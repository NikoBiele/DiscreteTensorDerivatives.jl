"""
    laplacian(stencils::StencilSet, values::AbstractArray) -> Array
    laplacian(stencils::StencilSet, values::AbstractArray, i::Integer)

Estimate the Laplacian `∇²Y = 2 Σ_k a_k ΔY_k` of a scalar, vector or tensor
field `Y`, applied to each component.

`values` uses the same layout as in `gradient`, with points along the last
dimension, and the result has the same size as `values`. The three-argument
method returns the Laplacian at point `i` alone: a scalar for a scalar
field, and an array with the size of one point's value otherwise.

The estimate is exact for quadratic fields with `order = 1` stencils, and
also for cubic fields with `order = 2` stencils. For smooth fields, it
converges at first and second order in the stencil size, respectively.
"""
function laplacian(stencils::StencilSet, values::AbstractArray{<:Real})
    N, npoints = size(stencils.points)               # dimension and number of points
    fieldsize = _fieldsize(values, npoints)          # size of the field value at one point
    V = reshape(values, :, npoints)                  # one column per point, components flattened
    R = float(promote_type(eltype(stencils.points), eltype(values)))  # result element type
    result = Array{R}(undef, size(V, 1), npoints)
    for i in 1:npoints
        result[:, i] = _laplacian_flat(stencils, V, i)
    end
    return reshape(result, fieldsize..., npoints)    # same layout as the input field
end

function laplacian(stencils::StencilSet, values::AbstractArray{<:Real}, i::Integer)
    N, npoints = size(stencils.points)               # dimension and number of points
    fieldsize = _fieldsize(values, npoints)          # size of the field value at one point
    V = reshape(values, :, npoints)                  # one column per point, components flattened
    L = _laplacian_flat(stencils, V, i)              # Laplacian of every component at point i
    return isempty(fieldsize) ? L[1] : reshape(L, fieldsize...)  # scalar for scalar fields
end

# 2 Σ_k a_k ΔY_k at point i, one entry per field component
function _laplacian_flat(stencils::StencilSet, V::AbstractMatrix, i::Integer)
    nbrs = stencils.neighbours[i]                    # neighbour indices of point i
    dY = V[:, nbrs] .- V[:, i]                       # value differences ΔY_k, components × k
    return 2 .* (dY * stencils.weights[i])           # components
end