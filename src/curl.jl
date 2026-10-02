"""
    curl(stencils::StencilSet, values::AbstractArray) -> Array
    curl(stencils::StencilSet, values::AbstractArray, i::Integer) -> Array

Estimate the curl `∇×Y = Σ_k a_k Δx_k × ΔY_k` of a vector or tensor field
`Y` in three dimensions, taking the cross product with the field's first
index.

`values` uses the same layout as in `gradient`, with points along the last
dimension, and its first dimension must be 3. The result has the same size
as `values`, with `result[i, r..., p] = ε_ijk ∂Y_kr.../∂x_j` at point `p`.
The three-argument method returns the curl at point `i` alone, with the
size of one point's value.

The curl is a contraction of the gradient and has the same accuracy: exact
for linear fields with `order = 1` stencils, and also for quadratic fields
with `order = 2` stencils.
"""
function curl(stencils::StencilSet, values::AbstractArray{<:Real})
    N, npoints = size(stencils.points)               # dimension and number of points
    fieldsize = _fieldsize(values, npoints)          # size of the field value at one point
    _check_curl(fieldsize, N)                        # 3D, with a spatial first field index
    V = reshape(values, :, npoints)                  # one column per point, components flattened
    R = float(promote_type(eltype(stencils.points), eltype(values)))  # result element type
    result = Array{R}(undef, size(V, 1), npoints)
    for i in 1:npoints
        result[:, i] = vec(_curl_flat(stencils, V, i))
    end
    return reshape(result, fieldsize..., npoints)    # same layout as the input field
end

function curl(stencils::StencilSet, values::AbstractArray{<:Real}, i::Integer)
    N, npoints = size(stencils.points)               # dimension and number of points
    fieldsize = _fieldsize(values, npoints)          # size of the field value at one point
    _check_curl(fieldsize, N)                        # 3D, with a spatial first field index
    V = reshape(values, :, npoints)                  # one column per point, components flattened
    return reshape(_curl_flat(stencils, V, i), fieldsize...)
end

# The curl needs three dimensions and a field whose first dimension is 3
function _check_curl(fieldsize::Tuple, N::Integer)
    N == 3 || throw(ArgumentError("curl is only defined in 3 dimensions, but the points have $N"))
    (!isempty(fieldsize) && fieldsize[1] == 3) || throw(DimensionMismatch(
        "curl needs a field whose first dimension is 3, but one point's value has size $fieldsize"))
    return nothing
end

# ε_ijk ∂Y_k.../∂x_j at point i, as a 3 × (remaining components) matrix
function _curl_flat(stencils::StencilSet, V::AbstractMatrix, i::Integer)
    G = _gradient_flat(stencils, V, i)               # 3 × components
    G3 = reshape(G, 3, 3, :)                         # [derivative j, first field index k, remaining]
    c1 = G3[2, 3, :] .- G3[3, 2, :]                  # ∂Y_3/∂x_2 - ∂Y_2/∂x_3
    c2 = G3[3, 1, :] .- G3[1, 3, :]                  # ∂Y_1/∂x_3 - ∂Y_3/∂x_1
    c3 = G3[1, 2, :] .- G3[2, 1, :]                  # ∂Y_2/∂x_1 - ∂Y_1/∂x_2
    return permutedims(hcat(c1, c2, c3))             # 3 × remaining
end