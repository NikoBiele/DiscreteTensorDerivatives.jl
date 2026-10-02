"""
    StencilSet(points::AbstractMatrix{<:Real}; order::Integer = 2,
               k::Integer = 2 * minimum_neighbours(N, order), threaded::Bool = true)

Find the `k` nearest neighbours of every point and compute their stencil
weights, once, for use by `gradient`, `divergence`, `curl` and `laplacian`.

`points` is an `N × P` matrix with one column per point in `N` dimensions.

With `order = 2`, the default, the weights cancel the leading error term,
which makes `gradient`, `divergence`, `curl` and `laplacian` second-order
accurate on scattered points. `order = 1` gives first-order accurate
operators from smaller stencils, which suits small point clouds or cheaper
builds.

The minimum number of neighbours is `N(N+3)/2 + N(N+1)(N+2)/6` for
`order = 2` (9 in 2D, 19 in 3D) and `N(N+3)/2` for `order = 1` (5 in 2D, 9
in 3D). The default `k` is twice the minimum, because stencils at the
minimum size are ill-conditioned.

With `threaded = true`, the weights of different points are computed in
parallel on the threads Julia was started with (see `Threads.nthreads()`).
Set `threaded = false` when calling `StencilSet` from code that is already
parallel.

# Fields
- `points::Matrix{T}`: the point coordinates, one column per point.
- `neighbours::Vector{Vector{Int}}`: indices of each point's `k` neighbours, nearest first, excluding the point itself.
- `weights::Vector{Vector{T}}`: the weights `a_k` of each point's neighbours, in the same order.
- `order::Int`: the accuracy order the weights were computed for.

# Example
```julia
points = rand(3, 1000)                   # 1000 random points in the unit cube
stencils = StencilSet(points)            # second order, 38 neighbours per point
stencils1 = StencilSet(points; order=1)  # first order, 18 neighbours per point
```
"""
struct StencilSet{T<:AbstractFloat}
    points::Matrix{T}
    neighbours::Vector{Vector{Int}}
    weights::Vector{Vector{T}}
    order::Int
end

function StencilSet(points::AbstractMatrix{T}; order::Integer = 2,
                    k::Integer = 2 * minimum_neighbours(size(points, 1), order),
                    threaded::Bool = true) where {T<:AbstractFloat}
    N, npoints = size(points)                    # dimension and number of points
    kmin = minimum_neighbours(N, order)          # neighbours needed to fix all moment conditions
    k >= kmin || throw(ArgumentError("k = $k is below the minimum $kmin for order $order in $N dimensions"))
    k < npoints || throw(ArgumentError("k = $k must be smaller than the number of points ($npoints)"))

    pts = Matrix{T}(points)                      # own a plain copy of the coordinates
    tree = KDTree(pts)                           # spatial tree over the columns, shared read-only by all tasks

    neighbours = Vector{Vector{Int}}(undef, npoints)
    weights = Vector{Vector{T}}(undef, npoints)

    # Neighbours and weights of every point in `range`. The range queries the
    # shared tree for its own points only, and one workspace and one offset
    # buffer serve the whole range, so the only allocations per point are the
    # search results and the neighbour indices and weights that are kept. Each
    # point is written by exactly one call, so ranges can run in parallel.
    function fill_range!(range)
        candidates, _ = knn(tree, pts[:, range], k + 1, true)  # k + 1 nearest per point (itself included), sorted
        workspace = _Workspace{T}(N, k, order)   # scratch arrays for the weights
        dX = Matrix{T}(undef, N, k)              # offset buffer
        for (local_index, i) in enumerate(range)
            nbrs = Vector{Int}(undef, k)         # this point's neighbours, kept
            count = 0                            # neighbours found so far
            for j in candidates[local_index]     # nearest first
                if j != i && count < k           # skip the point itself, keep the first k others
                    count += 1
                    nbrs[count] = j
                end
            end
            for q in 1:k, d in 1:N
                dX[d, q] = pts[d, nbrs[q]] - pts[d, i]  # offsets Δx_k = x_k - x_0
            end
            neighbours[i] = nbrs
            weights[i] = coefficients!(Vector{T}(undef, k), workspace, dX, order)  # this point's weights, kept
        end
        return nothing
    end

    if threaded && Threads.nthreads() > 1
        nchunks = min(npoints, 4 * Threads.nthreads())          # a few chunks per thread, for load balance
        chunks = Iterators.partition(1:npoints, cld(npoints, nchunks))  # contiguous ranges of points
        tasks = [Threads.@spawn fill_range!(chunk) for chunk in chunks] # one task per range
        foreach(wait, tasks)                                    # wait for all, rethrowing any error
    else
        fill_range!(1:npoints)
    end

    return StencilSet{T}(pts, neighbours, weights, Int(order))
end

# Integer or other non-float coordinates are converted to floating point
StencilSet(points::AbstractMatrix{<:Real}; kwargs...) = StencilSet(float.(points); kwargs...)

"""
    minimum_neighbours(N::Integer, order::Integer = 1) -> Int

The smallest number of neighbours for which the moment conditions of the
given `order` can hold exactly in `N` dimensions: `N(N+3)/2` for order 1,
plus `N(N+1)(N+2)/6` for order 2.
"""
function minimum_neighbours(N::Integer, order::Integer = 1)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    first_and_second = N * (N + 3) ÷ 2           # first moment (N) and second moment (N(N+1)/2)
    third = N * (N + 1) * (N + 2) ÷ 6            # distinct entries of the symmetric third moment
    return order == 1 ? first_and_second : first_and_second + third
end