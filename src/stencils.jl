"""
    StencilSet(points::AbstractMatrix{<:Real}; order::Integer = 2,
               k::Integer = 2 * minimum_neighbours(N, order), kmax::Integer = 2k,
               threaded::Bool = true)

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

Where a point's `k` nearest neighbours cannot meet all moment conditions
independently, its stencil grows one neighbour at a time, up to `kmax`
neighbours, until they can. This happens at boundaries, where the nearest
points can lie on too few lines or planes, for example on the faces, edges
and corners of a regular grid. All other points keep exactly `k` neighbours.

With `threaded = true`, the weights of different points are computed in
parallel on the threads Julia was started with (see `Threads.nthreads()`).
Set `threaded = false` when calling `StencilSet` from code that is already
parallel.

Throws an `ArgumentError` naming the point if a point's neighbours cannot
satisfy the moment conditions even after growing to `kmax` neighbours. This
happens when the points lie on a curve or surface instead of filling their
space.

# Fields
- `points::Matrix{T}`: the point coordinates, one column per point.
- `neighbours::Vector{Vector{Int}}`: indices of each point's neighbours, nearest first, excluding the point itself: `k` of them, or more where the stencil was grown.
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
                    kmax::Integer = 2k,
                    threaded::Bool = true) where {T<:AbstractFloat}
    N, npoints = size(points)                    # dimension and number of points
    kmin = minimum_neighbours(N, order)          # neighbours needed to fix all moment conditions
    k >= kmin || throw(ArgumentError("k = $k is below the minimum $kmin for order $order in $N dimensions"))
    k < npoints || throw(ArgumentError("k = $k must be smaller than the number of points ($npoints)"))
    kmax >= k || throw(ArgumentError("kmax = $kmax must be at least k = $k"))
    kgrow = min(kmax, npoints - 1)               # the largest stencil the points allow

    pts = Matrix{T}(points)                      # own a plain copy of the coordinates
    tree = KDTree(pts)                           # spatial tree over the columns, shared read-only by all tasks

    neighbours = Vector{Vector{Int}}(undef, npoints)
    weights = Vector{Vector{T}}(undef, npoints)

    # Neighbours and weights of every point in `range`. The range queries the
    # shared tree for its own points only, and one workspace and one offset
    # buffer serve the whole range, so the only allocations per point are the
    # search results and the neighbour indices and weights that are kept, plus
    # those of the few stencils that are grown. Each point is written by exactly
    # one call, so ranges can run in parallel.
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
            try
                a = Vector{T}(undef, k)          # this point's weights, kept
                independence, residual = coefficients!(a, workspace, dX, order)
                if independence < _min_independence(T) && kgrow > k
                    # some conditions are (nearly) dependent on these neighbours: try larger stencils
                    nbrs, a, residual = _grow_stencil(tree, pts, i, order, k, kgrow, nbrs, a, residual)
                end
                _require_moments(residual)       # the first two conditions must hold
                neighbours[i] = nbrs
                weights[i] = a
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError("point $i: " * err.msg))  # say which point's stencil failed
            end
        end
        return nothing
    end

    if threaded && Threads.nthreads() > 1
        nchunks = min(npoints, 4 * Threads.nthreads())          # a few chunks per thread, for load balance
        chunks = Iterators.partition(1:npoints, cld(npoints, nchunks))  # contiguous ranges of points
        tasks = [Threads.@spawn fill_range!(chunk) for chunk in chunks] # one task per range
        try
            foreach(wait, tasks)                                # wait for all
        catch err
            # an error inside a task arrives wrapped in a TaskFailedException; rethrow the
            # original, so that a threaded build fails exactly as a serial one does
            err isa TaskFailedException ? throw(err.task.exception) : rethrow()
        end
    else
        fill_range!(1:npoints)
    end

    return StencilSet{T}(pts, neighbours, weights, Int(order))
end

# Integer or other non-float coordinates are converted to floating point
StencilSet(points::AbstractMatrix{<:Real}; kwargs...) = StencilSet(float.(points); kwargs...)

# The independence of the moment conditions below which a stencil is grown (see
# `_minimum_norm_solve!`). Below it, a condition is dropped or nearly dependent
# on the others, which makes the weights large; well-shaped stencils have an
# independence of 0.05 or more
_min_independence(::Type{T}) where {T<:AbstractFloat} = max(T(1e-6), sqrt(eps(T)))

# Grow the stencil of point i one neighbour at a time, from k + 1 up to kmax
# neighbours, and return the first one whose moment conditions are independent,
# as (neighbours, weights, residual). If none is, return the smallest stencil,
# starting with the given one, that satisfies the first two moment conditions,
# or the given one if none does, so that `_require_moments` reports it
function _grow_stencil(tree, pts::Matrix{T}, i::Integer, order::Integer, k::Integer, kmax::Integer,
                       nbrs::Vector{Int}, a::Vector{T}, residual::T) where {T<:AbstractFloat}
    N = size(pts, 1)                                 # dimension
    candidates, _ = knn(tree, pts[:, i], kmax + 1, true)  # kmax + 1 nearest (itself included), sorted
    others = filter(!=(i), candidates)               # the nearest points other than i itself
    fallback = (nbrs, a, residual)                   # the stencil to return if none becomes independent
    fallback_holds = residual <= _max_residual(T)    # whether it satisfies the first two conditions
    for M in k+1:min(kmax, length(others))
        grown = others[1:M]                          # the M nearest neighbours
        dX = pts[:, grown] .- pts[:, i]              # their offsets Δx_k = x_k - x_0
        w = Vector{T}(undef, M)                      # their weights
        independence, res = coefficients!(w, _Workspace{T}(N, M, order), dX, order)
        independence >= _min_independence(T) && return grown, w, res  # independent: done
        if !fallback_holds && res <= _max_residual(T)
            fallback = (grown, w, res)               # the smallest stencil that satisfies the first two conditions
            fallback_holds = true
        end
    end
    return fallback
end

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