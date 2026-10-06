"""
    coefficients(dX::AbstractMatrix{T}, order::Integer = 1) where {T<:AbstractFloat} -> Vector{T}

Compute the stencil weights `a` for one central point. `dX` is the `N × M`
matrix whose columns are the offsets `Δx_k = x_k - x_0` of the `M`
neighbours from the central point in `N` dimensions.

For `order = 1`, the weights satisfy

    Σ_k a_k Δx_k = 0    and    Σ_k a_k Δx_k Δx_kᵀ = I,

which needs `M ≥ N(N+3)/2` neighbours. For `order = 2`, they also satisfy,
where the neighbours allow it,

    Σ_k a_k Δx_k ⊗ Δx_k ⊗ Δx_k = 0,

which removes the leading error of the gradient and the Laplacian and needs
`M ≥ N(N+3)/2 + N(N+1)(N+2)/6` neighbours. Among all weights satisfying the
conditions, the returned `a` has minimum norm.

The conditions are linear in `a`, so they are solved directly as one small
system. This gives the same weights as the original Gram-matrix formulation,
`a = P ((G ∘ G) P)⁺ diag(G)`, because `G ∘ G` carries exactly the
information of the second-moment equations.

Throws an `ArgumentError` if the neighbours cannot satisfy the first two
conditions, which happens when they lie on a curve or surface, or on too few
layers of a regular grid; the derivatives would then be meaningless. The
third-moment conditions are not checked: where the neighbours cannot meet
them, as on the faces of a regular grid, the stencil remains correct to first
order.
"""
function coefficients(dX::AbstractMatrix{T}, order::Integer = 1) where {T<:AbstractFloat}
    N, M = size(dX)                                  # dimension and number of neighbours
    a = Vector{T}(undef, M)                          # the weights to return
    workspace = _Workspace{T}(N, M, order)           # scratch arrays for a single call
    _, residual = coefficients!(a, workspace, dX, order)  # fills a; the independence is not needed here
    _require_moments(residual)                       # the first two conditions must hold
    return a
end

# Scratch arrays for computing the weights of one stencil, reusable across
# stencils with the same dimension, number of neighbours and order
struct _Workspace{T<:AbstractFloat}
    At::Matrix{T}   # M × m: one column per moment condition, one row per neighbour
    b::Vector{T}    # m: right-hand side of the moment conditions
    y::Vector{T}    # m: solution coefficients in the orthonormalized basis
end

function _Workspace{T}(N::Integer, M::Integer, order::Integer) where {T<:AbstractFloat}
    m = minimum_neighbours(N, order)                 # number of moment conditions
    return _Workspace{T}(Matrix{T}(undef, M, m), Vector{T}(undef, m), Vector{T}(undef, m))
end

# The largest violation of the first- and second-moment conditions accepted, in
# the scaled units the conditions are solved in. Stencils of points that fill
# the space satisfy them to about 1e-10 in Float64 and 1e-5 in Float32; stencils
# whose neighbours lie on a curve or surface, or on too few layers of a grid,
# miss them by 0.1 or more
_max_residual(::Type{T}) where {T<:AbstractFloat} = max(T(1e-6), sqrt(eps(T)))

# Throw if the weights violate the first- and second-moment conditions by more
# than `_max_residual`. These conditions make the operators correct; the weights
# cannot satisfy them when the neighbours don't span every direction, and the
# solver then silently drops them. A NaN residual also throws
function _require_moments(residual::T) where {T<:AbstractFloat}
    residual <= _max_residual(T) || throw(ArgumentError(
        "the neighbours cannot satisfy the moment conditions (residual $(round(residual; sigdigits = 2))); " *
        "they may lie on a curve or surface, or too few of them span every direction: " *
        "try a larger k, or give the points in coordinates of their own dimension"))
    return nothing
end

# In-place version of `coefficients`: write the weights into `a`, using the
# arrays of `workspace` as scratch space, so that nothing is allocated. Returns
# two numbers about this stencil and leaves it to the caller to act on them: the
# independence of the moment conditions (see `_minimum_norm_solve!`), typically
# 0.05 or more for a well-shaped stencil and zero when a condition had to be
# dropped, and the largest violation of the first- and second-moment conditions
# (see `_moment_residual`), which `_require_moments` checks
function coefficients!(a::AbstractVector{T}, workspace::_Workspace{T},
                       dX::AbstractMatrix{T}, order::Integer) where {T<:AbstractFloat}
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    N, M = size(dX)                                  # dimension and number of neighbours
    (length(a) == M && size(workspace.At) == (M, minimum_neighbours(N, order))) ||
        throw(DimensionMismatch("weights or workspace do not match $M neighbours in $N dimensions at order $order"))
    h = maximum(norm, eachcol(dX))                   # stencil radius: the largest offset length
    h > 0 || throw(ArgumentError("all neighbours coincide with the central point"))
    _moment_system!(workspace, dX, h, order)         # fill At and b with the moment conditions
    independence = _minimum_norm_solve!(a, workspace)  # minimum-norm weights for the scaled offsets
    residual = _moment_residual(dX, a, h)            # violation of the conditions that make the operators correct
    a ./= h^2                                        # undo the scaling, since Σ a_k Δx_k Δx_kᵀ = I makes a ∝ 1/h²
    return independence, residual
end

# The largest violation of the first- and second-moment conditions by the
# weights `a`, in scaled units (offsets divided by h, weights not yet divided by
# h²). These two conditions make the operators correct; the third-moment
# conditions of order 2 only raise their accuracy, and are left out, since
# stencils on the faces of a regular grid cannot meet them yet remain correct to
# first order. A NaN anywhere makes the result NaN.
function _moment_residual(dX::AbstractMatrix{T}, a::AbstractVector{T}, h::T) where {T<:AbstractFloat}
    N, M = size(dX)                                  # dimension and number of neighbours
    worst = zero(T)                                  # the largest violation so far
    for i in 1:N                                     # Σ a_k S_ik = 0
        moment = zero(T)
        for k in 1:M
            moment += a[k] * (dX[i, k] / h)
        end
        worst = _larger_violation(worst, abs(moment))
    end
    for i in 1:N, j in i:N                           # Σ a_k S_ik S_jk = δ_ij
        moment = zero(T)
        for k in 1:M
            moment += a[k] * (dX[i, k] / h) * (dX[j, k] / h)
        end
        worst = _larger_violation(worst, abs(moment - (i == j ? one(T) : zero(T))))
    end
    return worst
end

# The larger of two violations; once a NaN appears it stays, since every comparison with it is false
_larger_violation(worst, violation) = (isnan(violation) || violation > worst) ? violation : worst

# The moment conditions as a linear system, stored transposed for column access:
# column r of At holds condition r evaluated at every neighbour, with the
# offsets scaled by h so that all conditions are of similar size. The conditions
# are: first moments zero (N columns), second moment the identity (N(N+1)/2
# columns, one per entry i ≤ j) and, for order 2, third moments zero
# (N(N+1)(N+2)/6 columns, one per entry i ≤ j ≤ l)
function _moment_system!(workspace::_Workspace{T}, dX::AbstractMatrix{T},
                         h::T, order::Integer) where {T<:AbstractFloat}
    At, b = workspace.At, workspace.b                # the arrays to fill
    N, M = size(dX)                                  # dimension and number of neighbours
    fill!(b, zero(T))                                # most conditions have a zero right-hand side
    r = 0                                            # index of the current condition
    for i in 1:N                                     # Σ a_k S_ik = 0
        r += 1
        for k in 1:M
            At[k, r] = dX[i, k] / h
        end
    end
    for i in 1:N, j in i:N                           # Σ a_k S_ik S_jk = δ_ij
        r += 1
        for k in 1:M
            At[k, r] = (dX[i, k] / h) * (dX[j, k] / h)
        end
        b[r] = i == j ? one(T) : zero(T)
    end
    if order == 2
        for i in 1:N, j in i:N, l in j:N             # Σ a_k S_ik S_jk S_lk = 0
            r += 1
            for k in 1:M
                At[k, r] = (dX[i, k] / h) * (dX[j, k] / h) * (dX[l, k] / h)
            end
        end
    end
    return nothing
end

# Minimum-norm solution of Atᵀ a = b, written into a. The columns of At are
# orthonormalized in place by Gram-Schmidt with two passes (enough for full
# orthogonality), while the right-hand side is forward-substituted alongside.
# The solution is then a combination of the orthonormal columns. Conditions
# that are linearly dependent on earlier ones leave a negligible remainder and
# are dropped. Returns the independence of the conditions: the smallest
# fraction of any condition left after removing its components along the
# earlier ones. It is zero when a condition was dropped, and small when one is
# nearly dependent, which makes the weights large.
function _minimum_norm_solve!(a::AbstractVector{T}, workspace::_Workspace{T}) where {T<:AbstractFloat}
    At, b, y = workspace.At, workspace.b, workspace.y  # system and scratch coefficients
    M, m = size(At)                                  # neighbours and conditions
    tol = 100 * M * eps(T)                           # relative size below which a condition is dependent
    independence = one(T)                            # smallest remaining fraction so far
    for r in 1:m
        t = b[r]                                     # right-hand side, reduced by earlier conditions
        original = norm(view(At, :, r))              # size of the condition before orthogonalization
        for pass in 1:2                              # two passes of Gram-Schmidt
            for s in 1:r-1
                c = zero(T)                          # component of column r along column s
                for k in 1:M
                    c += At[k, s] * At[k, r]
                end
                for k in 1:M
                    At[k, r] -= c * At[k, s]         # remove that component
                end
                t -= c * y[s]                        # and its share of the right-hand side
            end
        end
        remaining = norm(view(At, :, r))             # what is left of the condition
        fraction = original > 0 ? remaining / original : zero(T)  # its independent fraction
        independence = min(independence, fraction)
        if remaining <= tol * original
            At[:, r] .= zero(T)                      # dependent condition: drop it
            y[r] = zero(T)
        else
            for k in 1:M
                At[k, r] /= remaining                # normalize the new basis column
            end
            y[r] = t / remaining
        end
    end
    fill!(a, zero(T))                                # a = Σ_r y_r q_r over the basis columns
    for r in 1:m, k in 1:M
        a[k] += At[k, r] * y[r]
    end
    return independence
end