function coefficients(center::Vector{T}, points::Matrix{T},
    nnearest::S) where {T <: AbstractFloat, S <: Integer}
    # n = 2
    # k = 20 # for stability use  k > n*(n+3)/2
    # Take a point near the center of cube

    # Least-squares recipe for coefficients
    C = center * ones(1,nnearest) # central node
    dX = points .- C  # diffs from central node
    G = dX' * dX
    F = G .* G
    v = diag(G)
    N = pinv(G) * G
    N = Diagonal(ones(length(N[1,:]))) .- N
    coeffs =  N * pinv(F*N) * v

    return coeffs
end