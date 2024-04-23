function divergence(center::Vector{T}, centerval::Vector{T},
    points::Matrix{T}, pointvals::Matrix{T},
    idxs::Vector{S}, coeffs::Vector{T}) where {T <: AbstractFloat, S <: Integer}

    # Divergence estimate for vector values
    dx = points[:,idxs] .- center # coordinate diffs from central node
    dy = pointvals[:,idxs] .- centerval # value diffs from central node
    div = sum([coeffs[q] * dot(dx[:,q], dy[:,q]) for q = 1:length(idxs)]) # divergence estimate (dot product)

    return div
end # validated

function divergence(center::Vector{T}, centerval::Matrix{T},
    points::Matrix{T}, pointvals::Vector{Matrix{T}},
    idxs::Vector{S}, coeffs::Vector{T}) where {T <: AbstractFloat, S <: Integer}

    # Divergence estimate for matrix values
    dx = points[:,idxs] .- center # coordinate diffs from central node
    dy = [pointvals[q] - centerval for q in idxs] # value diffs from central node
    div = sum([coeffs[q] * (dy[q]' * dx[:,q]) for q = 1:length(idxs)]) # divergence estimate (inner product)

    return div
end # validated

function divergence(center::Vector{T}, centerval::Array{T},
    points::Matrix{T}, pointvals::Vector{Array{T}},
    idxs::Vector{S}, coeffs::Vector{T}) where {T <: AbstractFloat, S <: Integer}

    # Divergence estimate for array values
    Ndim = size(points)[1]
    dx = points[:,idxs] .- center # coordinate diffs from central node
    dy = [pointvals[q] .- centerval for q in idxs] # value diffs from central node
    div = zeros(Ndim,Ndim)
    for i = 1:Ndim
        div[:,i] = sum([coeffs[q] * (dy[q][:,:,i]' * dx[:,q]) for q = 1:length(idxs)]) # divergence estimate (inner product)
    end

    return div
end # to be validated