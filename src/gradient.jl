function gradient(center::Vector{T}, centerval::T,
                    points::Matrix{T}, pointvals::Vector{T},
                    idxs::Vector{S}, coeffs::Vector{T}) where {T <: AbstractFloat, S <: Integer}

    # Gradient estimate for scalar point values
    dx = points[:,idxs] .- center # coordinate diffs from central node
    dy = pointvals[idxs] .- centerval # value diffs from central node
    grad = dx * (coeffs .*dy) # Estimated gradient (tensor product (matrix multiplication))

    return grad
end # validated

function gradient(center::Vector{T}, centerval::Vector{T},
                    points::Matrix{T}, pointvals::Matrix{T},
                    idxs::Vector{S}, coeffs::Vector{T}) where {T <: AbstractFloat, S <: Integer}

    # Gradient estimate for vector values (for gradient of gradients, the result is the Hessian)
    dx = points[:,idxs] .- center # coordinate diffs from central node
    dy = pointvals[:,idxs] .- centerval # value diffs from central node
    grad = sum([coeffs[q] * (dx[:,q] * dy[:,q]') for q = 1:length(idxs)]) # gradient estimate (tensor product)

    return grad
end # validated

function gradient(center::Vector{T}, centerval::Matrix{T},
                    points::Matrix{T}, pointvals::Vector{Matrix{T}},
                    idxs::Vector{S}, coeffs::Vector{T}) where {T <: AbstractFloat, S <: Integer}

    # Gradient estimate for matrix values (for a Hessian matrix the result is the gradient of the Hessian)
    Ndim = size(points)[1]
    dx = points[:,idxs] .- center # coordinate diffs from central node
    dy = [pointvals[q]' - centerval for q in idxs] # value diffs from central node
    grad = zeros(Ndim,Ndim,Ndim)
    gradT = zeros(Ndim,Ndim,Ndim)
    for i = 1:Ndim
        # take the derivative of each row vector in the matrix
        grad[:,:,i] = sum([coeffs[q] * (dx[:,q] * dy[q][i,:]') for q = 1:length(idxs)]) # gradient estimate (tensor product)
        # take the derivative of each column vector in the matrix
        gradT[:,:,i] = sum([coeffs[q] * (dx[:,q] * dy[q][:,i]') for q = 1:length(idxs)]) # gradient estimate (tensor product)
    end
    grad_out = zeros(Ndim,Ndim,Ndim)
    for i = 1:Ndim
        grad_out[:,:,i] = (grad[:,:,i] .+ gradT[:,:,i]')./2
    end

    return grad_out
end # validated