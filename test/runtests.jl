using DiscreteTensorDerivatives
using ConvolutionInterpolations
using LinearAlgebra
using Test

# Deterministic pseudo-random points in the unit cube from a 32-bit linear
# congruential generator: no Random dependency, identical points on every
# Julia version, and irregular enough that stencils are not symmetric
function lcg_points(N, P; seed = 0x12345678)
    state = UInt32(seed)                         # generator state, wraps around at 2^32
    points = Matrix{Float64}(undef, N, P)
    for j in 1:P, i in 1:N
        state = state * 0x0019660d + 0x3c6ef35f  # Numerical Recipes LCG step
        points[i, j] = state / 2.0^32            # map to [0, 1)
    end
    return points
end

# Median of a collection, to avoid a Statistics dependency
function median_of(values)
    sorted = sort(collect(values))
    n = length(sorted)
    return isodd(n) ? sorted[(n + 1) ÷ 2] : (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

# Indices of points at least `margin` away from the boundary of the unit cube
interior(points, margin) = [i for i in axes(points, 2) if all(margin .< points[:, i] .< 1 - margin)]

@testset "DiscreteTensorDerivatives.jl" begin

    @testset "Moment conditions" begin
        for N in (2, 3), order in (1, 2), scale in (1.0, 1e-4)
            points = scale .* lcg_points(N, 400)     # also tiny stencils, to catch scaling problems
            stencils = StencilSet(points; order = order)
            @test stencils.order == order
            first = second = third = 0.0             # worst residual of each moment condition
            for i in axes(points, 2)
                dX = points[:, stencils.neighbours[i]] .- points[:, i]  # offsets of point i
                a = stencils.weights[i]                                  # weights of point i
                h = maximum(norm, eachcol(dX))                           # stencil radius
                first = max(first, norm(dX * a) * h)                     # Σ a Δx = 0, made dimensionless
                second = max(second, norm(dX * Diagonal(a) * dX' - I))   # Σ a Δx Δxᵀ = I
                if order == 2
                    m3 = [sum(a[q] * dX[r, q] * dX[s, q] * dX[t, q] for q in eachindex(a))
                          for r in 1:N, s in 1:N, t in 1:N]              # Σ a Δx ⊗ Δx ⊗ Δx
                    third = max(third, norm(m3) / h)                     # made dimensionless
                end
            end
            @test first < 1e-10
            @test second < 1e-10
            @test third < 1e-10
        end
    end

    @testset "Exactness on polynomial fields" begin
        points = lcg_points(3, 600)
        columns = collect(eachcol(points))
        stencils1 = StencilSet(points; order = 1)
        stencils2 = StencilSet(points; order = 2)
        tol = 1e-9

        # Scalar polynomials of degree 1, 2 and 3 with their exact gradients and Laplacians
        linear(p)     = 1 + p[1] - 2p[2] + 3p[3]
        quadratic(p)  = linear(p) + p[1]^2 + 2p[1] * p[2] - p[2] * p[3] + 3p[3]^2
        cubic(p)      = quadratic(p) + p[1]^3 - p[1] * p[2] * p[3] + 2p[2]^2 * p[3]
        ∇linear(p)    = [1.0, -2.0, 3.0]
        ∇quadratic(p) = ∇linear(p) + [2p[1] + 2p[2], 2p[1] - p[3], -p[2] + 6p[3]]
        ∇cubic(p)     = ∇quadratic(p) + [3p[1]^2 - p[2] * p[3], -p[1] * p[3] + 4p[2] * p[3], -p[1] * p[2] + 2p[2]^2]
        Δlinear(p)    = 0.0
        Δquadratic(p) = 8.0
        Δcubic(p)     = 8.0 + 6p[1] + 4p[3]

        # Largest gradient and Laplacian errors of a scalar field over all points
        gradient_error(s, f, ∇f) = maximum(abs, gradient(s, f.(columns)) .- stack(∇f, columns))
        laplacian_error(s, f, Δf) = maximum(abs, laplacian(s, f.(columns)) .- Δf.(columns))

        # Order 1: the gradient is exact for linear fields, the Laplacian also for quadratic ones
        @test gradient_error(stencils1, linear, ∇linear) < tol
        @test laplacian_error(stencils1, linear, Δlinear) < tol
        @test laplacian_error(stencils1, quadratic, Δquadratic) < tol
        @test gradient_error(stencils1, quadratic, ∇quadratic) > 1e-3   # not exact: confirms the order

        # Order 2: the gradient is exact for quadratic fields, the Laplacian also for cubic ones
        @test gradient_error(stencils2, quadratic, ∇quadratic) < tol
        @test laplacian_error(stencils2, quadratic, Δquadratic) < tol
        @test laplacian_error(stencils2, cubic, Δcubic) < tol
        @test gradient_error(stencils2, cubic, ∇cubic) > 1e-3           # not exact: confirms the order

        # Linear vector field v = A x + b: every operator is exact at order 1
        A = [1.0 2.0 0.0; -1.0 0.0 3.0; 2.0 1.0 -1.0]
        b = [0.5, -1.0, 2.0]
        V = stack(p -> A * p + b, columns)                              # 3 × P
        @test maximum(abs, gradient(stencils1, V) .- A') < tol          # [a, j] = ∂v_j/∂x_a = A[j, a]
        @test maximum(abs, divergence(stencils1, V) .- tr(A)) < tol
        @test maximum(abs, curl(stencils1, V) .- [-2.0, -2.0, -3.0]) < tol
        @test maximum(abs, laplacian(stencils1, V)) < tol

        # Quadratic vector field: every operator is exact at order 2
        vq(p)    = [p[1] * p[2] + p[3], p[2]^2 - p[1] * p[3], p[1]^2 + p[2] * p[3]]
        ∇vq(p)   = [p[2] (-p[3]) 2p[1]; p[1] 2p[2] p[3]; 1.0 (-p[1]) p[2]]  # [a, j] = ∂v_j/∂x_a
        divvq(p) = 4p[2]
        curlvq(p) = [p[3] + p[1], 1 - 2p[1], -p[3] - p[1]]
        Δvq(p)   = [0.0, 2.0, 2.0]
        Vq = stack(vq, columns)                                          # 3 × P
        @test maximum(abs, gradient(stencils2, Vq) .- stack(∇vq, columns)) < tol
        @test maximum(abs, divergence(stencils2, Vq) .- divvq.(columns)) < tol
        @test maximum(abs, curl(stencils2, Vq) .- stack(curlvq, columns)) < tol
        @test maximum(abs, laplacian(stencils2, Vq) .- stack(Δvq, columns)) < tol

        # Linear matrix field M_ij = Σ_a B[a, i, j] x_a + 1: gradient, divergence and curl are exact
        B = reshape(collect(1.0:27.0), 3, 3, 3) .- 14.0
        M = stack(p -> [sum(B[a, i, j] * p[a] for a in 1:3) for i in 1:3, j in 1:3] .+ 1.0, columns)  # 3 × 3 × P
        div_expected = [sum(B[a, a, j] for a in 1:3) for j in 1:3]     # Σ_a ∂M_aj/∂x_a
        curl_expected = permutedims(hcat([B[2, 3, r] - B[3, 2, r] for r in 1:3],
                                         [B[3, 1, r] - B[1, 3, r] for r in 1:3],
                                         [B[1, 2, r] - B[2, 1, r] for r in 1:3]))  # [i, r] = ε_ijk ∂M_kr/∂x_j
        @test maximum(abs, gradient(stencils1, M) .- B) < tol           # [a, i, j] = ∂M_ij/∂x_a
        @test maximum(abs, divergence(stencils1, M) .- div_expected) < tol
        @test maximum(abs, curl(stencils1, M) .- curl_expected) < tol
    end

    @testset "Convergence order" begin
        # Smooth 2D field with its exact gradient and Laplacian
        f(p)  = sin(3p[1]) * exp(p[2]) + cos(2p[2])
        ∇f(p) = [3cos(3p[1]) * exp(p[2]), sin(3p[1]) * exp(p[2]) - 2sin(2p[2])]
        Δf(p) = -8sin(3p[1]) * exp(p[2]) - 4cos(2p[2])

        # Median interior gradient and Laplacian errors for P points
        function median_errors(P, order)
            points = lcg_points(2, P)
            columns = collect(eachcol(points))
            stencils = StencilSet(points; order = order)
            values = f.(columns)
            G = gradient(stencils, values)
            L = laplacian(stencils, values)
            inner = interior(points, 0.2)
            return (median_of(norm(G[:, i] - ∇f(columns[i])) for i in inner),
                    median_of(abs(L[i] - Δf(columns[i])) for i in inner))
        end

        # Four times as many points halves the stencil radius in 2D
        coarse1, fine1 = median_errors(2000, 1), median_errors(8000, 1)
        coarse2, fine2 = median_errors(2000, 2), median_errors(8000, 2)
        @test all(coarse1 ./ fine1 .> 1.6)    # first order: error ratio about 2
        @test all(coarse2 ./ fine2 .> 3.2)    # second order: error ratio about 4
        @test all(fine2 .< fine1 ./ 5)        # order 2 is much more accurate at the same resolution
    end

    @testset "Shapes, single points and errors" begin
        points = lcg_points(3, 200)
        stencils = StencilSet(points)
        f = vec(sum(points; dims = 1))           # scalar field
        V = 2 .* points                          # vector field, 3 × 200
        M = permutedims(reshape(points, 3, 200, 1) .* reshape([1.0, -1.0], 1, 1, 2), (1, 3, 2))  # 3 × 2 × 200

        @test size(gradient(stencils, f)) == (3, 200)
        @test size(laplacian(stencils, f)) == (200,)
        @test size(gradient(stencils, V)) == (3, 3, 200)
        @test size(divergence(stencils, V)) == (200,)
        @test size(curl(stencils, V)) == (3, 200)
        @test size(laplacian(stencils, V)) == (3, 200)
        @test size(gradient(stencils, M)) == (3, 3, 2, 200)
        @test size(divergence(stencils, M)) == (2, 200)
        @test size(curl(stencils, M)) == (3, 2, 200)
        @test size(laplacian(stencils, M)) == (3, 2, 200)

        # Single-point methods agree with the whole-cloud ones
        @test gradient(stencils, V, 7) ≈ gradient(stencils, V)[:, :, 7]
        @test divergence(stencils, V, 7) isa Float64
        @test divergence(stencils, V, 7) ≈ divergence(stencils, V)[7]
        @test curl(stencils, V, 7) ≈ curl(stencils, V)[:, 7]
        @test laplacian(stencils, f, 7) isa Float64
        @test laplacian(stencils, f, 7) ≈ laplacian(stencils, f)[7]
        @test divergence(stencils, M, 7) ≈ divergence(stencils, M)[:, 7]

        # Neighbour counts and invalid arguments
        @test minimum_neighbours(2, 1) == 5
        @test minimum_neighbours(3, 1) == 9
        @test minimum_neighbours(2, 2) == 9
        @test minimum_neighbours(3, 2) == 19
        @test stencils.order == 2                                          # second order is the default
        @test length(stencils.neighbours[1]) == 38
        @test length(StencilSet(points; order = 1).neighbours[1]) == 18
        @test_throws ArgumentError StencilSet(points; k = 5)               # below the minimum of 19
        @test_throws ArgumentError StencilSet(points; order = 3)           # unsupported order
        @test_throws ArgumentError StencilSet(points[:, 1:10])             # k = 38 needs more points
        @test_throws DimensionMismatch gradient(stencils, zeros(5))        # wrong number of points
        @test_throws DimensionMismatch divergence(stencils, f)             # divergence of a scalar field
        @test_throws DimensionMismatch divergence(stencils, zeros(2, 200)) # first dimension is not 3
        @test_throws ArgumentError curl(StencilSet(lcg_points(2, 100)), zeros(2, 100))  # curl in 2D

        # Integer coordinates become Float64, and Float32 stays Float32
        @test StencilSet(round.(Int, 1000 .* points)) isa StencilSet{Float64}
        stencils32 = StencilSet(Float32.(points))
        @test eltype(stencils32.weights[1]) == Float32
        @test maximum(abs, gradient(stencils32, Float32.(f)) .- 1) < 1e-3  # ∇(x + y + z) = (1, 1, 1)
    end

    @testset "Threaded and serial builds agree" begin
        for N in (2, 3), order in (1, 2)
            points = lcg_points(N, 3000)                                   # enough points for several chunks per thread
            threaded_build = StencilSet(points; order = order)             # chunked, one task per range
            serial_build = StencilSet(points; order = order, threaded = false)  # one range covering all points
            @test threaded_build.neighbours == serial_build.neighbours     # identical neighbour lists
            @test threaded_build.weights == serial_build.weights           # bitwise identical weights
        end
    end

    @testset "Regular grids and degenerate points" begin
        # A regular grid of n points per axis in the unit cube, one column per point
        grid(n, N) = reduce(hcat, [collect(Float64.(Tuple(c) .- 1) ./ (n - 1))
                                   for c in CartesianIndices(ntuple(_ -> n, N))])

        # Stencils on a regular grid are exact for linear fields everywhere, faces included,
        # and the Laplacian is exact for quadratic ones, at both orders with the default k. On a
        # 3D grid the nearest neighbours of face, edge and corner points lie on too few planes,
        # so those stencils grow beyond k, up to the default kmax = 2k; inner points keep k
        for (N, order) in [(2, 1), (2, 2), (3, 1), (3, 2)]
            points = grid(15, N)
            stencils = StencilSet(points; order)
            k = 2 * minimum_neighbours(N, order)                           # the default stencil size
            sizes = length.(stencils.neighbours)                           # neighbours of every point
            @test all(k .<= sizes .<= 2k)                                  # never fewer than k, never more than kmax
            if N == 3
                @test any(sizes .> k)                                      # some boundary stencils grew
                centre = argmin(vec(sum(abs2, points .- 0.5; dims = 1)))   # the point at the centre
                @test sizes[centre] == k                                   # inner stencils did not grow
            end
            slope = collect(1.0:N)                                         # gradient of the linear field
            @test maximum(abs, gradient(stencils, vec(slope' * points)) .- slope) < 1e-9
            @test maximum(abs, laplacian(stencils, vec(sum(abs2, points; dims = 1))) .- 2N) < 1e-9
            if order == 2
                # With the third-moment conditions met at every point, the gradient of a
                # quadratic field is exact everywhere, on the faces, edges and corners too
                square = vec(sum(abs2, points; dims = 1))                  # |x|², whose gradient is 2x
                @test maximum(abs, gradient(stencils, square) .- 2 .* points) < 1e-9
            end
        end
        @test_throws ArgumentError StencilSet(grid(15, 2); kmax = 10)      # kmax below the default k = 18

        # Points that do not fill their space cannot satisfy the moment conditions, even with
        # grown stencils, so building stencils on them raises an error instead of giving
        # meaningless derivatives
        plane = vcat(lcg_points(2, 500), zeros(1, 500))                   # a plane in 3D
        tilted = [1.0 0.0; 0.0 1.0; 0.3 0.7] * lcg_points(2, 500)         # a tilted plane in 3D
        sphere = reduce(hcat, [[sin(acos(2v - 1)) * cos(2π * u),           # points spread evenly
                                sin(acos(2v - 1)) * sin(2π * u),           # over the unit sphere's
                                2v - 1] for (u, v) in eachcol(lcg_points(2, 1000))])  # curved surface
        angles = range(0, 6.2, length = 500)
        circle = vcat(cos.(angles)', sin.(angles)')                        # a circle in 2D
        line = vcat(collect(range(0, 1, length = 500))', zeros(1, 500))    # a line in 2D
        for points in (plane, tilted, sphere, circle, line)
            @test_throws ArgumentError StencilSet(points)
            @test_throws ArgumentError StencilSet(points; threaded = false)  # the same error without threads
        end

        # The message names the point and the cause
        err = try StencilSet(sphere) catch e; e end
        @test err isa ArgumentError
        @test occursin("point", err.msg) && occursin("moment conditions", err.msg)

        @test_throws ArgumentError StencilSet(grid(15, 2); kmax = 10)      # kmax below the default k = 18
        @test_throws ArgumentError StencilSet(grid(15, 3); order = 1, kmax = 18)  # kmax = k: no growth, so 3D grid faces fail
    end

    @testset "Agreement with ConvolutionInterpolations.jl" begin
        # A smooth random field: the interpolant of an 8 × 8 grid of pseudo-random values
        grid = range(0.0, 1.0, length = 8)
        data = lcg_points(8, 8; seed = 0x0badf00d)
        field = convolution_interpolation((grid, grid), data)
        field_x = convolution_interpolation((grid, grid), data; derivative = (1, 0))
        field_y = convolution_interpolation((grid, grid), data; derivative = (0, 1))

        # Sample the field at scattered points, and its exact gradient from the interpolant
        points = lcg_points(2, 8000)
        values = [field(p[1], p[2]) for p in eachcol(points)]
        exact = stack([field_x(p[1], p[2]), field_y(p[1], p[2])] for p in eachcol(points))
        inner = interior(points, 0.2)
        scale = median_of(norm(exact[:, i]) for i in inner)     # typical gradient magnitude

        # Median relative interior gradient error for each order
        function relative_error(order)
            G = gradient(StencilSet(points; order = order), values)
            return median_of(norm(G[:, i] - exact[:, i]) for i in inner) / scale
        end
        error1, error2 = relative_error(1), relative_error(2)
        @test error1 < 0.2
        @test error2 < 0.06
        @test error2 < error1 / 2
    end
end