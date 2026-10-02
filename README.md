# DiscreteTensorDerivatives.jl

DiscreteTensorDerivatives.jl computes the gradient, divergence, curl and Laplacian of scalar, vector and tensor fields sampled at scattered points. It needs no mesh and no connectivity: each operator is a weighted sum of differences over a point's nearest neighbours, with weights computed once per point cloud. The method is second-order accurate by default, with a cheaper first-order option for small point clouds.

## Installation

Install the package from the General registry:

```julia
using Pkg                                        # Julia's package manager
Pkg.add("DiscreteTensorDerivatives")             # install the registered package
```

## Quick start

Build the stencils once for a point cloud, then apply any operator to any field sampled at those points:

```julia
using DiscreteTensorDerivatives                     # load the package

points = rand(3, 5000)                              # 5000 random points in the unit cube, one column per point
stencils = StencilSet(points)                       # find neighbours and compute weights once (second order by default)

# A vector field v(x) = (x₁x₂, sin x₃, x₁²x₃), one column per point
v = reduce(hcat, [[p[1] * p[2], sin(p[3]), p[1]^2 * p[3]] for p in eachcol(points)])

G = gradient(stencils, v)                           # 3 × 3 × 5000: G[a, j, p] = ∂v_j/∂x_a at point p
D = divergence(stencils, v)                         # length 5000: ∇·v at every point
C = curl(stencils, v)                               # 3 × 5000: ∇×v at every point
L = laplacian(stencils, v)                          # 3 × 5000: ∇²v, component by component
```

Every operator also accepts a point index, to evaluate at a single point:

```julia
gradient(stencils, v, 42)                           # 3 × 3 gradient at point 42 only
divergence(stencils, v, 42)                         # a single number: ∇·v at point 42
```

Because the gradient of any field is itself a field, derivatives can be chained. For example, the Hessian of a scalar field is the gradient of its gradient:

```julia
f = [sin(p[1]) * exp(p[2]) for p in eachcol(points)]  # a scalar field, one value per point
H = gradient(stencils, gradient(stencils, f))         # 3 × 3 × 5000: H[a, b, p] = ∂²f/∂x_a∂x_b at point p
```

## Field layout

Field values always have the point index last, matching the layout of `points`:

| Field | `values` | `gradient` | `divergence` | `curl` (3D) | `laplacian` |
|---|---|---|---|---|---|
| Scalar | `P` | `N × P` | – | – | `P` |
| Vector | `N × P` | `N × N × P` | `P` | `N × P` | `N × P` |
| Matrix | `N × m × P` | `N × N × m × P` | `m × P` | `N × m × P` | `N × m × P` |

The gradient puts the derivative index first, so for a vector field `G[a, j, p]` $= \partial v_j / \partial x_a$, which is the transpose of the usual Jacobian. Divergence and curl act on the field's first index, which must therefore equal the spatial dimension `N`.

## The method

Let $\mathbf{x}_0$ be a point and $\mathbf{x}_k$ its neighbours, with offsets $\Delta\mathbf{x}_k = \mathbf{x}_k - \mathbf{x}_0$ and field differences $\Delta Y_k = Y_k - Y_0$. A Taylor expansion about $\mathbf{x}_0$ gives

$$
\Delta Y_k = \Delta\mathbf{x}_k \cdot \nabla Y + \tfrac{1}{2}\, \Delta\mathbf{x}_k^\mathsf{T} \mathbf{H}\, \Delta\mathbf{x}_k + \mathcal{O}(h^3),
$$

where $\mathbf{H}$ is the Hessian and $h$ is the stencil radius. Now choose weights $a_k$ for the neighbours so that the first two moments of the stencil satisfy

$$
\sum_k a_k\, \Delta\mathbf{x}_k = \mathbf{0}, \qquad \sum_k a_k\, \Delta\mathbf{x}_k \Delta\mathbf{x}_k^\mathsf{T} = \mathbf{I}.
$$

Multiplying the Taylor expansion by $a_k \Delta\mathbf{x}_k$ and summing leaves the gradient, because the second moment is the identity. Multiplying by $a_k$ alone and summing removes the gradient term, because the first moment vanishes, and leaves half the trace of the Hessian. This gives all four operators as one-line sums:

$$
\nabla \otimes Y \approx \sum_k a_k\, \Delta\mathbf{x}_k \otimes \Delta Y_k, \qquad
\nabla \cdot Y \approx \sum_k a_k\, \Delta\mathbf{x}_k \cdot \Delta Y_k,
$$

$$
\nabla \times Y \approx \sum_k a_k\, \Delta\mathbf{x}_k \times \Delta Y_k, \qquad
\nabla^2 Y \approx 2 \sum_k a_k\, \Delta Y_k.
$$

The same weights serve every operator and every field, whether scalar, vector or tensor.

### Computing the weights

The moment conditions are linear in the weights. Written out component by component, with $\Delta x_{k,i}$ the $i$-th component of the offset $\Delta\mathbf{x}_k$, they are

$$
\sum_k a_k\, \Delta x_{k,i} = 0, \qquad \sum_k a_k\, \Delta x_{k,i}\, \Delta x_{k,j} = \delta_{ij} \quad (i \le j),
$$

which is $N(N+3)/2$ equations in $N$ dimensions, since the second moment is symmetric. Collecting them gives a small linear system $\mathbf{A}\mathbf{a} = \mathbf{b}$ with one column per neighbour. With more neighbours than equations, the system has many solutions, and the package picks the one with minimum norm:

$$
\mathbf{a} = \arg\min_{\mathbf{a}} \|\mathbf{a}\| \quad \text{subject to} \quad \mathbf{A}\mathbf{a} = \mathbf{b}.
$$

Small weights keep the operators from amplifying rounding errors and noise in the field values. The system is solved by orthonormalizing the rows of $\mathbf{A}$ with two passes of Gram–Schmidt, after scaling the offsets by the stencil radius so that all equations are of similar size. The conditions hold exactly when there are at least $N(N+3)/2$ neighbours in general position.

### Second order

The leading error of every operator comes from the third moment of the stencil. With `order = 2` (default), the weights are also required to satisfy

$$
\sum_k a_k\, \Delta x_{k,i}\, \Delta x_{k,j}\, \Delta x_{k,l} = 0 \quad (i \le j \le l).
$$

These $N(N+1)(N+2)/6$ equations are added to the same linear system, and they make all four operators second-order accurate.

## Performance

`StencilSet` does all the expensive work, namely the neighbour search and the weights of every point, and it runs in parallel on all threads Julia was started with. Start Julia with `julia --threads=auto` to use every core, and pass `threaded = false` when calling `StencilSet` from code that is already parallel. On a 20-core desktop CPU, building second-order stencils for 100000 points in 3D takes about 0.2 seconds. The operators themselves are cheap by comparison, so one `StencilSet` can be reused for any number of fields on the same points.

## Accuracy and neighbour counts

| | Gradient, divergence, curl exact for | Laplacian exact for | Convergence | Minimum neighbours (2D / 3D) | Default `k` (2D / 3D) |
|---|---|---|---|---|---|
| `order = 1` | linear fields | quadratic fields | first order | 5 / 9 | 10 / 18 |
| `order = 2` (default) | quadratic fields | cubic fields | second order | 9 / 19 | 18 / 38 |

The default number of neighbours `k` is twice the minimum, because stencils at the minimum size are ill-conditioned. Set it explicitly with `StencilSet(points; k = 30)`, and use `minimum_neighbours(N, order)` to find the lower bound. Accuracy is lower near the boundary of the point cloud, where stencils are one-sided.

## Acknowledgements

This method descends from a 2016 Stack Overflow answer by a user named Hans, which introduced the moment conditions and the four operators above. That answer computed the weights through the Gram matrix $\mathbf{G} = \mathbf{B}^\mathsf{T}\mathbf{B}$ of the offset matrix $\mathbf{B}$, as $\mathbf{a} = \mathbf{P}\, \big( (\mathbf{G} \circ \mathbf{G})\, \mathbf{P} \big)^+ \mathrm{diag}(\mathbf{G})$, where $\mathbf{P}$ projects onto the null space of $\mathbf{B}$. This formula gives exactly the same weights as the minimum-norm solution above, because $\mathbf{G} \circ \mathbf{G}$ carries the same information as the second-moment equations. This package solves the moment equations directly, which is simpler and faster, and adds the second-order extension.