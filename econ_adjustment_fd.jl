module EconAdjustmentFD

using LinearAlgebra
using SpecialFunctions: erf

# Finite-difference solver for constrained Wasserstein gradient flow
# on [0,L] with reflecting (Neumann) boundaries, quadratic V and W.

export ModelParams, run_simulation, truncated_gaussian_equilibrium

# ---------------------------------------------------------------
# Parameters and grid
# ---------------------------------------------------------------

struct ModelParams
    κ::Float64      # curvature of V
    x0::Float64     # minimizer of V
    γ::Float64      # interaction strength in W
    σ2::Float64     # diffusion coefficient σ^2
    M::Float64      # target first moment
    L::Float64      # domain maximum (we approximate ℝ₊ by [0,L])
    Nx::Int         # number of grid points
    dt::Float64     # time step
    t_final::Float64
end

"""
    make_grid(params) -> x, dx

Uniform grid on [0,L] with Nx points.
"""
function make_grid(params::ModelParams)
    x = range(0.0, params.L; length = params.Nx)
    dx = step(x)
    return x, dx
end

# ---------------------------------------------------------------
# Potentials: quadratic case
# V(x) = κ/2 (x - x0)^2
# W(z) = γ/2 z^2  ⇒ ∂x(W * ρ)(x) = γ (x - M)
# (M is fixed by the constraint)
# ---------------------------------------------------------------

V(x, p::ModelParams) = 0.5 * p.κ * (x - p.x0)^2

"""
    dVdx(x, p)

Analytical derivative of V.
"""
dVdx(x, p::ModelParams) = p.κ .* (x .- p.x0)

"""
    dWdx(x, p)

For W(z) = γ/2 * z^2, one has ∂x(W*ρ)(x) = γ (x - M).
Here M is the fixed first moment (constraint).
"""
dWdx(x, p::ModelParams) = p.γ .* (x .- p.M)

# ---------------------------------------------------------------
# Neumann (reflecting) boundary operators
# ---------------------------------------------------------------

"""
    neumann_extend(f)

Extend array f by one ghost cell at each end with equal boundary values,
implementing Neumann (zero-flux) type reflection.
"""
function neumann_extend(f::AbstractVector)
    left = first(f)
    right = last(f)
    return vcat(left, f, right)
end

"""
    grad_x(f, dx)

Central difference gradient with Neumann boundary conditions.
Returns an array of same length as f.
"""
function grad_x(f::AbstractVector, dx::Float64)
    f_ext = neumann_extend(f)
    return (f_ext[3:end] .- f_ext[1:end-2]) ./ (2.0 * dx)
end

"""
    div_x(F, dx)

Divergence (spatial derivative) of flux F with Neumann BC.
"""
function div_x(F::AbstractVector, dx::Float64)
    F_ext = neumann_extend(F)
    return (F_ext[3:end] .- F_ext[1:end-2]) ./ (2.0 * dx)
end

# ---------------------------------------------------------------
# PDE RHS with endogenous λ_t
#
# PDE:
#   ∂_t ρ = ∂_x[ ρ ∂_x(V + W*ρ + λ_t x) + σ² ∂_x ρ ]
#
# For quadratic W, ∂_x(V + W*ρ + λ_t x) = κ(x - x0) + γ(x - M) + λ_t
# (λ_t enters as a constant in the drift).
#
# We choose λ_t so that the instantaneous time derivative of the first
# moment vanishes: d/dt ∫ x ρ dx = 0 in discrete form.
# ---------------------------------------------------------------

"""
    pde_rhs_without_lambda(ρ, x, dx, p)

RHS of the PDE with λ_t ≡ 0.
"""
function pde_rhs_without_lambda(ρ::AbstractVector,
                                x::AbstractVector,
                                dx::Float64,
                                p::ModelParams)
    # gradients of potentials
    dV = dVdx(x, p)
    dW = dWdx(x, p)

    # ∂_x ρ
    dρ = grad_x(ρ, dx)

    # drift gradient (without λ): ∂_x(V + W*ρ) = dV + dW
    grad_eff = dV .+ dW

    # flux = ρ * grad_eff + σ² * ∂_x ρ
    flux = ρ .* grad_eff .+ p.σ2 .* dρ

    rhs = div_x(flux, dx)
    return rhs
end

"""
    compute_lambda_and_rhs(ρ, x, dx, p)

Compute λ_t so that the instantaneous first-moment derivative vanishes,
i.e. d/dt ∫ x ρ dx = 0 in discrete form, and return `(λ_t, rhs)` with

    rhs = rhs0 + λ_t * div_x(ρ),

where `rhs0` is the PDE RHS for λ = 0.
"""
function compute_lambda_and_rhs(ρ::AbstractVector,
                                x::AbstractVector,
                                dx::Float64,
                                p::ModelParams)

    # RHS with λ = 0
    rhs0 = pde_rhs_without_lambda(ρ, x, dx, p)

    # λ–contribution: flux_λ = λ ρ  =>  RHS_λ = λ div_x(ρ)
    div_ρ = div_x(ρ, dx)

    # Contribution of rhs0 to the first moment:
    # moment_dot0 = d/dt ∫ x ρ dx |_{λ=0}
    moment_dot0 = sum(x .* rhs0) * dx

    # In the continuum with Neumann BC and total mass 1,
    # K = ∫ x ∂_x ρ dx = -∫ ρ dx = -1 exactly.
    # Using this exact value avoids numerical ill-conditioning.
    K = -1.0

    # Enforce d/dt ∫ xρ = 0  ⇒  moment_dot0 + λ K = 0
    λ = -moment_dot0 / K  # == moment_dot0

    rhs = rhs0 .+ λ .* div_ρ
    return λ, rhs
end

# ---------------------------------------------------------------
# Projection onto { mass = 1, mean = M }
# ---------------------------------------------------------------

"""
    project_mass_and_mean!(ρ, x, dx, M)

In-place projection of a density vector ρ onto the affine subspace
    { ρ : ∫ρ dx = 1,   ∫xρ dx = M }
in the least-squares sense (L² on the grid), followed by clipping
small negatives and renormalizing mass.
"""
function project_mass_and_mean!(ρ::Vector{Float64},
                                x::AbstractVector{Float64},
                                dx::Float64,
                                M::Float64)

    # Current mass and mean
    mass = sum(ρ) * dx
    mean = sum(x .* ρ) * dx

    # If we are already very close, skip
    if abs(mass - 1.0) < 1e-10 && abs(mean - M) < 1e-10
        return
    end

    # Geometric coefficients for the 2×2 system
    # A = [ ∑1 dx      ∑x dx
    #       ∑x dx      ∑x² dx ]
    A00 = length(x) * dx
    A01 = sum(x) * dx
    A11 = sum(x .^ 2) * dx

    # Desired corrections in mass and mean
    b0 = 1.0 - mass
    b1 = M   - mean

    detA = A00 * A11 - A01 * A01
    if abs(detA) < 1e-14
        # Extremely degenerate geometry – just normalize mass and exit
        ρ ./= mass
        return
    end

    # Solve A * [α; β] = [b0; b1]
    α = ( b0 * A11 - b1 * A01) / detA
    β = (-b0 * A01 + b1 * A00) / detA

    # Apply affine correction: ρ ← ρ + α + β x
    @inbounds for i in eachindex(ρ)
        ρ[i] += α + β * x[i]
    end

    # Enforce nonnegativity and renormalize mass again
    @inbounds for i in eachindex(ρ)
        ρ[i] = max(ρ[i], 0.0)
    end
    mass = sum(ρ) * dx
    if mass > 0
        ρ ./= mass
    else
        error("Projection produced zero mass – scheme unstable or parameters too aggressive.")
    end
end

# ---------------------------------------------------------------
# Diffusion (implicit) and one time step
# ---------------------------------------------------------------

"""
    diffusion_rhs(ρ, dx, σ2)

Compute σ² Δρ with a second–order finite–difference Laplacian
and reflecting (Neumann) boundary conditions on [0,L].
"""
function diffusion_rhs(ρ::Vector{Float64}, dx::Float64, σ2::Float64)
    N = length(ρ)
    out = similar(ρ)
    invdx2 = 1.0 / (dx^2)

    # left boundary (Neumann)
    out[1] = 2.0 * (ρ[2] - ρ[1]) * invdx2

    # interior points
    @inbounds for i in 2:N-1
        out[i] = (ρ[i+1] - 2.0*ρ[i] + ρ[i-1]) * invdx2
    end

    # right boundary (Neumann)
    out[N] = 2.0 * (ρ[N-1] - ρ[N]) * invdx2

    return σ2 .* out
end

"""
    build_diffusion_matrix(N, dx, σ2, dt) -> A::Tridiagonal

Return the matrix A = I - dt * σ² Δ_h with Neumann BC,
where Δ_h is the same discrete Laplacian as in `diffusion_rhs`.
"""
function build_diffusion_matrix(N::Int, dx::Float64, σ2::Float64, dt::Float64)
    invdx2 = 1.0 / (dx^2)

    main  = ones(Float64, N)
    lower = zeros(Float64, N-1)
    upper = zeros(Float64, N-1)

    # interior rows: Δρ_i = (ρ_{i+1} - 2ρ_i + ρ_{i-1}) / dx²
    @inbounds for i in 2:N-1
        main[i]      -= dt * σ2 * (-2.0 * invdx2)
        lower[i-1]   -= dt * σ2 * ( 1.0 * invdx2)
        upper[i]     -= dt * σ2 * ( 1.0 * invdx2)
    end

    # left boundary row: Δρ_1 = 2(ρ₂ - ρ₁)/dx²  → (-2, 2, 0,...)
    main[1]  -= dt * σ2 * (-2.0 * invdx2)
    upper[1] -= dt * σ2 * ( 2.0 * invdx2)

    # right boundary row: Δρ_N = 2(ρ_{N-1} - ρ_N)/dx² → (..., 2, -2)
    main[N]     -= dt * σ2 * (-2.0 * invdx2)
    lower[N-1]  -= dt * σ2 * ( 2.0 * invdx2)

    return Tridiagonal(lower, main, upper)
end

"""
    step_forward(ρ, x, dx, p, A) -> ρ_new, λ_t

Semi-implicit step:
  - drift and constraint terms are explicit,
  - diffusion σ² Δρ is implicit via (I - dt σ² Δ_h) ρ^{n+1} = ⋯
"""
function step_forward(ρ::Vector{Float64},
                      x::AbstractVector{Float64},
                      dx::Float64,
                      p::ModelParams,
                      A::Tridiagonal{Float64,Vector{Float64}})

    # Full RHS and λ from current state
    λ_t, rhs_full = compute_lambda_and_rhs(ρ, x, dx, p)

    # Explicit approximation of diffusion part using current ρ
    diff_full = diffusion_rhs(ρ, dx, p.σ2)

    # Drift + constraint contribution (explicit)
    drift_rhs = rhs_full .- diff_full

    # Right-hand side for semi-implicit update:
    #   (I - dt σ² Δ_h) ρ^{n+1} = ρ^n + dt * (drift part)
    rhs = ρ .+ p.dt .* drift_rhs

    # Solve tridiagonal system
    ρ_new = A \ rhs

    # Enforce non-negativity and renormalise probability mass
    @inbounds for i in eachindex(ρ_new)
        ρ_new[i] = max(ρ_new[i], 0.0)
    end
    mass = sum(ρ_new) * dx
    ρ_new ./= mass

    return ρ_new, λ_t
end

# ---------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------

"""
    run_simulation(p; ρ0 = nothing, policy_shock = nothing)

Run the constrained PDE simulation from t=0 to t_final.

Arguments
---------
p::ModelParams
ρ0 ::Vector{Float64} (optional)
    Initial density on the grid. If `nothing`, we use a Gaussian-like bump
    centered at M.
policy_shock ::Function (optional)
    A function `policy_shock(t, p)` that returns an updated `p`
    (e.g. shift x0, change κ, γ, etc.).

Returns
-------
A NamedTuple with fields:
    x, ρ, times, means, variances, lambdas, ρ_history
"""
function run_simulation(p::ModelParams;
                        ρ0::Union{Nothing,Vector{Float64}} = nothing,
                        policy_shock = nothing)

    x, dx = make_grid(p)

    # initial condition: Gaussian-like centered at M
    if ρ0 === nothing
        s0 = p.L / 10.0
        ρ0 = exp.(-0.5 .* ((x .- p.M) ./ s0).^2)
        ρ0 = max.(ρ0, 0.0)
        ρ0 ./= sum(ρ0) * dx
    else
        ρ0 = copy(ρ0)
        ρ0 = max.(ρ0, 0.0)
        ρ0 ./= sum(ρ0) * dx
    end

    # Project initial condition exactly onto {mass=1, mean=M}
    project_mass_and_mean!(ρ0, x, dx, p.M)

    ρ = ρ0
    t = 0.0
    A = build_diffusion_matrix(length(x), dx, p.σ2, p.dt)

    times    = Float64[t]
    means    = Float64[sum(x .* ρ) * dx]
    vars     = Float64[sum((x .- means[end]).^2 .* ρ) * dx]
    lambdas  = Float64[0.0]                 # λ(0) (approx)
    ρ_history = Vector{Vector{Float64}}()
    push!(ρ_history, copy(ρ))

    while t < p.t_final
        # Possibly update parameters (e.g. policy shock)
        if policy_shock !== nothing
            p = policy_shock(t, p)
        end

        ρ, λ_t = step_forward(ρ, x, dx, p, A)

        # Optional safeguard: re-project every step
        project_mass_and_mean!(ρ, x, dx, p.M)

        t += p.dt
        push!(times, t)
        push!(lambdas, λ_t)

        m = sum(x .* ρ) * dx
        v = sum((x .- m).^2 .* ρ) * dx
        push!(means, m)
        push!(vars, v)

        push!(ρ_history, copy(ρ))
    end

    return (
        x         = x,
        ρ         = ρ,
        times     = times,
        means     = means,
        variances = vars,
        lambdas   = lambdas,
        ρ_history = ρ_history,
    )
end

# ---------------------------------------------------------------
# Quadratic equilibrium on ℝ₊ (truncated Gaussian)
# ---------------------------------------------------------------

const INV_SQRT2PI = 1.0 / sqrt(2.0 * π)

phi(z) = INV_SQRT2PI * exp(-0.5 * z^2)

function Phi(z)
    # CDF of N(0,1) via erf
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))
end

"""
    truncated_gaussian_equilibrium(p; max_iter=50, tol=1e-10)

Compute (μ, s, λ_∞) for the truncated Gaussian equilibrium on ℝ₊
in the quadratic case.

Returns
-------
(μ, s, λ_inf)
"""
function truncated_gaussian_equilibrium(p::ModelParams;
                                        max_iter::Int = 50,
                                        tol::Float64 = 1e-10)

    # s² = 2σ² / (κ + γ)
    s = sqrt(2.0 * p.σ2 / (p.κ + p.γ))

    # Mills ratio λ(a) = φ(a) / (1 - Φ(a))
    function mills(a)
        denom = 1.0 - Phi(a)
        denom = max(denom, 1e-15)
        return phi(a) / denom
    end

    # F(μ) = μ + s * λ(-μ/s) - M
    function F(μ)
        a = -μ / s
        return μ + s * mills(a) - p.M
    end

    # F'(μ) = 1 - λ(a) (a + λ(a)),  a = -μ/s
    function Fprime(μ)
        a = -μ / s
        λa = mills(a)
        return 1.0 - λa * (a + λa)
    end

    # Weak truncation heuristic for initial guess
    λ0 = mills(0.0)
    μ = max(p.M - s * λ0, p.M - s / sqrt(2.0 * π))

    for _ in 1:max_iter
        val = F(μ)
        if abs(val) < tol
            break
        end
        dval = Fprime(μ)
        if abs(dval) < 1e-14
            break
        end
        μ -= val / dval
    end

    # Lagrange multiplier λ_∞ = κ x0 + γ M - (κ+γ) μ
    λ_inf = p.κ * p.x0 + p.γ * p.M - (p.κ + p.γ) * μ

    return μ, s, λ_inf
end

end # module