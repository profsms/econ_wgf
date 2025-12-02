#!/usr/bin/env julia
#
# experiments_econ_adjustment.jl
#
# Reproducible numerical experiments for
# "Economic Adjustment as a Constrained Wasserstein Gradient Flow"
#
# Assumes:
#   - econ_adjustment_fd.jl defines module EconAdjustmentFD with:
#       ModelParams
#       run_simulation(p::ModelParams; ρ0 = nothing, policy_shock = nothing)
#       truncated_gaussian_equilibrium(p::ModelParams)
#
# Usage (from shell):
#   julia experiments_econ_adjustment.jl
#
# Or from REPL:
#   include("experiments_econ_adjustment.jl")

using Printf
using Statistics
using Plots
using Distributions

# ------------------------------------------------------------------
# 1. Load PDE solver module
# ------------------------------------------------------------------

include("econ_adjustment_fd.jl")
using .EconAdjustmentFD

# We assume SimulationResult has at least fields:
#   x::Vector{Float64}
#   ρ::Vector{Float64}           # density at final time
#   times::Vector{Float64}
#   means::Vector{Float64}
#   variances::Vector{Float64}
#   lambdas::Vector{Float64}
#   ρ_history::Vector{Vector{Float64}}

# ------------------------------------------------------------------
# 2. Utility functions: Gini, energy, W2-on-grid
# ------------------------------------------------------------------

"""
    gini(x, ρ)

Approximate Gini coefficient for a nonnegative income/wealth grid `x`
and corresponding density values `ρ` on [0,∞), with ∫ρ ≈ 1 and dx uniform.
"""
function gini(x::AbstractVector, ρ::AbstractVector)
    N = length(x)
    dx = x[2] - x[1]

    # Sort by x (should already be sorted)
    order    = sortperm(x)
    x_sorted = x[order]
    ρ_sorted = ρ[order]

    # Total income (first moment)
    M = dx * sum(x_sorted .* ρ_sorted)

    pop_cum = cumsum(ρ_sorted) .* dx               # population share
    inc_cum = cumsum(x_sorted .* ρ_sorted) .* dx ./ M  # income share

    # Area under Lorenz curve by trapezoids
    area   = 0.0
    last_p = 0.0
    last_y = 0.0
    @inbounds for i in 1:N
        p = pop_cum[i]
        y = inc_cum[i]
        area += (y + last_y) * (p - last_p) / 2
        last_p, last_y = p, y
    end

    return 1 - 2 * area
end

"""
    energy_quadratic(x, ρ, p::ModelParams)

Discrete approximation of the free-energy functional 𝔈(ρ) for the
quadratic–Gaussian case:

    V(x) = κ/2 (x - x₀)²
    W(z) = γ/2 z²
    entropy term with σ²

Assumes equispaced grid `x` and density `ρ` with ∫ρ ≈ 1.
"""
function energy_quadratic(x::AbstractVector, ρ::AbstractVector, p::ModelParams)
    κ  = p.κ
    x0 = p.x0
    γ  = p.γ
    σ2 = p.σ2

    dx = x[2] - x[1]

    # Potential term
    V     = 0.5 * κ * (x .- x0).^2
    E_pot = dx * sum(V .* ρ)

    # Interaction term: 0.5 ∫∫ W(x-y) ρ(x)ρ(y) dx dy
    # With W(z) = γ/2 z², use discrete double sum
    N    = length(x)
    E_int = 0.0
    @inbounds for i in 1:N
        xi = x[i]
        ρi = ρ[i]
        for j in 1:N
            z2 = (xi - x[j])^2
            E_int += 0.5 * γ * 0.5 * z2 * ρi * ρ[j]
        end
    end
    E_int *= dx^2

    # Entropy term σ² ∫ ρ log ρ
    eps   = 1e-16
    E_ent = σ2 * dx * sum(ρ .* log.(ρ .+ eps))

    return E_pot + E_int + E_ent
end

"""
    w2_on_grid(x, ρ, μ, s)

Approximate W₂ distance between numerical density ρ on grid x and
a Gaussian N(μ, s²) via quantiles.
(For the paper you can replace this by an exact 1D OT computation if desired.)
"""
function w2_on_grid(x::AbstractVector, ρ::AbstractVector, μ::Float64, s::Float64)
    dx    = x[2] - x[1]
    F_num = cumsum(ρ) .* dx

    # Uniform quantiles
    q   = range(0.0, 1.0; length = length(x))
    x_q = similar(q)
    idx = 1
    @inbounds for (k, u) in enumerate(q)
        while idx < length(F_num) && F_num[idx] < u
            idx += 1
        end
        x_q[k] = x[idx]
    end

    d   = Normal(μ, s)
    y_q = quantile.(Ref(d), q)

    return sqrt(mean((x_q .- y_q).^2))
end

# ------------------------------------------------------------------
# 3. Experiment 1: Baseline convergence to equilibrium
# ------------------------------------------------------------------

function experiment_baseline(; saveprefix::String = "exp1_baseline")
    println("=== Experiment 1: Baseline convergence ===")

    # Baseline parameters
    p = ModelParams(
        1.0,   # κ
        1.0,   # x0
        0.5,   # γ
        0.05,  # σ²
        1.0,   # M
        4.0,   # L (domain [0,L])
        400,   # Nx
        1e-3,  # dt
        5.0,   # t_final
    )

    res = run_simulation(p)
    x   = res.x
    ρT  = res.ρ
    t   = res.times
    μ_t = res.means
    v_t = res.variances
    λ_t = res.lambdas
    ρ_hist = res.ρ_history

    # ---------- make all time-series the same length ----------
    n = minimum((
        length(t),
        length(μ_t),
        length(v_t),
        length(λ_t),
        length(ρ_hist),
    ))

    t   = t[1:n]
    μ_t = μ_t[1:n]
    v_t = v_t[1:n]
    λ_t = λ_t[1:n]
    ρ_hist = ρ_hist[1:n]

    # Analytical equilibrium (truncated Gaussian)
    μ_eq, s_eq, λ_eq = truncated_gaussian_equilibrium(p)
    @printf("Equilibrium mean μ_eq = %.4f\n", μ_eq)
    @printf("Equilibrium std  s_eq = %.4f\n", s_eq)
    @printf("Equilibrium λ_eq      = %.4f\n", λ_eq)
    @printf("Simulated mean(T)     = %.4f\n", μ_t[end])
    @printf("Simulated var(T)      = %.4f\n", v_t[end])

    # --------- plots ---------

    # Density at final time
    pltρ = plot(x, ρT;
        xlabel = "x",
        ylabel = "ρ(x,T)",
        title  = "Final density (baseline)",
        lw     = 2,
    )
    png(pltρ, saveprefix * "_density_T.png")

    # Mean over time
    plt_mean = plot(t, μ_t;
        xlabel = "t",
        ylabel = "mean",
        label  = "mean",
        lw     = 2,
        title  = "Mean over time (baseline)",
    )
    hline!([p.M];
        linestyle = :dash,
        label     = "M (constraint)",
    )
    png(plt_mean, saveprefix * "_mean.png")

    # Variance over time
    plt_var = plot(t, v_t;
        xlabel = "t",
        ylabel = "variance",
        label  = "variance",
        lw     = 2,
        title  = "Variance over time (baseline)",
    )
    png(plt_var, saveprefix * "_variance.png")

    # Lambda over time
    pltλ = plot(t, λ_t;
        xlabel = "t",
        ylabel = "λ(t)",
        lw     = 2,
        label  = "λ(t)",
        title  = "Shadow price λ(t) (baseline)",
    )
    hline!([λ_eq];
        linestyle = :dash,
        label     = "λ_eq",
    )
    png(pltλ, saveprefix * "_lambda.png")

    # Energy over time
    E_full = [energy_quadratic(x, ρ, p) for ρ in ρ_hist]
    nE = min(length(E_full), length(t))
    E_t = E_full[1:nE]
    tE  = t[1:nE]

    pltE = plot(tE, E_t;
        xlabel = "t",
        ylabel = "𝔈(ρ_t)",
        lw     = 2,
        label  = "𝔈(ρ_t)",
        title  = "Energy dissipation (baseline)",
    )
    png(pltE, saveprefix * "_energy.png")

    # Gini over time
    G_full = [gini(x, ρ) for ρ in ρ_hist]
    nG = min(length(G_full), length(t))
    G_t = G_full[1:nG]
    tG  = t[1:nG]

    pltG = plot(tG, G_t;
        xlabel = "t",
        ylabel = "Gini(ρ_t)",
        lw     = 2,
        label  = "Gini",
        title  = "Inequality over time (baseline)",
    )
    png(pltG, saveprefix * "_gini.png")

    println("Baseline plots saved with prefix '$saveprefix'.")
    return res, p
end


# ------------------------------------------------------------------
# 4. Experiment 2: Sensitivity to interaction (γ) and diffusion (σ²)
# ------------------------------------------------------------------

function experiment_sensitivity(; saveprefix::String = "exp2_sensitivity")
    println("=== Experiment 2: Sensitivity analysis (γ, σ²) ===")

    κ   = 1.0
    x0  = 1.0
    M   = 1.0
    L   = 4.0
    Nx  = 400
    dt  = 1e-3
    T   = 5.0

    gammas = [0.0, 0.25, 0.5, 1.0]
    sigmas = [0.02, 0.05, 0.10]

    results = Dict{Tuple{Float64,Float64},Any}()

    for γ in gammas, σ2 in sigmas
        @printf("Running γ = %.2f, σ² = %.3f ...\n", γ, σ2)
        p   = ModelParams(κ, x0, γ, σ2, M, L, Nx, dt, T)
        res = run_simulation(p)
        x   = res.x
        ρT  = res.ρ

        μT  = last(res.means)
        vT  = last(res.variances)
        GT  = gini(x, ρT)
        ET  = energy_quadratic(x, ρT, p)

        results[(γ, σ2)] = (p = p, res = res,
                            mean_T   = μT,
                            var_T    = vT,
                            gini_T   = GT,
                            energy_T = ET)
    end

    # Gini vs γ for each σ²
    plt1 = plot(title = "Gini at T vs interaction strength γ",
                xlabel = "γ",
                ylabel = "Gini(T)")
    for σ2 in sigmas
        Gvals = [results[(γ, σ2)].gini_T for γ in gammas]
        plot!(gammas, Gvals;
              marker = :o,
              lw     = 2,
              label  = "σ²=$(σ2)")
    end
    png(plt1, saveprefix * "_gini_vs_gamma.png")

    # Variance vs σ² for each γ
    plt2 = plot(title = "Var at T vs diffusion σ²",
                xlabel = "σ²",
                ylabel = "Var(T)")
    for γ in gammas
        Vvals = [results[(γ, σ2)].var_T for σ2 in sigmas]
        plot!(sigmas, Vvals;
              marker = :o,
              lw     = 2,
              label  = "γ=$(γ)")
    end
    png(plt2, saveprefix * "_var_vs_sigma2.png")

    println("Sensitivity plots saved with prefix '$saveprefix'.")
    return results
end

# ------------------------------------------------------------------
# 5. Experiment 3: Policy shock in the potential center x₀
# ------------------------------------------------------------------

"""
    experiment_policy_shock(; saveprefix="exp3_policy_shock")

Two-stage simulation:
  - Stage 1: run with baseline x₀ for t ∈ [0, T₁]
  - Stage 2: use ρ(T₁) as initial condition, shift x₀ → x₀ + Δx₀,
             run for t ∈ [T₁, T₁+T₂]

Interpreted as a policy shock that shifts the optimal state.
"""
function experiment_policy_shock(; saveprefix::String = "exp3_policy_shock")
    println("=== Experiment 3: Policy shock in x₀ ===")

    # Baseline parameters
    κ   = 1.0
    x0  = 1.0
    γ   = 0.5
    σ2  = 0.05
    M   = 1.0
    L   = 4.0
    Nx  = 400
    dt  = 1e-3

    T1  = 3.0   # pre-shock phase
    T2  = 3.0   # post-shock phase
    Δx0 = 0.5   # magnitude of policy shock

    # ---- Stage 1: pre-shock ----
    p1   = ModelParams(κ, x0, γ, σ2, M, L, Nx, dt, T1)
    res1 = run_simulation(p1)

    # Use final density of stage 1 as initial condition for stage 2
    ρ_init2 = res1.ρ

    # ---- Stage 2: post-shock ----
    p2   = ModelParams(κ, x0 + Δx0, γ, σ2, M, L, Nx, dt, T2)
    res2 = run_simulation(p2; ρ0 = ρ_init2)

    # Common spatial grid and endpoint densities
    x      = res1.x
    ρ_pre  = res1.ρ
    ρ_post = res2.ρ

    # ---- Build combined trajectories for mean / variance ----
    t1 = res1.times
    t2 = res2.times .+ T1
    t_full = vcat(t1, t2)

    μ_full = vcat(res1.means,     res2.means)
    v_full = vcat(res1.variances, res2.variances)

    n_mom = minimum((length(t_full), length(μ_full), length(v_full)))
    t_mom = t_full[1:n_mom]
    μ_t   = μ_full[1:n_mom]
    v_t   = v_full[1:n_mom]

    # ---- Build combined λ-path on a matching time grid ----
    λ_full = vcat(res1.lambdas, res2.lambdas)

    # λ is defined only after the first time step in each run,
    # so we align it with t_mom[2:end].
    if n_mom >= 2
        nλ  = min(length(λ_full), n_mom - 1)
        tλ  = t_mom[2:1+nλ]
        λ_t = λ_full[1:nλ]
    else
        tλ  = Float64[]
        λ_t = Float64[]
    end

    # ---- Plots ----

    # Densities before vs after shock
    plt1 = plot(x, ρ_pre;
        lw     = 2,
        label  = "pre-shock (t = T₁)",
        xlabel = "x",
        ylabel = "ρ(x,t)",
        title  = "Density before and after policy shock",
    )
    plot!(x, ρ_post;
        lw       = 2,
        linestyle = :dash,
        label    = "post-shock (t = T₁ + T₂)",
    )
    png(plt1, saveprefix * "_density_pre_post.png")

    # Mean and variance over time with shock marked
    plt2 = plot(t_mom, μ_t;
        lw     = 2,
        label  = "mean",
        xlabel = "t",
        ylabel = "mean / variance",
        title  = "Response of mean and variance to policy shock",
    )
    plot!(t_mom, v_t;
        lw    = 2,
        label = "variance",
    )
    vline!([T1];
        linestyle = :dash,
        label     = "shock time",
    )
    png(plt2, saveprefix * "_moments.png")

    # Shadow price path
    plt3 = plot(tλ, λ_t;
        lw     = 2,
        label  = "λ(t)",
        xlabel = "t",
        ylabel = "λ(t)",
        title  = "Shadow price response to policy shock",
    )
    vline!([T1];
        linestyle = :dash,
        label     = "shock time",
    )
    png(plt3, saveprefix * "_lambda.png")

    println("Policy shock plots saved with prefix '$saveprefix'.")
    return (p1 = p1, res1 = res1, p2 = p2, res2 = res2)
end

# ------------------------------------------------------------------
# 6. Main: run all experiments
# ------------------------------------------------------------------

function main()
    baseline_res, baseline_p = experiment_baseline()
    sensitivity_res          = experiment_sensitivity()
    policy_res               = experiment_policy_shock()
    println("All experiments finished.")
end

# Run if script is executed as main
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end