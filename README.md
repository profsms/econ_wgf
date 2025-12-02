# Economic Adjustment as a Constrained Wasserstein Gradient Flow

This repository contains the reproducible numerical experiments for the paper

> **Stanisław M. S. Halkiewicz, Vicenţiu D. Rădulescu**  
> *Economic Adjustment as a Constrained Wasserstein Gradient Flow*.

The code implements a finite–difference solver for the constrained Wasserstein
gradient flow introduced in the paper and produces all figures used in the
Numerical Experiments section.

---

## Repository structure

- `econ_adjustment_fd.jl`  
  Julia module `EconAdjustmentFD` implementing:
  - the finite–difference solver for the constrained Fokker–Planck equation on \([0,L]\) with reflecting boundary at \(x=0\),
  - the semi–implicit time stepping scheme (implicit diffusion, explicit drift),
  - enforcement of the first–moment constraint \(\int x \rho = M\),
  - the closed–form truncated–Gaussian benchmark for the quadratic case.

- `experiments_econ_adjustment.jl`  
  Standalone script that:
  - loads `EconAdjustmentFD`,
  - runs the three experiments described in the paper,
  - saves all plots as PNG files in the working directory.

You can treat this repository as a small, self–contained Julia project: there
are no additional source files or external data dependencies.

---

## Julia version and dependencies

The code was tested with **Julia 1.10**. It uses the following standard
packages:

- `LinearAlgebra`
- `SpecialFunctions`
- `Printf`
- `Statistics`
- `Plots`
- `Distributions`

You can install them from the Julia REPL:

```julia
using Pkg
Pkg.add.(["SpecialFunctions", "Plots", "Distributions"])
```
(`LinearAlgebra`, `Printf` and `Statistics` are part of the standard library.)

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/profsms/econ_wgf.git
cd <repo-name>
```
2. Run all experiments
From the shell
```bash
julia experiments_econ_adjustment.jl
```
or from the julia REPL
```julia
include("experiments_econ_adjustment.jl")
main()
```
