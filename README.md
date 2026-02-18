# PolynomialChaos.jl

Polynomial chaos expansion on [GTPSA](https://github.com/niclasmattsson/GTPSA.jl) for moments of random-variable functions and Twiss propagation through transfer maps.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/eiad-hamwi/PolynomialChaos.jl")
```

## Quick start

### Example 1: lognormal moments

For `Y = exp(ξ)` with `ξ ~ N(0,1)`: `E[Y] = e^(1/2)`, `Var(Y) = e^2 - e`.

```julia
using PolynomialChaos, GTPSA

xi = vars(Descriptor(1, 20))[1]
Y = exp(xi)
dists = [StandardNormal]

@assert abs(expectation(Y, dists) - exp(0.5)) < 2e-11
@assert abs(variance(Y, dists) - (exp(2) - exp(1))) < 2e-8
```

### Example 2: drift beta-waist

At a waist (`alpha = 0`) in a drift of length `s`: `beta(s) = beta* + s^2/beta*`.

```julia
using PolynomialChaos, GTPSA
D = Descriptor(6, 2)

epsx = 1e-6
bi = BeamInit(
    x0 = 0.0, px0 = 0.0, y0 = 0.0, py0 = 0.0, δ0 = 0.0,
    σδ = 0.0,
    εx = epsx, βx = 0.5, αx = 0.0, ηx = 0.0, ηpx = 0.0,
    εy = 1e-6, βy = 1.0, αy = 0.0,
)

x = vars(D)
map_out = copy(x)
map_out[1] = x[1] + 1.0 * x[2]

Σ = propagate_sigma(map_out, bi).cov
@assert isapprox(Σ[1, 1] / epsx, 0.5 + 1.0^2 / 0.5; rtol = 1e-12)
```

### Example 3: convergence with order

Error in `E[exp(ξ)]` versus GTPSA order:

```julia
using PolynomialChaos, GTPSA

for p in 2:2:20
    Y = exp(vars(Descriptor(1, p))[1])
    err = abs(expectation(Y, [StandardNormal]) - exp(0.5))
    println("order $p  error $err")
end
```

### Beam covariance through a transfer map

```julia
using PolynomialChaos, GTPSA, BeamTracking
D = Descriptor(6, 2)

line = your_beamline()
x = vars(D)[1:6]
b = Bunch(reshape(x, 1, 6); species = line.species_ref, p_over_q_ref = line.p_over_q_ref)
track!(b, line)
map_out = vec(b.coords.v)

bi = BeamInit(
    x0 = 0.0, px0 = 0.0, y0 = 0.0, py0 = 0.0, δ0 = 0.0,
    σδ = 1e-4,
    εx = 1e-6, βx = 12.6, αx = 1.8, ηx = 2.7, ηpx = -0.42,
    εy = 1e-6, βy = 4.1, αy = -0.65,
)

stats = propagate_sigma(map_out, bi; desc = D)
bi_out = propagate_twiss(map_out, bi; desc = D)
bi_back = propagate_twiss(map_out, bi_out; backward = true, desc = D)
```

## API overview

| Area | Signatures |
|------|------------|
| **Distributions** | `Normal(μ, σ)`, `Uniform(a, b)`, `Deterministic(value)`, `StandardNormal` |
| **Raw moment** | `raw_moment(d, n) -> E[X^n]` |
| **Moments** | `expectation(t, dists)`, `cross_expectation(t1, t2, dists)`, `variance(t, dists)`, `std(t, dists)`, `covariance(ts, dists)` |
| **Beam** | `BeamInit(...)`, `twiss_to_sigma(bi)`, `sigma_to_twiss(μ, Σ)` |
| **Propagation** | `propagate_sigma(map_out, bi; desc)`, `propagate_twiss(map_out, bi; backward, desc)` |

- **`propagate_sigma`** -> `(mean, cov, composed)`.
- **`propagate_twiss`** -> `BeamInit`.

## Q&A

Q: How do I compute `E[exp(ξ)]` for `ξ ~ N(0,1)`?  
A: Build `Y = exp(vars(Descriptor(1, p))[1])`, then call `expectation(Y, [StandardNormal])`.

Q: How do I get `Var[f]` or `Std[f]` from a TPS `f`?  
A: Use `variance(f, dists)` and `std(f, dists)`.

Q: How do I compute `E[w1 * w2^2]` for independent standard normals?  
A: Use `cross_expectation(w1, w2^2, [StandardNormal, StandardNormal])`.

Q: How do I compute moments of a non-normal input?  
A: Use `raw_moment`, e.g. `raw_moment(Uniform(0, 2), 2)`.

Q: How do I convert Twiss parameters to covariance?  
A: Build `BeamInit(...)` and call `twiss_to_sigma(bi)`.

Q: How do I extract Twiss from mean/covariance?  
A: Call `sigma_to_twiss(μ, Σ)`.

Q: How do I propagate beam statistics through a map?  
A: Call `propagate_sigma(map_out, bi; desc=Descriptor(6, 2))`.

Q: What does `propagate_sigma` return?  
A: `(mean, cov, composed)`.

Q: How do I propagate backward?  
A: Call `propagate_twiss(map_out, bi; backward=true, desc=Descriptor(6, 2))`.

Q: How do I check drift beta-waist behavior quickly?  
A: Use a drift map `x -> x + s*px` and verify `beta(s) = beta* + s^2/beta*`.

## Tests and plot artifacts

From the package directory, with an environment that has BeamTracking and lattice dependencies:

```bash
julia --project=~/.julia/env/lean test/runtests.jl
POLYCHAOS_PLOT_TESTS=1 julia --project=~/.julia/env/lean test/runtests.jl
julia --project=~/.julia/env/lean test/run_with_plots.jl
```

Set `POLYCHAOS_PLOT_DIR` to change the plot output directory.

## License

MIT
