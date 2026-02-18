# Run from package root:
# julia --project=. docs/examples/moments_and_beam_sigma.jl

using PolynomialChaos, GTPSA

println("=== Example 1: lognormal moments ===")
xi = GTPSA.vars(GTPSA.Descriptor(1, 20))[1]
Y = exp(xi)
dists = [StandardNormal]
@assert abs(expectation(Y, dists) - exp(0.5)) < 2e-11
@assert abs(variance(Y, dists) - (exp(2) - exp(1))) < 2e-8
println("E[Y] and Var[Y] match exact values.")

println("\n=== Example 2: drift beta-waist ===")
epsx = 1e-6
bi = BeamInit(
    x0 = 0.0, px0 = 0.0, y0 = 0.0, py0 = 0.0, δ0 = 0.0,
    σδ = 0.0,
    εx = epsx, βx = 0.5, αx = 0.0, ηx = 0.0, ηpx = 0.0,
    εy = 1e-6, βy = 1.0, αy = 0.0,
)
x = GTPSA.vars(GTPSA.Descriptor(6, 2))
map_out = copy(x)
map_out[1] = x[1] + 1.0 * x[2]
Σ = propagate_sigma(map_out, bi).cov
@assert isapprox(Σ[1, 1] / epsx, 0.5 + 1.0^2 / 0.5; rtol = 1e-12)
println("beta(s) matches beta* + s^2/beta*.")

println("\n=== Example 3: convergence with order ===")
for p in 2:2:20
    Yp = exp(GTPSA.vars(GTPSA.Descriptor(1, p))[1])
    err = abs(expectation(Yp, [StandardNormal]) - exp(0.5))
    println("order $p  error $err")
end

println("\nAll example checks passed.")
