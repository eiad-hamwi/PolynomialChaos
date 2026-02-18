# propagate_twiss_table: propagate Twiss from BeamInit through a Beamline into a TypedTable
# Run from package root: julia --project=~/.julia/env/lean docs/examples/propagate_twiss_table.jl
# See README § "Twiss table along a beamline (SciBmad)"
# For sub-ms on long lattices: add GTPSA, then use linear=true, desc=GTPSA.Descriptor(6,1)

using PolynomialChaos
using SciBmad

# Build lattice and initial conditions
d = Drift(L=1.0)
qf = Quadrupole(Kn1=0.36, L=0.5)
qd = Quadrupole(Kn1=-0.36, L=0.5)
bl = Beamline([qf, d, qd, d], species_ref=Species("electron"), E_ref=18e9)
bi = BeamInit(
    x0=0.0, px0=0.0, y0=0.0, py0=0.0, δ0=0.0,
    σδ=1e-4, εx=1e-6, βx=10.0, αx=0.0, ηx=0.0, ηpx=0.0,
    εy=1e-6, βy=10.0, αy=0.0,
)

# Propagate Twiss through lattice → table (like SciBmad twiss)
t = propagate_twiss_table(bl, bi)
println("Twiss table ($(length(t)) rows):")
println(t)
