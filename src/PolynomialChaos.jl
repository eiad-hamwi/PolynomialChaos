"""
Polynomial chaos expansion on GTPSA: E[f], Var[f], Cov[f] for functions of
random variables; propagation of beam Twiss parameters through transfer maps.
"""
module PolynomialChaos

using GTPSA
using LinearAlgebra

# Extension method when SciBmad is loaded
function propagate_twiss_table end

include("distributions.jl")
include("moments.jl")
include("twiss.jl")

export AbstractDistribution,
       Normal,
       Uniform,
       Deterministic,
       StandardNormal,
       raw_moment,
       collect_monomials,
       expectation,
       cross_expectation,
       variance,
       covariance,
       std,
       BeamInit,
       cholesky_factor,
       twiss_to_sigma,
       sigma_to_twiss,
       propagate_sigma,
       propagate_twiss,
       propagate_twiss_table

end
