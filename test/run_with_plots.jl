# Run tests with plot artifacts enabled. Sets POLYCHAOS_PLOT_TESTS=1 and optionally
# POLYCHAOS_PLOT_DIR, then runs the test suite and prints saved PNG paths.
ENV["POLYCHAOS_PLOT_TESTS"] = "1"
if !haskey(ENV, "POLYCHAOS_PLOT_DIR")
    ENV["POLYCHAOS_PLOT_DIR"] = joinpath(@__DIR__, "artifacts")
end

include(joinpath(@__DIR__, "runtests.jl"))
