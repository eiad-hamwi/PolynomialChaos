using GTPSA
using LinearAlgebra

function default_beam_init()
    return PolynomialChaos.BeamInit(
        x0 = 0.105,
        px0 = 0.0025,
        y0 = 0.0,
        py0 = 0.0,
        δ0 = 0.0,
        σδ = 1e-4,
        εx = 1e-6,
        βx = 12.6,
        αx = 1.8,
        ηx = 2.7,
        ηpx = -0.42,
        εy = 1e-6,
        βy = 4.1,
        αy = -0.65,
    )
end

# Self-contained 6D linear map with drift-like transport and dispersion coupling.
function make_linear_map(desc::Descriptor; sx::Float64=1.0, sy::Float64=0.8, dx::Float64=0.12, dpx::Float64=-0.03)
    x = GTPSA.vars(desc)[1:6]
    m = copy(x)
    m[1] = x[1] + sx * x[2] + dx * x[6]
    m[2] = x[2] + dpx * x[6]
    m[3] = x[3] + sy * x[4]
    m[4] = x[4]
    m[5] = x[5]
    m[6] = x[6]
    return m
end

@testset "PolynomialChaos self-contained integration" begin
    bi = default_beam_init()

    @testset "Quadratic moments with analytic reference" begin
        D = Descriptor(5, 2)
        w = GTPSA.vars(D)[1:5]
        dists = fill(StandardNormal, 5)

        # f = c + linear + quadratic terms chosen for simple analytic moments
        # E[f] = c, since E[w3^2 - 1] = 0 and E[w4*w5] = 0.
        c = 1.2
        f = c + 0.3 * w[1] - 0.2 * w[2] + 0.5 * (w[3] * w[3] - 1.0) - 0.4 * (w[4] * w[5])

        μ = expectation(f, dists)
        σ2 = variance(f, dists)

        # Var = 0.3^2 + (-0.2)^2 + 0.5^2*Var(w3^2-1) + (-0.4)^2*Var(w4*w5)
        # Var(w3^2-1)=2, Var(w4*w5)=1
        μ_ref = c
        σ2_ref = 0.09 + 0.04 + 0.25 * 2.0 + 0.16 * 1.0

        @test isapprox(μ, μ_ref; rtol = 1e-12, atol = 1e-12)
        @test isapprox(σ2, σ2_ref; rtol = 1e-12, atol = 1e-12)
    end

    @testset "Forward/backward Twiss round-trip" begin
        Dmap = Descriptor(6, 2)
        map_out = make_linear_map(Dmap)

        bi_out = propagate_twiss(map_out, bi; desc = Dmap)
        bi_back = propagate_twiss(map_out, bi_out; backward = true, desc = Dmap)

        @test isapprox(bi_back.σδ, bi.σδ; rtol = 1e-10, atol = 1e-12)
        @test isapprox(bi_back.εx, bi.εx; rtol = 1e-10, atol = 1e-12)
        @test isapprox(bi_back.βx, bi.βx; rtol = 1e-10, atol = 1e-11)
        @test isapprox(bi_back.αx, bi.αx; rtol = 1e-10, atol = 1e-11)
        @test isapprox(bi_back.ηx, bi.ηx; rtol = 1e-10, atol = 1e-11)
        @test isapprox(bi_back.ηpx, bi.ηpx; rtol = 1e-10, atol = 1e-11)
        @test isapprox(bi_back.εy, bi.εy; rtol = 1e-10, atol = 1e-12)
        @test isapprox(bi_back.βy, bi.βy; rtol = 1e-10, atol = 1e-11)
        @test isapprox(bi_back.αy, bi.αy; rtol = 1e-10, atol = 1e-11)
    end

    @testset "Linear covariance limit" begin
        Dmap = Descriptor(6, 1)
        map_lin = make_linear_map(Dmap)

        stats = propagate_sigma(map_lin, bi; desc = Dmap)
        J = GTPSA.jacobian(map_lin)
        Σin = twiss_to_sigma(bi)
        Σlin = J * Σin * transpose(J)

        @test isapprox(stats.cov, Σlin; rtol = 1e-10, atol = 1e-12)
    end

    if get(ENV, "POLYCHAOS_PLOT_TESTS", "0") == "1"
        @testset "Metrics and plot artifacts" begin
            Dmap = Descriptor(6, 2)
            s_grid = collect(range(0.2, 4.0; length = 20))

            s_list = Float64[]
            sigma_x_list = Float64[]
            sigma_y_list = Float64[]
            beta_x_list = Float64[]
            beta_y_list = Float64[]
            eta_x_list = Float64[]

            for s in s_grid
                map_out = make_linear_map(Dmap; sx = s, sy = 0.7 * s, dx = 0.12, dpx = -0.03)
                stats = propagate_sigma(map_out, bi; desc = Dmap)
                Σ = stats.cov

                Σ11 = Σ[1, 1]
                Σ22 = Σ[2, 2]
                Σ12 = Σ[1, 2]
                Σ33 = Σ[3, 3]
                Σ44 = Σ[4, 4]
                Σ34 = Σ[3, 4]
                Σ16 = Σ[1, 6]
                Σ66 = Σ[6, 6]

                push!(s_list, s)
                push!(sigma_x_list, sqrt(max(Σ11, 0.0)))
                push!(sigma_y_list, sqrt(max(Σ33, 0.0)))

                det_x = Σ11 * Σ22 - Σ12 * Σ12
                eps_x = sqrt(max(det_x, 0.0))
                push!(beta_x_list, eps_x > 1e-20 ? Σ11 / eps_x : NaN)

                det_y = Σ33 * Σ44 - Σ34 * Σ34
                eps_y = sqrt(max(det_y, 0.0))
                push!(beta_y_list, eps_y > 1e-20 ? Σ33 / eps_y : NaN)

                push!(eta_x_list, Σ66 > 1e-20 ? Σ16 / Σ66 : NaN)
            end

            metrics = (;
                s = s_list,
                sigma_x = sigma_x_list,
                sigma_y = sigma_y_list,
                beta_x = beta_x_list,
                beta_y = beta_y_list,
                eta_x = eta_x_list,
            )

            outdir = get(ENV, "POLYCHAOS_PLOT_DIR", joinpath(@__DIR__, "artifacts"))
            include(joinpath(@__DIR__, "plot_test_results.jl"))
            files = plot_tracking_metrics(metrics; outdir = outdir)
            @test length(files) == 6
            for f in files
                @test isfile(f)
            end
        end
    end
end

