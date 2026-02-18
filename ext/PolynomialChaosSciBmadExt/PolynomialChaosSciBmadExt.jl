module PolynomialChaosSciBmadExt

using PolynomialChaos
using SciBmad
using TypedTables
using GTPSA
using LinearAlgebra

const BTBL = Base.get_extension(SciBmad.BeamTracking, :BeamTrackingBeamlinesExt)

"""
    propagate_twiss_table(bl::Beamline, bi::BeamInit; desc=GTPSA.Descriptor(6, 2), at=:, linear=false)

Propagate optical lattice parameters from `bi` through `bl`, returning a TypedTable
with Twiss columns similar to SciBmad's `twiss`: `beamline_index`, `name`, `s`,
`beta_1`, `alpha_1`, `eta_1`, `etap_1`, `beta_2`, `alpha_2`, `orbit_x`, `orbit_px`, etc.

Uses Bunch+track! to obtain the transfer map at each location, then `propagate_twiss`.
Requires SciBmad and TypedTables (e.g. `--project=~/.julia/env/lean`).

# Arguments
- `bl`: Beamline to propagate through.
- `bi`: Initial beam conditions (BeamInit).

# Keywords
- `desc`: GTPSA Descriptor for the map (default `Descriptor(6, 2)`).
- `at`: Colon `:` for all elements + end, or a vector of LineElements to restrict output.
- `linear`: If `true`, use fast matrix propagation (J*μ+c, J*Σ*J'); use with `desc=Descriptor(6, 1)`.
            Yields sub-ms for long lattices; matches full PCE for linear optics.
"""
function PolynomialChaos.propagate_twiss_table(bl::Beamline, bi::PolynomialChaos.BeamInit;
    desc::GTPSA.Descriptor=GTPSA.Descriptor(6, 2),
    at::Union{Colon,AbstractVector}=:,
    linear::Bool=false,
)
    line = bl.line
    at = at isa AbstractVector && eltype(at) == LineElement ? sort(collect(at); by=e->e.beamline_index) : at
    N_ele = at isa Colon ? length(line) + 1 : length(at)
    idxs = Vector{Int}(undef, N_ele)
    names = Vector{String}(undef, N_ele)
    s_vals = Vector{Float64}(undef, N_ele)

    scur = 0.0
    idx = 1
    for ele in line
        if at isa Colon || ele in at
            idxs[idx] = ele.beamline_index
            names[idx] = ele.name
            s_vals[idx] = scur
            idx += 1
        end
        scur += ele.L
    end
    if at isa Colon
        idxs[N_ele] = -1
        names[N_ele] = "END OF BEAMLINE"
        s_vals[N_ele] = scur
    end

    vars_D = collect(GTPSA.vars(desc)[1:6])
    b = Bunch(reshape(vars_D, 1, 6); species=bl.species_ref, p_over_q_ref=bl.p_over_q_ref)
    BTBL.check_bl_bunch!(bl, b, false)
    beta_1 = Vector{Float64}(undef, N_ele)
    alpha_1 = Vector{Float64}(undef, N_ele)
    eta_1 = Vector{Float64}(undef, N_ele)
    etap_1 = Vector{Float64}(undef, N_ele)
    beta_2 = Vector{Float64}(undef, N_ele)
    alpha_2 = Vector{Float64}(undef, N_ele)
    orbit_x = Vector{Float64}(undef, N_ele)
    orbit_px = Vector{Float64}(undef, N_ele)
    orbit_y = Vector{Float64}(undef, N_ele)
    orbit_py = Vector{Float64}(undef, N_ele)
    orbit_z = Vector{Float64}(undef, N_ele)
    orbit_pz = Vector{Float64}(undef, N_ele)

    vars_D = collect(GTPSA.vars(desc)[1:6])
    b.coords.v .= reshape(vars_D, 1, 6)
    row = 1

    if linear
        μ_init = Float64[bi.x0, bi.px0, bi.y0, bi.py0, 0.0, bi.δ0]
        Σ_init = PolynomialChaos.twiss_to_sigma(bi)
        x0 = zeros(6)
        record_linear!() = begin
            bi_out = PolynomialChaos.sigma_to_twiss(μ, Σ)
            beta_1[row] = bi_out.βx
            alpha_1[row] = bi_out.αx
            eta_1[row] = bi_out.ηx
            etap_1[row] = bi_out.ηpx
            beta_2[row] = bi_out.βy
            alpha_2[row] = bi_out.αy
            orbit_x[row] = bi_out.x0
            orbit_px[row] = bi_out.px0
            orbit_y[row] = bi_out.y0
            orbit_py[row] = bi_out.py0
            orbit_z[row] = 0.0
            orbit_pz[row] = bi_out.δ0
            row += 1
        end
        # At each step, map is composed from start; apply to initial μ,Σ (not previous)
        μ = copy(μ_init)
        Σ = copy(Σ_init)
        (at isa Colon || line[1] in at) && record_linear!()
        for i in 1:length(line)
            track!(b, line[i])
            map_out = vec(b.coords.v)
            c = GTPSA.evaluate(map_out, x0)
            J = GTPSA.jacobian(map_out)
            μ = J * μ_init .+ c
            Σ = J * Σ_init * J'
            (at isa Colon || (i < length(line) && line[i + 1] in at)) && record_linear!()
        end
    else
        record!() = begin
            bi_out = PolynomialChaos.propagate_twiss(vec(b.coords.v), bi; desc=desc)
            beta_1[row] = bi_out.βx
            alpha_1[row] = bi_out.αx
            eta_1[row] = bi_out.ηx
            etap_1[row] = bi_out.ηpx
            beta_2[row] = bi_out.βy
            alpha_2[row] = bi_out.αy
            orbit_x[row] = bi_out.x0
            orbit_px[row] = bi_out.px0
            orbit_y[row] = bi_out.y0
            orbit_py[row] = bi_out.py0
            orbit_z[row] = 0.0
            orbit_pz[row] = bi_out.δ0
            row += 1
        end
        (at isa Colon || line[1] in at) && record!()
        for i in 1:length(line)
            track!(b, line[i])
            (at isa Colon || (i < length(line) && line[i + 1] in at)) && record!()
        end
    end

    return Table(;
        beamline_index=idxs,
        name=names,
        s=s_vals,
        beta_1, alpha_1, eta_1, etap_1,
        beta_2, alpha_2,
        orbit_x, orbit_px, orbit_y, orbit_py, orbit_z, orbit_pz,
    )
end

end
