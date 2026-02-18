"""
    BeamInit(; x0, px0, y0, py0, δ0, σδ, εx, βx, αx, ηx, ηpx, εy, βy, αy)

Beam initial conditions: centroid (x0, px0, y0, py0, δ0), rms relative momentum spread `σδ`,
horizontal/vertical Courant–Snyder (ε, β, α), and horizontal dispersion (ηx, ηpx).
Used with `cholesky_factor`, `propagate_sigma`, and `propagate_twiss`.
"""
Base.@kwdef struct BeamInit{T<:Real}
    x0::T
    px0::T
    y0::T
    py0::T
    δ0::T
    σδ::T
    εx::T
    βx::T
    αx::T
    ηx::T
    ηpx::T
    εy::T
    βy::T
    αy::T
end

@inline function _gamma(β::T, α::T) where {T}
    β > zero(T) || throw(ArgumentError("beta must be > 0"))
    return (one(T) + α * α) / β
end

"""
    cholesky_factor(bi::BeamInit)

Build the 6x5 matrix mapping 5 standard normals to `(x, px, y, py, z, δ)`.
"""
function cholesky_factor(bi::BeamInit{T}) where {T<:Real}
    sqεx = sqrt(bi.εx)
    sqεy = sqrt(bi.εy)
    sqβx = sqrt(bi.βx)
    sqβy = sqrt(bi.βy)

    Lδx = bi.ηx * bi.σδ
    Lδpx = bi.ηpx * bi.σδ

    # Intermediate order rows: (δ, x, px, y, py, z)
    L = zeros(T, 6, 5)
    L[1, 1] = bi.σδ
    L[2, 1] = Lδx
    L[2, 2] = sqεx * sqβx
    L[3, 1] = Lδpx
    L[3, 2] = -sqεx * bi.αx / sqβx
    L[3, 3] = sqεx / sqβx
    L[4, 4] = sqεy * sqβy
    L[5, 4] = -sqεy * bi.αy / sqβy
    L[5, 5] = sqεy / sqβy

    # Permute rows to BeamTracking order: (x, px, y, py, z, δ)
    p = (2, 3, 4, 5, 6, 1)
    return L[collect(p), :]
end

"""
    twiss_to_sigma(bi::BeamInit)

Construct 6x6 covariance matrix in `(x, px, y, py, z, δ)` order.
"""
function twiss_to_sigma(bi::BeamInit{T}) where {T<:Real}
    Σ = zeros(T, 6, 6)
    σδ2 = bi.σδ * bi.σδ

    γx = _gamma(bi.βx, bi.αx)
    γy = _gamma(bi.βy, bi.αy)

    Σxβ11 = bi.εx * bi.βx
    Σxβ12 = -bi.εx * bi.αx
    Σxβ22 = bi.εx * γx

    # Add dispersion contributions to horizontal plane.
    Σ[1, 1] = Σxβ11 + σδ2 * bi.ηx * bi.ηx
    Σ[1, 2] = Σxβ12 + σδ2 * bi.ηx * bi.ηpx
    Σ[2, 1] = Σ[1, 2]
    Σ[2, 2] = Σxβ22 + σδ2 * bi.ηpx * bi.ηpx

    Σ[1, 6] = σδ2 * bi.ηx
    Σ[6, 1] = Σ[1, 6]
    Σ[2, 6] = σδ2 * bi.ηpx
    Σ[6, 2] = Σ[2, 6]

    Σ[3, 3] = bi.εy * bi.βy
    Σ[3, 4] = -bi.εy * bi.αy
    Σ[4, 3] = Σ[3, 4]
    Σ[4, 4] = bi.εy * γy

    Σ[6, 6] = σδ2
    return Σ
end

@inline function _twiss_from_2x2(Σ11::Float64, Σ12::Float64, Σ22::Float64)
    detv = max(Σ11 * Σ22 - Σ12 * Σ12, 0.0)
    ε = sqrt(detv)
    if ε == 0.0
        return (β = 0.0, α = 0.0, ε = 0.0)
    end
    β = Σ11 / ε
    α = -Σ12 / ε
    return (β = β, α = α, ε = ε)
end

"""
    sigma_to_twiss(μ, Σ; template=nothing)

Extract `BeamInit` from output mean/covariance. Dispersion terms are recovered
from `(x,px)` correlation with `δ`.
"""
function sigma_to_twiss(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    template::Union{Nothing, BeamInit}=nothing,
)
    length(μ) == 6 || throw(ArgumentError("mean vector must have length 6"))
    size(Σ) == (6, 6) || throw(ArgumentError("covariance must be 6x6"))

    σδ2 = max(float(Σ[6, 6]), 0.0)
    σδ = sqrt(σδ2)
    ηx = σδ2 > 0.0 ? float(Σ[1, 6]) / σδ2 : 0.0
    ηpx = σδ2 > 0.0 ? float(Σ[2, 6]) / σδ2 : 0.0

    Σx11 = float(Σ[1, 1]) - σδ2 * ηx * ηx
    Σx12 = float(Σ[1, 2]) - σδ2 * ηx * ηpx
    Σx22 = float(Σ[2, 2]) - σδ2 * ηpx * ηpx
    xt = _twiss_from_2x2(max(Σx11, 0.0), Σx12, max(Σx22, 0.0))

    # Vertical plane can be extracted directly for this uncoupled model.
    yt = _twiss_from_2x2(max(float(Σ[3, 3]), 0.0), float(Σ[3, 4]), max(float(Σ[4, 4]), 0.0))

    return BeamInit(
        x0 = float(μ[1]),
        px0 = float(μ[2]),
        y0 = float(μ[3]),
        py0 = float(μ[4]),
        δ0 = float(μ[6]),
        σδ = σδ,
        εx = xt.ε,
        βx = xt.β,
        αx = xt.α,
        ηx = ηx,
        ηpx = ηpx,
        εy = yt.ε,
        βy = yt.β,
        αy = yt.α,
    )
end

@inline function _standard_normal_vector(n::Int)
    return fill(Normal(0.0, 1.0), n)
end

@inline function _beam_coords_from_w(w::AbstractVector{<:GTPSA.TPS}, bi::BeamInit)
    L = cholesky_factor(bi)
    z0 = (float(bi.x0), float(bi.px0), float(bi.y0), float(bi.py0), 0.0, float(bi.δ0))
    z = Vector{typeof(w[1])}(undef, 6)
    for i in 1:6
        acc = zero(w[1])
        @inbounds for j in 1:5
            acc += float(L[i, j]) * w[j]
        end
        z[i] = z0[i] + acc
    end
    return z
end

"""
    propagate_sigma(map_out, bi; desc=Descriptor(6,2))

Propagate the Gaussian beam covariance through a nonlinear GTPSA map.
Returns `(mean, cov, composed)`.
"""
function propagate_sigma(
    map_out::AbstractVector{<:GTPSA.TPS},
    bi::BeamInit;
    desc::GTPSA.Descriptor=GTPSA.Descriptor(6, 2),
)
    length(map_out) >= 6 || throw(ArgumentError("map_out must have at least 6 TPS outputs"))
    nn = GTPSA.numnn(first(map_out))
    GTPSA.numnn(desc) == nn || throw(ArgumentError("descriptor size must match map descriptor size ($nn)"))

    vars_desc = GTPSA.vars(desc)
    w = vars_desc[1:5]
    z = _beam_coords_from_w(w, bi)

    compose_in = Vector{typeof(vars_desc[1])}(undef, nn)
    compose_in[1:6] = z
    for i in 7:nn
        compose_in[i] = vars_desc[i]
    end

    composed = GTPSA.compose(collect(map_out), compose_in)
    dists = Vector{AbstractDistribution}(undef, nn)
    for i in 1:5
        dists[i] = StandardNormal
    end
    for i in 6:nn
        dists[i] = Deterministic(0.0)
    end
    μ = [expectation(composed[i], dists) for i in 1:6]
    Σ = covariance(composed[1:6], dists)
    return (mean = μ, cov = Σ, composed = composed)
end

"""
    propagate_twiss(map_out, bi; backward=false, desc=Descriptor(6,2))

Forward or backward propagation of uncoupled Twiss/dispersion parameters.
"""
function propagate_twiss(
    map_out::AbstractVector{<:GTPSA.TPS},
    bi::BeamInit;
    backward::Bool=false,
    desc::GTPSA.Descriptor=GTPSA.Descriptor(6, 2),
)
    use_map = backward ? GTPSA.inv(collect(map_out)) : collect(map_out)
    stats = propagate_sigma(use_map, bi; desc=desc)
    return sigma_to_twiss(stats.mean, stats.cov; template=bi)
end

