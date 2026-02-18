@inline function _check_dist_count(t::GTPSA.TPS, dists)
    n = length(dists)
    n > 0 || throw(ArgumentError("at least one distribution is required"))
    nd = GTPSA.numnn(t)
    n <= nd || throw(ArgumentError("distribution count $n exceeds descriptor size $nd"))
    return n
end

"""
    collect_monomials(t::GTPSA.TPS, nvars::Int=GTPSA.numnn(t))

Collect nonzero monomials of a TPS as `(exponents, coefficient)` tuples.
"""
function collect_monomials(t::GTPSA.TPS, nvars::Int=GTPSA.numnn(t))
    nvars > 0 || throw(ArgumentError("nvars must be > 0"))
    mono = Vector{UInt8}(undef, nvars)
    coef = Ref{GTPSA.numtype(t)}()
    idx = -1
    out = Vector{Tuple{Vector{UInt8}, GTPSA.numtype(t)}}()
    idx = GTPSA.cycle!(t, idx, nvars, mono, coef)
    while idx >= 0
        push!(out, (copy(mono), coef[]))
        idx = GTPSA.cycle!(t, idx, nvars, mono, coef)
    end
    return out
end

@inline function _moment_product(exponents::AbstractVector{UInt8}, dists)
    acc = 1.0
    @inbounds for i in eachindex(dists)
        m = raw_moment(dists[i], Int(exponents[i]))
        m == 0.0 && return 0.0
        acc *= m
    end
    return acc
end

"""
    expectation(t::GTPSA.TPS, dists)

Compute `E[t]` for independent random variables with given marginal distributions.
"""
function expectation(t::GTPSA.TPS, dists::AbstractVector{<:AbstractDistribution})
    nvars = _check_dist_count(t, dists)
    mono = Vector{UInt8}(undef, nvars)
    coef = Ref{GTPSA.numtype(t)}()
    idx = -1
    out = 0.0
    idx = GTPSA.cycle!(t, idx, nvars, mono, coef)
    while idx >= 0
        out += float(coef[]) * _moment_product(mono, dists)
        idx = GTPSA.cycle!(t, idx, nvars, mono, coef)
    end
    return out
end

expectation(x::Number, dists::AbstractVector{<:AbstractDistribution}) = float(x)

"""
    cross_expectation(t1, t2, dists)

Compute `E[t1 * t2]` for independent random variables.
"""
function cross_expectation(
    t1::GTPSA.TPS,
    t2::GTPSA.TPS,
    dists::AbstractVector{<:AbstractDistribution},
)
    nvars = _check_dist_count(t1, dists)
    _check_dist_count(t2, dists)

    m1 = collect_monomials(t1, nvars)
    m2 = collect_monomials(t2, nvars)
    e = zeros(UInt16, nvars)

    out = 0.0
    @inbounds for (exp1, c1) in m1
        for (exp2, c2) in m2
            for i in 1:nvars
                e[i] = UInt16(exp1[i]) + UInt16(exp2[i])
            end
            mp = 1.0
            for i in 1:nvars
                m = raw_moment(dists[i], Int(e[i]))
                if m == 0.0
                    mp = 0.0
                    break
                end
                mp *= m
            end
            out += float(c1) * float(c2) * mp
        end
    end
    return out
end

cross_expectation(x::Number, y::Number, dists::AbstractVector{<:AbstractDistribution}) = float(x) * float(y)
cross_expectation(t::GTPSA.TPS, y::Number, dists::AbstractVector{<:AbstractDistribution}) = expectation(t, dists) * float(y)
cross_expectation(x::Number, t::GTPSA.TPS, dists::AbstractVector{<:AbstractDistribution}) = float(x) * expectation(t, dists)

"""
    variance(t, dists)

Compute `Var[t] = E[t^2] - E[t]^2`.
"""
function variance(t::Union{GTPSA.TPS, Number}, dists::AbstractVector{<:AbstractDistribution})
    μ = expectation(t, dists)
    return max(cross_expectation(t, t, dists) - μ * μ, 0.0)
end

"""
    covariance(ts, dists)

Compute covariance matrix for a vector of TPS objects.
"""
function covariance(ts::AbstractVector, dists::AbstractVector{<:AbstractDistribution})
    n = length(ts)
    μ = [expectation(ts[i], dists) for i in 1:n]
    Σ = Matrix{Float64}(undef, n, n)
    for i in 1:n
        Σ[i, i] = max(cross_expectation(ts[i], ts[i], dists) - μ[i] * μ[i], 0.0)
        for j in (i + 1):n
            cij = cross_expectation(ts[i], ts[j], dists) - μ[i] * μ[j]
            Σ[i, j] = cij
            Σ[j, i] = cij
        end
    end
    return Σ
end

std(t::Union{GTPSA.TPS, Number}, dists::AbstractVector{<:AbstractDistribution}) = sqrt(variance(t, dists))

