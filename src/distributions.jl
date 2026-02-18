"""
    AbstractDistribution

Abstract type for univariate distributions used in PCE moment computation.
"""
abstract type AbstractDistribution end

"""
    Normal(mu, sigma)

Normal distribution; `raw_moment(d, n)` gives E[X^n].
"""
struct Normal{T<:Real} <: AbstractDistribution
    mu::T
    sigma::T
end

"""
    Uniform(a, b)

Uniform distribution on [a, b]; `raw_moment(d, n)` gives E[X^n].
"""
struct Uniform{T<:Real} <: AbstractDistribution
    a::T
    b::T
end

"""
    Deterministic(value)

Degenerate distribution at `value`; `raw_moment(d, n)` returns value^n.
"""
struct Deterministic{T<:Real} <: AbstractDistribution
    value::T
end

"""Standard normal N(0, 1)."""
const StandardNormal = Normal(0.0, 1.0)

@inline function _double_factorial_odd(n::Int)
    n <= 0 && return 1.0
    acc = 1.0
    k = n
    while k > 1
        acc *= k
        k -= 2
    end
    return acc
end

@inline function _std_normal_raw_moment(n::Int)
    n < 0 && throw(ArgumentError("moment order must be >= 0"))
    isodd(n) && return 0.0
    return _double_factorial_odd(n - 1)
end

"""
    raw_moment(d, n)

Raw moment E[X^n] for distribution `d` and nonnegative integer `n`.
"""
@inline function raw_moment(d::Normal, n::Int)
    n < 0 && throw(ArgumentError("moment order must be >= 0"))
    if d.sigma < 0
        throw(ArgumentError("Normal sigma must be >= 0"))
    end
    n == 0 && return 1.0

    mu = float(d.mu)
    sigma = float(d.sigma)
    total = 0.0
    for k in 0:n
        zk = _std_normal_raw_moment(k)
        zk == 0.0 && continue
        total += binomial(n, k) * (mu^(n - k)) * (sigma^k) * zk
    end
    return total
end

@inline function raw_moment(d::Uniform, n::Int)
    n < 0 && throw(ArgumentError("moment order must be >= 0"))
    a = float(d.a)
    b = float(d.b)
    if !(b > a)
        throw(ArgumentError("Uniform requires b > a"))
    end
    n == 0 && return 1.0
    return (b^(n + 1) - a^(n + 1)) / ((n + 1) * (b - a))
end

@inline function raw_moment(d::Deterministic, n::Int)
    n < 0 && throw(ArgumentError("moment order must be >= 0"))
    n == 0 && return 1.0
    return float(d.value)^n
end

