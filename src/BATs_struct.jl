
"""
BATs(κ₀,τ₀,ϕ₀,κ₁,τ₁,ϕ₁,ν)
The *BATs (Bulk-And-Tails) distribution* has cumulative distribution function
```math
F(x) = T_ν(H(x))
```
where T_ν is the student-t cdf with ν degrees of freedom and H is a monotone increasing function.
ϕ₀ and ϕ₁ are location parameters for the lower and upper tails.
τ₀ and τ₁ are scale parameters for the lower and upper tails.
κ₀ and κ₁ are shape parameters for the lower and upper tails.
Negative κ indicates a bounded tail. Positive κ indicates a heavy tail.
The case κ=0, defined by continuity, indicates a thin Gaussian tail.

`fitbats` provides maximum likelihood estimation for these parameters. In addition, we provide 
`pdf`, `cdf`, `logpdf`, `logcdf`, `quantile`, and `rand` functions for `BATs`, all of which follow the 
`Distributions.jl` framework. We also provide R-friendly versions of these functions and show how to call them from R.

```julia
BATs(κ₀,τ₀,ϕ₀,κ₁,τ₁,ϕ₁,ν)     # BATs distribution
params(d)        # Get the parameters
minimum(d)       # lower bound of support
maximum(d)       # upper bound of support
```
External links
* [Stein, M. L. (2021).  A parametric model for distributions with flexible behavior in both tails. Environmetrics, 32(2):Paper No. e2658, 24.](https://onlinelibrary.wiley.com/doi/abs/10.1002/env.2658)
"""
struct BATs{T<:Real} <: ContinuousUnivariateDistribution
κ₀::T
τ₀::T
ϕ₀::T
κ₁::T
τ₁::T
ϕ₁::T
ν::T
end

params(d::BATs) = (d.κ₀,d.τ₀,d.ϕ₀,d.κ₁,d.τ₁,d.ϕ₁,d.ν)

Distributions.minimum(d::BATs) = (d.κ₀ >= 0.0 ? -Inf : d.ϕ₀ - d.τ₀*iΨ(-1.0/d.κ₀))
Distributions.maximum(d::BATs) = (d.κ₁ >= 0.0 ? Inf : d.ϕ₁ + d.τ₁*iΨ(-1.0/d.κ₁))

BATs(κ₀::Real,τ₀::Real,ϕ₀::Real,κ₁::Real,τ₁::Real,ϕ₁::Real,ν::Real) = BATs(promote(κ₀,τ₀,ϕ₀,κ₁,τ₁,ϕ₁,ν)...)

function BATs(xv) 
    @assert length(xv) == 7 "Please provide 7 arguments."
    BATs(xv...) 
end