
# BATs pdf, cdf, quantile, sampler defined below
# Parameter layout: (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) 
# (κ₀, τ₀, ϕ₀) are shape, scale, location for lower Tail
# (κ₁, τ₁, ϕ₁) are shape, scale, location for upper Tail

# Psi function, its derivative, and its inverse. Branched like this to stabilize
# numerics for large argument x.
Ψ(x)  = (x < 25.0) ? log1p(exp(x))           : x
dΨ(x) = (x < 25.0) ? exp(x)/(1.0+exp(x))     : 1.0
iΨ(x) = (x < 25.0) ? log(max(0.0, expm1(x))) : x # avoids an error with log of negative number

# One half of the H function, and the extension relevant for the density.
# Trying to be careful here about numerics, but it is kind of fishy as κ->0.
function H_part(x, κ, τ, ϕ)
  if iszero(κ)
    return exp(Ψ((x - ϕ) / τ))
  elseif isapprox(κ, 0.0, atol = 1.0e-8) # cutoff chosen by eye.
    a = Ψ((x - ϕ) / τ)
    return exp(a) - 0.5 * (a^2) * exp(a) * κ
  else
    return (1.0 + κ * Ψ((x - ϕ) / τ))^(1.0 / κ)
  end
end

H(x, κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁) = H_part(x, κ₁, τ₁, ϕ₁) - H_part(-x, κ₀, τ₀, -ϕ₀)
dH_part(x, κ, τ, ϕ) = ForwardDiff.derivative(z -> H_part(z, κ, τ, ϕ), x)

# Full PDF in Distributions.jl format. Note that the numerics
# here can be concerning: the return line out_1*(out_2+out_3) can commonly look
# like subnormal_tiny * (super_huge + super_huge), which is a floating point
# nightmare.
function Distributions.pdf(d::BulkAndTailsDist, x::Real) 
  # Check support.
  ((x ≥ maximum(d)) || (x ≤ minimum(d))) && return 0.0
  # Unpack parameters for readability.
  (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) = params(d)
  # Evaluate the density if we're inside the defined support.
  out_1 = Distributions.pdf(TDist(ν),H(x,κ₀,τ₀,ϕ₀,κ₁,τ₁,ϕ₁))
  out_2 = dH_part(x, κ₁, τ₁, ϕ₁)
  out_3 = dH_part(-x, κ₀, τ₀, -ϕ₀)
  max(0.0, out_1*(out_2+out_3)) # numerically dubious.
end

# Note: this t-cdf does not work with autodiff. Would need hand-coded tcdf function.
function Distributions.cdf(d::BulkAndTailsDist, x::Real) 
  if x ≤ minimum(d)
    return 0.0
  elseif x ≥ maximum(d)
    return 1.0
  else
    (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) = params(d)
    return Distributions.cdf(TDist(ν),H(x,κ₀,τ₀,ϕ₀,κ₁,τ₁,ϕ₁))
  end
end

# Roots.Bisection() is safe but relatively slow.
# Something like Roots.Brent() is faster, but seems to fail for positive κ without lowering rtol to 0.0.
function Distributions.quantile(d::BulkAndTailsDist, p::Real; method = Roots.Bisection(),  kwargs...)
  Roots.find_zero(x -> Distributions.cdf(d,x) - p, (minimum(d),maximum(d)), method; kwargs...)
end

function Distributions.logpdf(d::BulkAndTailsDist, x::Real) 
  # Check support. 
  ((x ≥ maximum(d)) || (x ≤ minimum(d))) && return -Inf
  # Unpack parameters for readability.
  (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) = params(d)
  # Evaluate the density if we're inside the defined support.
  out_1 = Distributions.logpdf(TDist(ν),H(x,κ₀,τ₀,ϕ₀,κ₁,τ₁,ϕ₁))
  out_2 = dH_part(x, κ₁, τ₁, ϕ₁)
  out_3 = dH_part(-x, κ₀, τ₀, -ϕ₀)
  out_1 + log(out_2+out_3)
end

# Note: this t-cdf does not work with autodiff. Would need hand-coded logtcdf function.
function Distributions.logcdf(d::BulkAndTailsDist, x::Real) 
  if x ≤ minimum(d)
    return -Inf
  elseif x ≥ maximum(d)
    return 0.0
  else
    (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) = params(d)
    return Distributions.logcdf(TDist(ν),H(x,κ₀,τ₀,ϕ₀,κ₁,τ₁,ϕ₁))
  end
end

function Base.rand(d::BulkAndTailsDist; rng=Random.GLOBAL_RNG, kwargs...) 
  quantile(d,rand(rng); kwargs...)
end

# Define R-friendly version.
batspdf(x, parms) = Distributions.pdf(BulkAndTailsDist(parms),x)
batscdf(x, parms) = Distributions.cdf(BulkAndTailsDist(parms),x)
batsquantile(x, parms; kwargs...) = Distributions.quantile(BulkAndTailsDist(parms),x; kwargs...)
batslogpdf(x, parms) = Distributions.logpdf(BulkAndTailsDist(parms),x)
batslogcdf(x, parms) = Distributions.logcdf(BulkAndTailsDist(parms),x)
batsrand(nsamples, parms) = rand(BulkAndTailsDist(parms),nsamples)

# Automatically broadcasts to vectors except for the pdf related functions and the quantile with kwargs.
batspdf(xv::Vector, parms) = Distributions.pdf.(BulkAndTailsDist(parms),xv)
batslogpdf(xv::Vector, parms) = Distributions.logpdf.(BulkAndTailsDist(parms),xv)
batsquantile(xv::Vector, parms; kwargs...) = Distributions.quantile.(BulkAndTailsDist(parms),xv; kwargs...)