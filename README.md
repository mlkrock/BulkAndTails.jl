
# BATsDistribution.jl

A package implementing the "Bulk-And-Tails" (BATs) distribution along with
maximum likelihood estimation of its parameters [1,2]. Due to the fact that
this estimation requires solving a nonlinear optimization problem with nonlinear constraints, 
this functionality is wrapped into its own package to compartmentalize the
dependencies on the several required dependencies.

# Discussion
BATs is a seven-parameter univariate distribution with flexible behavior in both tails.
Unlike classical methods for extremes (e.g., generalized Pareto distribution, generalized
extreme value distribution), which only fit a single tail of a distribution, BATs models the
entire distribution (i.e., the bulk and both tails). BATs has seven parameters (κ₀,τ₀,ϕ₀,κ₁,τ₁,ϕ₁,ν)
in total: shape parameters κ which control tail behavior, location parameters ϕ, scale
parameters τ, and ν degrees of freedom of a student-t distribution. The subscript 0 
refers to lower tail, and the subscript 1 refers to upper tail. If κ is negative, that tail is 
bounded; if κ is zero (defined by continuity), that tail is thin like a Gaussian tail; and if
κ is positive, that tail is a heavy tail.

# Demonstration

See the example file for a more heavily commented discussion of this same
demonstration. It may be necessary to try several initializations to the
optimization.

````{julia}
  using Distributions, BATsDistribution
  data = rand(TDist(1.0), 5_000) # 5k samples from Cauchy distribution
  (mle, obs_information_matrix) = fit_mle(BATs, data)
  pdf(BATs(mle), 10.0) # compare with Cauchy pdf at 10.0.
````

# Future Enhancements

1) For plenty of distributions with a known lower bound, it is possible to
simplify the model parameterization and fit the model with that known value. Not
doing that can cause slight issues in some cases due to the currently enforced
constraint on the tail index that guarantees second derivatives of the density
at the support endpoints. Incorporating this information and re-organizing the
estimation code is very doable, it just will take some time and we haven't done
it yet. This would be a fine PR for users interested in studying the
distribution more. 

2) Similar for upper bounds, and for both lower and upper bounds.

3) In principle, analytical derivatives for the PDF could be computed and
hard-coded in to avoid using automatic differentiation. But considering how fast
the code already is, this is not a priority.

# References

[1] Stein, M. L. (2021).  A parametric model for distributions with flexible behavior in both tails. Environmetrics, 32(2):Paper No. e2658, 24. (https://onlinelibrary.wiley.com/doi/abs/10.1002/env.2658)

[2] Krock, M., Bessac, J., Stein, M. L., and Monahan, A. H. Nonstationary seasonal model for daily mean temperature distribution bridging bulk and tails. (https://arxiv.org/pdf/2110.10046.pdf)

# Authors
Mitchell Krock <mk1867@stat.rutgers.edu> (active development)

Julie Bessac <jbessac@anl.gov> (active development)

Chris Geoga <christopher.geoga@rutgers.edu> (base implementation)

