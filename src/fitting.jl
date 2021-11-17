
##
# Parameter layout: (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) 
# The optimization: note that this model has data-determined nonlinear contraints
#
#   0.0 > ϕ₀ - τ₀*iΨ(-1.0/κ₀) - minimum(data) **IF κ₀ < 0.0**
#   0.0 < ϕ₁ + τ₁*iΨ(-1.0/κ₁) - maximum(data) **IF κ₁ < 0.0**
#   κ₀/ν > -0.5  # two derivatives at endpoint
#   κ₁/ν > -0.5  # two derivatives at endpoint
#
# The way that Ipopt does handles these is by putting box constraints on the
# nonlinear argument functions. The functions constr1 and constr2 give those
# constraints in mathematical language. The code representation for Ipopt is
# much harder to interpret.
#
##


# This function is constrained to be NEGATIVE. It is not differentiable at κ=0,
# and so the jacobian should break there. 
# 
# In math:
# 0.0 > ϕ₀ - τ₀*iΨ(-1.0/κ₀) - minimum(data) **IF κ₀ < 0.0**
function constr1(p, mindat)
  (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) = p
  κ₀ ≥ 0.0 && return -1.0 # If κ₀>0, return a very feasible number.
  ϕ₀ - τ₀*iΨ(-1.0/κ₀) - mindat
end

# This function is constrained to be POSITIVE. It is not differentiable at κ=0,
# and so the jacobian should break there. 
#
# In math:
# 0.0 < ϕ₁ + τ₁*iΨ(-1.0/κ₁) - maximum(data) **IF κ₁ < 0.0**
function constr2(p, maxdat)
  (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) = p
  κ₁ ≥ 0.0 && return 1.0 # If κ₁>0, return a very feasible number.
  ϕ₁ + τ₁*iΨ(-1.0/κ₁) - maxdat
end

# Tail indices:
@inline constr3(p) = p[1]/p[7]
@inline constr4(p) = p[4]/p[7]

function Distributions.fit_mle(::Type{BulkAndTailsDist}, data::AbstractArray; 
                               init=[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0], 
                               tol=1.0e-5, maxit=2000, print_level=0)

  # Box constraints (lowerb and upperb) and inequality constraints for the way
  # that I've implemented the nonlinear constraints.
  lowerb = [-1e22,  0.0,  -1e22, -1e22,  0.0,  -1e22, 0.0]
  upperb = [1e22,  1e22,   1e22,  1e22, 1e22,   1e22, 1e22]
  cons_low = [-1e22, 0.0, -0.5, -0.5]
  cons_upp = [0.0,  1e22, 1e22, 1e22]

  # objective, gradient, and Hessian, but with a cache to save some intermediate
  # results and cut down on repeated calculations.
  obj_cache  = Dict{UInt64, DiffResults.MutableDiffResult}()
  # objective function:
  obj = p->-sum(x->batslogpdf(x, p), data)
  # a method to add keys to the cache, which effectively computes the hessian
  # and gradient of obj and stores them for later access.
  addkey! = p -> begin
    haskey(obj_cache, hash(p)) && return
    res = DiffResults.HessianResult(p)
    ForwardDiff.hessian!(res, obj, p)
    obj_cache[hash(p)] = res
  end
  # a gradient for obj, which either gets it from the cache or adds it to the
  # cache and then gets it from the cache.
  grad! = (p,g) -> begin
    if haskey(obj_cache, hash(p))
      g .= DiffResults.gradient(obj_cache[hash(p)])
      return nothing
    end
    addkey!(p)
    grad!(p,g)
  end
  # analogous functionality for the Hessian.
  hess = p -> begin
    haskey(obj_cache, hash(p)) && return DiffResults.hessian(obj_cache[hash(p)])
    addkey!(p)
    hess(p)
  end

  # Writing the constraints in several forms, all of which are necessary for the
  # various ways that we use the constraints. Note that these constraints also
  # are involved in what ultimately gets given to Ipopt as the "hessian",
  # because the objective function gets a scaled constraint function added to it.
  #
  # (i):  one argument form
  # (ii): a tuple of one-argument functions
  # (iii): a one-argument vector-valued function
  _constr1(p) = constr1(p, minimum(data)) # (i)
  _constr2(p) = constr2(p, maximum(data)) # (i)
  constrf     = (_constr1, _constr2, constr3, constr4) # (ii)
  constr(p)   = [cons_f(p) for cons_f in constrf] # (iii)

  # Write the constraint in the final format that ipopt actually wants.
  ipopt_constr(p,g) = (g .= constr(p))

  # A properly reshaped jacobian (not very pretty sorry).
  g_jac(p) = vec(Matrix(ForwardDiff.jacobian(constr, p)'))

  # Prepare the final hessian and constraint jacobian functions.
  nconstr  = length(constrf)
  constrfh = [p->ForwardDiff.hessian(cj, p) for cj in constrf]
  ipopt_hess = (x,m,r,c,o,l,v) -> ipopt_hessian(x,m,r,c,o,l,v,hess,constrfh,nconstr)
  ipopt_jac_constr = (x,m,r,c,v) -> ipopt_constr_jac(x,m,r,c,v,g_jac,nconstr)

  # Set up the problem:
  prob = createProblem(length(init), # number of parameters
                       lowerb, upperb, # box for parameter constraints
                       length(constrf), # number of nonlinear constraints
                       cons_low, cons_upp, # box for nonlinear constraint fxns
                       length(init)*length(constrf), # size of CONSTRAINT JACOBIAN
                       div(length(init)*(length(init)+1), 2), # size of ltri of HESSIAN
                       obj,   
                       ipopt_constr,
                       grad!, 
                       ipopt_jac_constr,
                       ipopt_hess)
  addOption(prob, "sb", "yes")
  addOption(prob, "tol", tol)
  addOption(prob, "max_iter", Int(maxit))
  addOption(prob, "print_level", Int(print_level))
  prob.x = deepcopy(init)
  status = solveProblem(prob)

  # If a solution was found, return it. Otherwise throw an error. 
  iszero(status) && return (BulkAndTailsDist(prob.x), Matrix(hess(prob.x)))
  throw(error("Optimization failed with Ipopt return code $status."))
end

# for R users:
function fitbats(data; kwargs...) 
  (mle, hes) = fit_mle(BulkAndTailsDist, data; kwargs...)
  [getfield(mle, f) for f in fieldnames(BulkAndTailsDist)], hes
end

