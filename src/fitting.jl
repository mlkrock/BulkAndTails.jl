
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
                               tol=1.0e-5, maxit::Int=2000, print_level=0)

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
  ipopt_hess = (x,r,c,o,l,v) -> ipopt_hessian(x,r,c,o,l,v,hess,constrfh,nconstr)
  ipopt_jac_constr = (x,r,c,v) -> ipopt_constr_jac(x,r,c,v,g_jac,nconstr)

  # Set up the problem:
  prob = CreateIpoptProblem(
                       length(init), # number of parameters
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
  AddIpoptStrOption(prob, "sb", "yes")
  AddIpoptNumOption(prob, "tol", tol)
  AddIpoptIntOption(prob, "max_iter", maxit)
  AddIpoptIntOption(prob, "print_level", Int(print_level))
  prob.x = deepcopy(init)
  status = IpoptSolve(prob)

  # If a solution was found, return it. Otherwise throw an error. 
  iszero(status) && return (mle = BulkAndTailsDist(prob.x), hess = Matrix(hess(prob.x)), nll = obj(prob.x))
  throw(error("Optimization failed with Ipopt return code $status."))
end

# for R users:
function fitbats(data; kwargs...) 
  (mle, hes, nll) = fit_mle(BulkAndTailsDist, data; kwargs...)
  [getfield(mle, f) for f in fieldnames(BulkAndTailsDist)], hes, nll
end

## version for covariates

function bats_negloglikelihood_covariates(
  data::AbstractArray,
  τmatrix::AbstractArray,
  ϕmatrix::AbstractArray,
  parms
)

  m = size(data, 1)
  nϕ = size(ϕmatrix, 2)
  nτ = size(τmatrix, 2)

  @assert ((isequal(m,size(τmatrix,1))) && isequal(m,size(ϕmatrix,1))) "Dimension mismatch"

  #bit of messy indexing
  κ₀ = parms[1]
  τ₀vec = parms[2:(nτ+1)]
  ϕ₀vec = parms[(nτ+2):(nτ+1+nϕ)]
  κ₁ = parms[nτ+2+nϕ]
  τ₁vec = parms[(nτ+3+nϕ):(2*nτ+2+nϕ)]
  ϕ₁vec = parms[(2*nτ+3+nϕ):(2*nτ+2+2*nϕ)]
  ν = parms[2*nτ+3+2*nϕ]

  eltp = eltype(parms)
  output = zero(eltp)

  @inbounds for i in 1:m
    ϕ₀ = zero(eltp)
    ϕ₁ = zero(eltp)
    logτ₀ = zero(eltp)
    logτ₁ = zero(eltp)

    @inbounds for j in 1:nϕ
      ϕ₀ += ϕ₀vec[j] * ϕmatrix[i, j]
      ϕ₁ += ϕ₁vec[j] * ϕmatrix[i, j]
    end

    @inbounds for j in 1:nτ
      logτ₀ += τ₀vec[j] * τmatrix[i, j]
      logτ₁ += τ₁vec[j] * τmatrix[i, j]
    end
    
    output -= logpdf(BulkAndTailsDist(κ₀,exp(logτ₀),ϕ₀,κ₁,exp(logτ₁),ϕ₁,ν),data[i])
  end

  return output
  
end

# fit
function fit_bats_mle_covariates(data::AbstractArray, τmatrix::AbstractArray, ϕmatrix::AbstractArray, init;
  tol=1.0e-5, maxit::Int=1000, print_level=0)

  # Box constraints (lowerb and upperb) and inequality constraints for the way
  # that I've implemented the nonlinear constraints.
  lowerb = [repeat([-1e22], length(init)-1); 0.0]
  upperb = repeat([1e22], length(init))
  cons_low = [-0.5, -0.5]
  cons_upp = [1e22, 1e22]

  # objective function:
  obj = p -> bats_negloglikelihood_covariates(
    data,
    τmatrix,
    ϕmatrix,
    p
  )

  #cache functionality seems to be slower here, so just calling ForwardDiff directly
  grad!(p, g) = ForwardDiff.gradient!(g, obj, p)
  hess(p) = ForwardDiff.hessian(obj, p)

  # Writing the constraints in several forms, all of which are necessary for the
  # various ways that we use the constraints. Note that these constraints also
  # are involved in what ultimately gets given to Ipopt as the "hessian",
  # because the objective function gets a scaled constraint function added to it.
  #
  # (i):  one argument form
  # (ii): a tuple of one-argument functions
  # (iii): a one-argument vector-valued function
  nϕ = size(ϕmatrix,2)
  nτ = size(τmatrix,2)
  lenparms = length(init)
  function constr1_covariate(p, lenparms) #(i)
    p[1]/p[lenparms]
  end
  function constr2_covariate(p, nτ, nϕ, lenparms) #(i)
    p[2+nτ+nϕ]/p[lenparms]
  end
  _constr1(p) = constr1_covariate(p, lenparms) # (i)
  _constr2(p) = constr2_covariate(p, nτ, nϕ, lenparms) # (i)
  constrf     = (_constr1, _constr2) # (ii)
  constr(p)   = [cons_f(p) for cons_f in constrf] # (iii)

  # Write the constraint in the final format that ipopt actually wants.
  ipopt_constr(p,g) = (g .= constr(p))

  # A properly reshaped jacobian (not very pretty sorry).
  g_jac(p) = vec(Matrix(ForwardDiff.jacobian(constr, p)'))
  #g_jac(p) = vec(Matrix(wrapper(:jacobian, p)'))

  # Prepare the final hessian and constraint jacobian functions.
  nconstr  = length(constrf)
  constrfh = [p->ForwardDiff.hessian(cj, p) for cj in constrf]
  ipopt_hess = (x,r,c,o,l,v) -> ipopt_hessian(x,r,c,o,l,v,hess,constrfh,nconstr)
  ipopt_jac_constr = (x,r,c,v) -> ipopt_constr_jac(x,r,c,v,g_jac,nconstr)

  # Set up the problem:
  prob = CreateIpoptProblem(
  length(init), # number of parameters
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
  AddIpoptStrOption(prob, "sb", "yes")
  AddIpoptNumOption(prob, "tol", tol)
  AddIpoptIntOption(prob, "max_iter", maxit)
  AddIpoptIntOption(prob, "print_level", Int(print_level))
  prob.x = deepcopy(init)
  status = IpoptSolve(prob)

  return (status = status, nll = obj(prob.x), κ₀ = prob.x[1], τ₀ = prob.x[2:(nτ+1)], ϕ₀ = prob.x[(nτ+2):(nτ+1+nϕ)], κ₁ = prob.x[nτ+2+nϕ], τ₁ = prob.x[(nτ+3+nϕ):(2*nτ+2+nϕ)], ϕ₁ = prob.x[(2*nτ+3+nϕ):(2*nτ+2+2*nϕ)], ν = prob.x[2*nτ+3+2*nϕ]) 

end

#for R users, making the output a bit more accessible as a list without unicode characters
function fitbats_covariates(data, τmatrix, ϕmatrix, init; kwargs...) 
  x = fit_bats_mle_covariates(data, τmatrix, ϕmatrix, init; kwargs...)
  return Dict("status" => x.status, "nll" => x.nll, "kappa0" => x.κ₀, "tau0" => x.τ₀, "phi0" => x.ϕ₀, "kappa1" => x.κ₁, "tau1" => x.τ₁, "phi1" => x.ϕ₁, "nu" => x.ν)
end
