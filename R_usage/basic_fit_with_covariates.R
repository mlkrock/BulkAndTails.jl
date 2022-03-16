## Replicating the fit of daily average temperature data at Boston.

# The library that lets R talk to Julia easily and efficiently.
library(JuliaCall)

# You need to run these every time you start R.
julia_setup()
julia_library("BulkAndTails")

## loading the data

data = read.table("../data/boston.csv")
dat = data[, 1] #data values
yday = data[, 2] #day of year
year = data[, 3] #year
jday = data[, 4] #Julian day since first observation day

climatetrend = read.table("../data/logco2eq.csv")

climate_trend = rep(NA,length(dat))
for (i in 1:length(dat)){
  which_year = year[i]
  which_year_index = which(climatetrend[,1] == which_year)
  climate_trend[i] = climatetrend[which_year_index,2]
}

## setting up the covariate matrices

nsplines = 8
basismatrix = pbs::pbs(jday %% 365.25,df=nsplines, Boundary.knots = c(0,365.25)) #using R package pbs for periodic splines
seasonalmatrix = cbind(cos(2*pi*jday/365.25), sin(2*pi*jday/365.25))

phimatrix = cbind(rep(1.0, length(jday)), basismatrix, climate_trend, climate_trend*seasonalmatrix)
taumatrix = cbind(rep(1.0, length(jday)), basismatrix)

## example initial guess

regcoef = lm(dat ~ basismatrix)$coef

phi0init = c(regcoef, rep(0,3))
phi1init = c(regcoef, rep(0,3))
kappa0init = 0.0
kappa1init = 0.0
nuinit = 1.0
tau0init = rep(0,nsplines+1)
tau1init = rep(0,nsplines+1)

initguess = c(kappa0init, tau0init, phi0init, kappa1init, tau1init, phi1init, nuinit)

## fit
# note this takes 61 iterations and converges after about 5 mins on my laptop
# this will return an answer even if the software hasn't converged, see status below

mle = julia_call("fitbats_covariates", dat, taumatrix, phimatrix, initguess, print_level=5L)

# status description copied from https://github.com/jump-dev/Ipopt.jl/blob/eec0d01b685ff344444a6c77ebf89e9c301cb5d0/src/MOI_wrapper.jl#L1231
# From Ipopt/src/Interfaces/IpReturnCodes_inc.h
# const _STATUS_CODES = Dict(
#   0 => :Solve_Succeeded,
#   1 => :Solved_To_Acceptable_Level,
#   2 => :Infeasible_Problem_Detected,
#   3 => :Search_Direction_Becomes_Too_Small,
#   4 => :Diverging_Iterates,
#   5 => :User_Requested_Stop,
#   6 => :Feasible_Point_Found,
#   -1 => :Maximum_Iterations_Exceeded,
#   -2 => :Restoration_Failed,
#   -3 => :Error_In_Step_Computation,
#   -4 => :Maximum_CpuTime_Exceeded,
#   -5 => :Maximum_WallTime_Exceeded,
#   -10 => :Not_Enough_Degrees_Of_Freedom,
#   -11 => :Invalid_Problem_Definition,
#   -12 => :Invalid_Option,
#   -13 => :Invalid_Number_Detected,
#   -100 => :Unrecoverable_Exception,
#   -101 => :NonIpopt_Exception_Thrown,
#   -102 => :Insufficient_Memory,
#   -199 => :Internal_Error,
# )