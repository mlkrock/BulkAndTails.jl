# Replicating the fit of daily average temperature data at Boston.

using DelimitedFiles, RCall

## loading the data

data = readdlm("../data/boston.csv")
const dat = data[:, 1] #data values
const yday = Int64.(data[:, 2]) #day of year
const year = Int64.(data[:, 3]) #year
const jday = Int64.(data[:, 4]) #Julian day since first observation day
data = nothing

const climatetrend = readdlm("../data/logco2eq.csv")
const climate_trenddict = Dict(floor.(Int, climatetrend[:, 1]) .=> climatetrend[:, 2])

const climate_trend = zeros(length(dat))
for i in 1:length(dat)
    climate_trend[i] = climate_trenddict[year[i]]
end

## setting up the covariate matrices

const nsplines = 8
const basismatrix = rcopy(R"pbs::pbs($jday %% 365.25,df=$nsplines, Boundary.knots = c(0,365.25))") #using R package pbs for periodic splines
const seasonalmatrix = [cos.(2*π*jday./365.25) sin.(2*π*jday./365.25)]

const ϕmatrix = [repeat([1.0], length(jday)) basismatrix climate_trend climate_trend.*seasonalmatrix]
const τmatrix = [repeat([1.0], length(jday)) basismatrix]

## example initial guess

const XtX = τmatrix' * τmatrix
const regcoef = XtX \ τmatrix' * dat

const ϕ₀init = [copy(regcoef); zeros(3)]
const ϕ₁init = [copy(regcoef); zeros(3)]
const κ₀init = 0.0
const κ₁init = 0.0
const νinit = 1.0
const τ₀init = zeros(nsplines+1)
const τ₁init = zeros(nsplines+1)

const initguess = [κ₀init; τ₀init; ϕ₀init; κ₁init; τ₁init; ϕ₁init; νinit]

## fit 
# note this takes 61 iterations and converges after about 5 mins on my laptop 
# this will return an answer even if the software hasn't converged, see status below

const mle = fit_bats_mle_covariates(dat, τmatrix, ϕmatrix, initguess, print_level=5)

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