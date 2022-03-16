module BulkAndTails

  using Distributions, ForwardDiff, Ipopt, Roots, Random
  export BulkAndTailsDist, fitbats, fit_bats_mle_covariates, fitbats_covariates, batspdf, batscdf, batsquantile, batslogpdf, batslogcdf, batsrand

  include("BulkAndTailsDist_struct.jl")
  include("BulkAndTailsDist_functions.jl")
  include("fitting.jl")
  include("ipopt_interface.jl")
  
end # module
