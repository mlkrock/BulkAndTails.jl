module BulkAndTails

  using Distributions, ForwardDiff, Ipopt, Roots, Random
  export BulkAndTailsDist, fitbats

  include("BulkAndTailsDist_struct.jl")
  include("BulkAndTailsDist_functions.jl")
  include("fitting.jl")
  include("ipopt_interface.jl")
  
end # module
