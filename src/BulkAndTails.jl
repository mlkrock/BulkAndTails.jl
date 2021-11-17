module BATsDistribution

  using Distributions, ForwardDiff, Ipopt, Roots, Random
  export BATs, fitbats

  include("BATs_struct.jl")
  include("BATs_functions.jl")
  include("fitting.jl")
  include("ipopt_interface.jl")
  
end # module
