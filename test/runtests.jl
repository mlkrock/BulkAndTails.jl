
using BATsDistribution, Test

const testfiles = ("pdf_cdf.jl", "fitting.jl")

@testset "BATsDistribution.jl" begin
  for testfile in testfiles
    include(testfile)
  end
end

