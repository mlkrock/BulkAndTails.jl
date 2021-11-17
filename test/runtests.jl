
using BulkAndTails, Test

const testfiles = ("pdf_cdf.jl", "fitting.jl")

@testset "BulkAndTails.jl" begin
  for testfile in testfiles
    include(testfile)
  end
end

