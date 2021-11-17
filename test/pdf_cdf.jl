
using Distributions

const T_1df = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
const T_2df = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0]
const xv    = (-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0)

@testset "PDF" begin
  for x in xv
    @test isapprox(pdf(BulkAndTailsDist(T_1df), x), pdf(TDist(1.0), x))
    @test isapprox(pdf(BulkAndTailsDist(T_2df), x), pdf(TDist(2.0), x))
  end
end

@testset "LOGPDF" begin
  for x in xv
    @test isapprox(logpdf(BulkAndTailsDist(T_1df), x), logpdf(TDist(1.0), x))
    @test isapprox(logpdf(BulkAndTailsDist(T_2df), x), logpdf(TDist(2.0), x))
  end
end

@testset "CDF" begin
  for x in xv
    @test isapprox(cdf(BulkAndTailsDist(T_1df), x), cdf(TDist(1.0), x))
    @test isapprox(cdf(BulkAndTailsDist(T_2df), x), cdf(TDist(2.0), x))
  end
end

