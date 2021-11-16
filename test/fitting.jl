
# See the example file for a more commented version of this.

using StableRNGs
const data = rand(StableRNG(123), TDist(1.0), 10_000)
const (mle, observed_information_matrix) = fit_mle(BATs, data, print_level=0)

@testset "Fitting" begin
  for _x in (0.1, 1.0, 10.0, 100.0, 1000.0)
    @test isapprox(pdf(TDist(1.0), _x), pdf(mle, _x), atol=1e-2)
  end
end

