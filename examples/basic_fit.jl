
using Distributions, BulkAndTails

# Some fake data, in this case a simple Cauchy distribution.
const data = rand(TDist(1.0), 5_000) 

# you can set print_level=0 to have no Ipopt output. But it is nice to have the
# output, because if something goes wrong you can google "interpreting Ipopt
# output" and then look at the abundant printed information and probably figure
# out what's going wrong with your model.
#
# Note that Ipopt is pretty sensitive, and so there is a chance that this will
# kind of get stuck or something strange seeming and hit the maximum number of
# iterations or something. This isn't necessarily a problem with the model or
# code. It's a hard optimization problem. We suggest trying again with different
# initial values (an optional second arg here).
const (mle, observed_information_matrix) = fit_mle(BulkAndTailsDist, data, print_level=5)

# A vague check for fit quality, which seems pretty good.
for _x in (0.1, 1.0, 10.0, 100.0, 1000.0)
  println("x = $_x:")
  println("Cauchy PDF:      $(pdf(TDist(1.0), _x))")
  println("Fitted BATs PDF: $(pdf(mle,        _x))")
end

