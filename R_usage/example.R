
# At present, you need to run this file in the BATsDistribution.jl/R_usage because of the
# hard-coded path in it. Once BATsDistribution.jl is registered, though, that will change.

# The library that lets R talk to Julia easily and efficiently.
library(JuliaCall)

# You need to run these every time you start R.
julia_setup()
julia_library("Pkg")
julia_command("Pkg.activate(\"../\")") # stopgap until package registration.
julia_library("BATsDistribution")

# Simulate data.
data = rt(10000, df=3.5)

# Using a random init, try and fit it. See the Julia example for a brief
# description of the function call, and the comments of the source code for even
# more detail. If you want no optimizer output, set print_level=0L. The L is
# important, because that's the only way that R will treat it as an actual int.
mle_result = julia_call("BATsDistribution.fitbats", data, print_level=5L)

mle = mle_result[[1]] # MLE
hes = mle_result[[2]] # Hessian of negative log likelihood at MLE

# If you want to sanity check that this code did gave back a sensible MLE, you
# could always compare with the t(3.5) density in a few places:
pointwise_error <- function(lower, upper, length){
  xv = seq(lower, upper, length=length)
  plot(xv, abs(julia_call("BATsDistribution.batspdf", xv, mle) - dt(xv, df=3.5)), log="y",
       ylab="pointwise density error using MLE parameters")
}
pointwise_error(0.0, 100, 1000)

