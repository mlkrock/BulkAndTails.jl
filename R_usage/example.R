# The library that lets R talk to Julia easily and efficiently.
library(JuliaCall)

# You need to run these every time you start R.
julia_setup()
julia_library("BulkAndTails")

# Simulate data.
data = rt(10000, df=3.5)

# Try to fit using the default init, which is
# (κ₀, τ₀, ϕ₀, κ₁, τ₁, ϕ₁, ν) = (1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0).
# In general, may need to try
# with other inits. See the Julia example for a brief
# description of the function call, and the comments of the source code for even
# more detail. If you want no optimizer output, set print_level=0L. The L is
# important, because that's the only way that R will treat it as an actual int.
mle_result = julia_call("fitbats", data, print_level=5L)

mle = mle_result[[1]] # MLE
hes = mle_result[[2]] # Hessian of negative log likelihood at MLE
nll = mle_result[[3]] # Value of negative log likelihood at MLE

# If you want to sanity check that this code did gave back a sensible MLE, you
# could always compare with the t(3.5) density in a few places:
pointwise_error <- function(lower, upper, length){
  xv = seq(lower, upper, length=length)
  plot(xv, abs(julia_call("batspdf", xv, mle) - dt(xv, df=3.5)), log="y",
       ylab="pointwise density error using MLE parameters")
}
pointwise_error(0.0, 100, 1000)

