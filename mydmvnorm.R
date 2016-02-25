# function to compute multivariate normal density, utilize the fact that we have both Sigma and SigmaInverse matrix
#  input: xx:  a vector, i.e. a point in R^d
#         mu: mean \mu of the multivariate normal distribution
#         SigmaList: a list object, SigmaList[[1]] is the covariance matrix \Sigma, and SigmaList[[2]] is the inverse of covariance matrix
#  output: a real number that is the value of the density function at xx

mydmvnorm = function(xx, mu, SigmaList){

	Sigma = SigmaList[[1]]
	SigmaInverse = SigmaList[[2]]

	D = dim(Sigma)[1]

	xxTilde = xx - mu

	prob = exp((-0.5) * t(xxTilde) %*% SigmaInverse %*% xxTilde )
	prob = prob/((2 * pi)^(D/2) * sqrt(abs(det(Sigma))))

	return(prob)  
}