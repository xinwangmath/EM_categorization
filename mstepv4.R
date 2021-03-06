
# M step functions collection
# author: Xin Wang, xinwangmath@gmail.com

# M-step: use three functions to compute new mu_k, pi_k and Sigma_k

# mstepMu(Gamma, Y, N, D, K)
# input: Gamma, the gamma_{i k} coefficients; 
#        Y: the data matrix,
#        N: total number of data pts, D feature dimension, K = number of clusters
# output: Mu: a matrix, with the kth column being mu_k 

mstepMu = function(Gamma, Y, N, D, K){
	Nk = colSums(Gamma, na.rm = F)
	Mu = matrix(0, nrow = D, ncol = K)
	for(k in 1:K) {
		Mu[, k] = t( 1/Nk[k] * t(Gamma[, k]) %*% Y)
	}

	return(Mu)

}

# mstepPi(Gamma, n)
# input: Gamma, the gamma_{i k} coefficients; 
# output: myPi, a column vector of K dimension, with myPi[k] = pi_k

mstepPi = function(Gamma, n) {
	Nk = colSums(Gamma, na.rm = F)
	Nk = as.matrix(Nk)
	myPi = (1/n) * Nk; 
	return(myPi)
}


# helper function for the mstepSigmaKth and mstepSigma functions, generate the cov matrix used in these programs
# formCovKth(X, Gamma, Muk, Nkkth, k)
# input: X = myX, data matrix
#        Gamma, the gamma_{i k} coefficients; 
#        Muk = mu_k, computed by mstepMu function
#        Nkkth = N_k
#        k the index for cluster
# output: the covariance matrix used in the Sigma_k computation

formCovKth = function(X, Gamma, Muk, Nkkth, k){
	N = dim(X)[1]
	D = dim(X)[2]
	Muk = t(Muk)
	covKth = matrix(0, nrow = D, ncol = D)
	XTilde = t(apply(X, 1, function(xx){
		return( xx - Muk )
		}))
	XTildeGamma = diag(Gamma[, k]) %*% XTilde
	covKth = (1/Nkkth) * t(XTilde) %*% XTildeGamma
	return(covKth)
}

# mstepSigmaKth = function(X, Gamma, Muk, Nkkth, N, D, K, k, q)
# output: returns the k-th covariance matrix \Sigma_k and its inverse

mstepSigmaKth = function(X, Gamma, Muk, Nkkth, N, D, K, k, q){

	covKth = formCovKth(X, Gamma, Muk, Nkkth, k)
	myEig = eigen(covKth)

	
	Vq = myEig$vectors[, 1:q]
	residualSigmaSq = sum(myEig$values[(q+1):D])/(D-q)

	if(q > 0){
		Wq = Vq %*% diag(sqrt(myEig$values[1:q] - rep(residualSigmaSq, q)))
		SigmaKth = Wq %*% t(Wq) +  diag(residualSigmaSq, nrow = D, ncol = D)

	}
	

	if(q == 0){
		SigmaKth = diag(residualSigmaSq, nrow = D, ncol = D)
	}

	# now compute Sigma^{-1}
	# the formula is: (C = Sigma, W = Wq)
	# C^{-1} = (1/sqrt(residualSigmaSq)) * Identity - (1/reidualSigmaSq) * W M^{-1} W^T
	# where M = W^T W + residualSigmaSq * Identity ( here the identity is the q \times q identity matrix)
	# note this will reduce the computation of matrix inverse from D^3 to q^3

	if(q > 0){
		M = t(Wq) %*% Wq + diag(residualSigmaSq, nrow = q, ncol = q)
		SigmaKthInverse = diag( (1/sqrt(residualSigmaSq)), nrow = D, ncol= D) - (1/residualSigmaSq) * Wq %*% solve(M) %*% t(Wq)
	}
	if(q == 0){
		SigmaKthInverse = diag((1/residualSigmaSq), nrow = D, ncol = D)
	}

	# put \Sigma_k and its inverse in a list called SigmaKthList
	SigmaKthList = list(SigmaKth, SigmaKthInverse) 

	return(SigmaKthList)

} 

# mstepSigma = function(X, Gamma, Mu, N, D, K, q)
# output: returns all the covariance matrix \Sigma_k and their inverse, put in a list object SigmaList as described below:
#  SigmaList: a list of two matrices: SigmaList[[1]] is a matrix of dimension D \times (K*D), the first D columns 
#                     form Sigma_1, (k-1) * D + 1 to k*D columns form Sigma_k; SigmaList[[2]] is a matrix of dimension 
#                     D \times (K*D), formed by columns of inverse of Sigma_k s

mstepSigma = function(X, Gamma, Mu, N, D, K, q){

	Nk = colSums(Gamma, na.rm = F)
	Nk = as.matrix(Nk)

	SigmaList = mstepSigmaKth(X, Gamma, Mu[, 1], Nk[1], N, D, K, 1, q)

	for(i in 2:K){
		SigmaListNew = mstepSigmaKth(X, Gamma, Mu[, i], Nk[i], N, D, K, i, q)
		SigmaList[[1]] = cbind(SigmaList[[1]], SigmaListNew[[1]])
		SigmaList[[2]] = cbind(SigmaList[[2]], SigmaListNew[[2]])
	}

	return(SigmaList); 
}
