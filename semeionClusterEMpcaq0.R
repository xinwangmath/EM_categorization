
# main program for q = 0: the goal of this file is to compute the categorization
#  with EM algorithm and to save the data to a .RData file
#  The E step and M step functions are defined in two separated files
# author: Xin Wang, xinwangmath@gmail.com

# Note: we did a PCA dimension reduction at the beginning, since d = 256 is too high
# a dimension for 1593 data points. 

rm(list = ls())

# import mvtnorm library
library(mvtnorm)

# import the data
myData = read.csv('semeion.csv', header = F)

# save date into a matrix, generate correct labels
myXoriginal = data.matrix(myData[, 1:256])
myLabel = apply(myData[, 257:266], 1, function(xx){
	return(which(xx == "1")) - 1
})

# perform a PCA first to reduce the dimension to reducedN = 25
reducedN = 25
xBar = colMeans(myXoriginal)
xTilde = t(apply(myXoriginal, 1, function(xx){
	return(xx - xBar)
	}))
mySVD = svd(xTilde)
myPCs = xTilde %*% mySVD$v
myX = myPCs[, 1:reducedN]

# save the dimensions explicitly in D

N = dim(myX)[1]
D = dim(myX)[2]
K = 10

# define a matrix to save the \gamma_{i k}'s
myGamma = matrix(0, nrow = N, ncol = K)

# part-1
# use kmeans to cluster the data and provide initial \gamma_{i k}'s
clKmeans = kmeans(myX, K, iter.max =  30, nstart = 25)

for(i in 1:N){
	myGamma[i, clKmeans$cluster[i]] = 1
}

# import the M step and E step functions
source('mstepv4.R')
source('estepv4.R')

# solve for the q = 0 case
q = 0
# find \mu_k, \pi_k and \Sigma_k's from the initial \gamma{i k}'s
Mu = mstepMu(myGamma, myX, N, D, K)
Pi = mstepPi(myGamma, N)
SigmaList = mstepSigma(myX, myGamma, Mu, N, D, K, q)

# use 25 EM iterations
niter = 25

# define a vector to save the log-likelihood after each iteration
loglrecord = rep(0, (niter+1))


for(i in 1:niter){
	# E step
	newGammaLogPList = findGamma(myX, Pi, Mu, SigmaList, K)
	myGamma = newGammaLogPList[[1]]

	# save log-likelihood
	loglrecord[i] = newGammaLogPList[[2]]

	# M step
	Mu = mstepMu(myGamma, myX, N, D, K)
	Pi = mstepPi(myGamma, N)
	SigmaList = mstepSigma(myX, myGamma, Mu, N, D, K, q)	
}

# the final E step 
newGammaLogPList = findGamma(myX, Pi, Mu, SigmaList, K)
myGamma = newGammaLogPList[[1]]
# save the log-likelihood
loglrecord[(niter+1)] = newGammaLogPList[[2]]

# recover the high dimensional cluster mean vectors 
highDimMu = mySVD$v[, 1:reducedN] %*% Mu

# save the data
save(list = ls(all = TRUE), file = "semeionPCAv3q0.RData")
