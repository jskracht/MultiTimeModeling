library(forecast)
library(CausalImpact)
library(biganalytics)
library(zoo)
 
################## CONSTANTS ##################
priorSampleSize <- 32
expectedModelSize <- 3
expectedR2 <- 0.8
priorDataFrames <- 50
numberIterations <- 1000
variableSelectionMinimum <- 0.1
trainingPercent <- 0.8
normalize <- TRUE
 
################## FUNCTIONS ##################
# Fill in Null Data and Normalize
standardizeData <- function(data){
	# Linearly Interpolate Missing Values
	data <- na.spline(data)

	# Normalize Variables as Index 0:1
	if(normalize)
	{
		data <- as.zoo(apply(data, MARGIN=2, FUN=function(x)(x-min(x))/diff(range(x))))
	}

	# Remove NAN Columns
	data <- data[, colSums(!is.nan(data)) != 0]

	# Change Name of Response Column
	names(data)[1] = "y"

	return(data)
}
 
# Build Matrix of Most Significant Variables
getSignificantVariables <- function(model){
	variableRelevance = as.big.matrix(model$coefficients)
	bestVariables = colmax(variableRelevance)
	variables = subset(bestVariables, bestVariables > variableSelectionMinimum)
	return(variables)
}
 
# Run Causual Impact
causalImpact <- function(completeY, completeData, postY, predictData, startPrediction, length){
	# Null Out Post Data
	impactTrainingData <- completeData
	impactTrainingData[,1][startPrediction : length] <- NA
	impactPreY <- completeY
	impactPreY[startPrediction : length] <- NA

	# Set Prior Sampling Configuration
	sdy <- sd(impactPreY, na.rm = TRUE)
	sd.prior <- SdPrior(sigma.guess = 0.01 * sdy,
	upper.limit = sdy,
	sample.size = priorSampleSize)

	# Setup 1 Season
	ss <- list()
	ss <- AddLocalLevel(ss, impactPreY, sigma.prior = sd.prior)

	# Define x and y
	formula <- paste0("y", " ~ .")

	# Build Model
	model <- bsts(formula, data = impactTrainingData, state.specification = ss,
	expected.model.size = expectedModelSize,
	expected.r2 = expectedR2,
	prior.df = priorDataFrames,
	save.prediction.errors = TRUE,
	niter = numberIterations, seed = 1, ping = 0)

	# Bayesian Structural Time Series Model Prediction
	result = predict(model, predictData, burn = SuggestBurn(.1, model), oldData)

	# Renormalize Values Outside Range 0:1
	if(normalize)
	{
		result$median[result$median < 0] <- 0
		result$median[result$median > 1] <- 1
	}

	# Causal Impact
	impact <- CausalImpact(bsts.model = model, post.period.response = postY, alpha = .9)

	# Kernel Smoothing
	# kernel <- kernel("daniell", 3)
	# smooth <- kernapply(as.matrix(impact$series$point.pred), kernel)
	# plot.ts(smooth)

	impact$series$point.pred[impact$series$point.pred < 0] <- 0
	impact$series$point.pred[impact$series$point.pred > 1] <- 1
	impact$series$point.pred.lower[impact$series$point.pred.lower < 0] <- 0
	impact$series$point.pred.lower[impact$series$point.pred.lower > 1] <- 1
	impact$series$point.pred.upper[impact$series$point.pred.upper < 0] <- 0
	impact$series$point.pred.upper[impact$series$point.pred.upper > 1] <- 1

	# Plot Model Fit
	plot(impact, "original")
	return(impact)
}
 
# Run Validation Prediction
validatePrediction <- function(completeY, completeData, postY, predictData, endTraining, length){
	# Remove Post Data
	predictTrainingData <- completeData[c(1 : endTraining),]
	predictPreY <- predictTrainingData[,1]

	# Set Prior Sampling Configuration
	sdy <- sd(predictPreY, na.rm = TRUE)
	sd.prior <- SdPrior(sigma.guess = 0.01 * sdy,
	upper.limit = sdy,
	sample.size = priorSampleSize)

	# Setup 1 Season
	ss <- list()
	ss <- AddLocalLevel(ss, predictPreY, sigma.prior = sd.prior)

	# Define x and y
	formula <- paste0("y", " ~ .")

	# Build Model
	model <- bsts(formula, data = predictTrainingData, state.specification = ss,
	expected.model.size = expectedModelSize,
	expected.r2 = expectedR2,
	prior.df = priorDataFrames,
	save.prediction.errors = TRUE,
	niter = numberIterations, seed = 1, ping = 0)

	# Bayesian Structural Time Series Model Prediction
	result = predict(model, predictData, burn = SuggestBurn(.1, model), oldData)

	# Renormalize Values Outside Range 0:1
	if(normalize)
	{
		result$median[result$median < 0] <- 0
		result$median[result$median > 1] <- 1
	}

	# Plot Model Fit
	if(normalize)
	{
		if(endTraining != length)
		{
			par(mfrow=c(2,1))
			plot(completeY, ylab = NULL, xlab = "Time", ylim = c(0,1))
		}            
		plot(result, interval.width = 0, ylim = c(0,1))
	} else {
		if(endTraining != length)
		{
			par(mfrow=c(2,1))
			plot(completeY, ylab = NULL, xlab = "Time")
		}   
		plot(result, interval.width = 0)
	}
	return(result)
}
 
################## MAIN ##################
setwd("C:/Users/krachtj/Documents/MultiTimeModeling/test/31285C005M20131102_023207")
responseFiles <- list.files(recursive = TRUE, pattern="FaultCodeData.csv")
responseVariable <- read.zoo(responseFiles, format = "%Y-%m-%d %H:%M:%S", header = TRUE, sep = ",", index.column = 3, FUN=as.POSIXct, aggregate = head)
responseVariable <- responseVariable[,1]
responseVariable[is.na(responseVariable)] <- 0
responseVariable[!is.na(responseVariable)] <- 1
indicatorFiles <- list.files(recursive = TRUE, pattern="TestPoints_1Hz.v2.csv")
indicatorVariables <- read.zoo(indicatorFiles, format = "%m/%d/%Y %H:%M:%S", header = TRUE, sep = ",", index.column = 2, FUN=as.POSIXct, aggregate = head, colClasses=c("numeric", "character", rep("numeric",663)))
rawData <- cbind(responseVariable, indicatorVariables)
temp <- rawData[,1]
temp[is.na(rawData)] <- 0
rawData[,1] <- temp

completeData <- standardizeData(rawData)
 
# Get Split Point
length <- NROW(completeData)
endTraining <- round(length * trainingPercent)
 
# Get Y
completeY <- completeData[, 1]
 
# Split Off Data Used for Prediction
startPrediction <- endTraining + 1
predictData <- completeData[c(startPrediction : length),]
postY <- as.vector(completeY[startPrediction : length])
 
impact <- causalImpact(completeY, completeData, postY, predictData, startPrediction, length)
validation <- validatePrediction(completeY, completeData, postY, predictData, endTraining, length)
