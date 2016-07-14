library(Quandl)
library(forecast)
library(CausalImpact)
library(biganalytics)
Quandl.api_key("ti-yR6gd8be4x4yswvy4")
 
################## CONSTANTS ##################
priorSampleSize <- 32
expectedModelSize <- 3
expectedR2 <- 0.8
priorDataFrames <- 50
numberIterations <- 1000
variableSelectionMinimum <- 0.1
trainingPercent <- 0.8
normalize <- TRUE
frequency <- "monthly"
start <- "1967-06-01"
end <- "2016-06-01"
futureDataPoints <- 12
 
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

# Run Future Prediction
futurePrediction <- function(completeY, completeData, length){
	# Extrapolate each Indicator Variable
	ncol <- NCOL(completeData)
	forecast <- matrix(NA, nrow = futureDataPoints, ncol = ncol)
	for(i in 1 : ncol)
	{
		fit <- auto.arima(completeData[,i])
		forecast[,i] <- forecast(fit, h = futureDataPoints)$mean
	}

	# Extract and Name y Variable
	forecastY <- forecast[,1]
	dimnames(forecast) <- list(rownames(forecast, do.NULL = FALSE, prefix = "row"), colnames(forecast, do.NULL = FALSE, prefix = "col"))
	colnames(forecast)[colnames(forecast) == "col1"] <- "y"

	# Run Model
	result <- validatePrediction(completeY, completeData, forecastY, forecast, length, length)
	return(result)
}
 
################## MAIN ##################
responseVariable <- "FRED/RECPROUSM156N"
indicatorVariables <- c("FRED/ACOILBRENTEU","FRED/ACOILWTICO","FRED/AHETPI","FRED/AISRSA","FRED/ASEANTOT","FRED/AUINSA","FRED/AWHAETP","FRED/BAA10Y","FRED/BUSINV","FRED/CANTOT","FRED/CBI","FRED/CDSP","FRED/CES0500000003","FRED/CEU0500000002","FRED/CEU0500000003","FRED/CEU0500000008","FRED/CHNTOT","FRED/CIVPART","FRED/CNP16OV","FRED/COMPHAI","FRED/COMPNFB","FRED/COMPRNFB","FRED/COMREPUSQ159N","FRED/CPIAUCNS","FRED/CPIAUCSL","FRED/DCOILBRENTEU","FRED/DCOILWTICO","FRED/DDDM01USA156NWDB","FRED/DED1","FRED/DED3","FRED/DED6","FRED/DEXBZUS","FRED/DEXCAUS","FRED/DEXCHUS","FRED/DEXJPUS","FRED/DEXKOUS","FRED/DEXMXUS","FRED/DEXNOUS","FRED/DEXSDUS","FRED/DEXSFUS","FRED/DEXSIUS","FRED/DEXSZUS","FRED/DEXUSAL","FRED/DEXUSEU","FRED/DEXUSNZ","FRED/DEXUSUK","FRED/DGORDER","FRED/DGS10","FRED/DSPIC96","FRED/DSWP1","FRED/DSWP10","FRED/DSWP2","FRED/DSWP3","FRED/DSWP30","FRED/DSWP4","FRED/DSWP5","FRED/DSWP7","FRED/DTWEXB","FRED/DTWEXM","FRED/ECIWAG","FRED/ECOMNSA","FRED/ECOMSA","FRED/EECTOT","FRED/EMRATIO","FRED/ETOTALUSQ176N","FRED/EVACANTUSQ176N","FRED/FEDFUNDS","FRED/FRNTOT","FRED/FYFSGDA188S","FRED/GASREGCOVM","FRED/GASREGCOVW","FRED/GASREGM","FRED/GASREGW","FRED/GCT1501US","FRED/GCT1502US","FRED/GCT1503US","FRED/GERTOT","FRED/GOLDAMGBD228NLBM","FRED/GOLDPMGBD228NLBM","FRED/HCOMPBS","FRED/HDTGPDUSQ163N","FRED/HOABS","FRED/HOANBS","FRED/HOUST","FRED/HPIPONM226N","FRED/HPIPONM226S","FRED/IC4WSA","FRED/INDPRO","FRED/INTDSRUSM193N","FRED/IPBUSEQ","FRED/IPDBS","FRED/IPMAN","FRED/IPMAT","FRED/IPMINE","FRED/IR","FRED/IR10010","FRED/IREXPET","FRED/ISRATIO","FRED/JCXFE","FRED/JPNTOT","FRED/JTS1000HIL","FRED/JTS1000HIR","FRED/JTSHIL","FRED/JTSHIR","FRED/JTSJOL","FRED/JTSJOR","FRED/JTSLDL","FRED/JTSLDR","FRED/JTSQUL","FRED/JTSQUR","FRED/JTSTSL","FRED/JTSTSR","FRED/JTU1000HIL","FRED/JTU1000HIR","FRED/JTUHIL","FRED/JTUHIR","FRED/JTUJOL","FRED/JTUJOR","FRED/JTULDL","FRED/JTULDR","FRED/JTUQUL","FRED/JTUQUR","FRED/JTUTSL","FRED/JTUTSR","FRED/LNS12032194","FRED/LNS12032196","FRED/LNS14027660","FRED/LNS15000000","FRED/LNU05026642","FRED/M12MTVUSM227NFWA","FRED/M2V","FRED/MCOILBRENTEU","FRED/MCOILWTICO","FRED/MCUMFN","FRED/MEHOINUSA646N","FRED/MEHOINUSA672N","FRED/MFGOPH","FRED/MFGPROD","FRED/MNFCTRIRNSA","FRED/MNFCTRIRSA","FRED/MNFCTRMPCSMNSA","FRED/MNFCTRMPCSMSA","FRED/MNFCTRSMNSA","FRED/MNFCTRSMSA","FRED/MYAGM2USM052N","FRED/MYAGM2USM052S","FRED/NAPM","FRED/NAPMBI","FRED/NAPMCI","FRED/NAPMEI","FRED/NAPMEXI","FRED/NAPMII","FRED/NAPMIMP","FRED/NAPMNOI","FRED/NAPMPI","FRED/NAPMPRI","FRED/NAPMSDI","FRED/NILFWJN","FRED/NILFWJNN","FRED/NMFBAI","FRED/NMFBI","FRED/NMFCI","FRED/NMFEI","FRED/NMFEXI","FRED/NMFIMI","FRED/NMFINI","FRED/NMFINSI","FRED/NMFNOI","FRED/NMFPI","FRED/NMFSDI","FRED/NROU","FRED/NROUST","FRED/OPHMFG","FRED/OPHNFB","FRED/OPHPBS","FRED/OUTBS","FRED/OUTMS","FRED/OUTNFB","FRED/PAYEMS","FRED/PAYNSA","FRED/PCE","FRED/PCECTPICTM","FRED/PCEPI","FRED/PCEPILFE","FRED/PCETRIM12M159SFRBDAL","FRED/PCETRIM1M158SFRBDAL","FRED/PNRESCON","FRED/PNRESCONS","FRED/POP","FRED/POPTHM","FRED/PPIACO","FRED/PRRESCON","FRED/PRRESCONS","FRED/PRS30006013",
"FRED/PRS30006023","FRED/PRS84006013","FRED/PRS84006023","FRED/PRS84006163","FRED/PRS84006173","FRED/PRS85006023","FRED/PRS85006163","FRED/PRS85006173","FRED/RCPHBS","FRED/RETAILIMSA","FRED/RETAILIRSA","FRED/RETAILMPCSMNSA","FRED/RETAILMPCSMSA","FRED/RETAILSMNSA","FRED/RETAILSMSA","FRED/RHORUSQ156N","FRED/RIFLPCFANNM","FRED/RPI","FRED/RRSFS","FRED/RSAFS","FRED/RSAFSNA","FRED/RSAHORUSQ156S","FRED/RSEAS","FRED/RSFSXMV","FRED/RSNSR","FRED/RSXFS","FRED/T10Y2Y","FRED/T10Y3M","FRED/T10YFF","FRED/T10YIEM","FRED/T5YIEM","FRED/T5YIFR","FRED/TB3SMFFM","FRED/TCU","FRED/TDSP","FRED/TEDRATE","FRED/TLCOMCON","FRED/TLCOMCONS","FRED/TLNRESCON","FRED/TLNRESCONS","FRED/TLPBLCON","FRED/TLPBLCONS","FRED/TLPRVCON","FRED/TLPRVCONS","FRED/TLRESCON","FRED/TLRESCONS","FRED/TOTBUSIMNSA","FRED/TOTBUSIRNSA","FRED/TOTBUSMPCIMNSA","FRED/TOTBUSMPCIMSA","FRED/TOTBUSMPCSMNSA","FRED/TOTBUSMPCSMSA","FRED/TOTBUSSMNSA","FRED/TOTBUSSMSA","FRED/TOTDTEUSQ163N","FRED/TRFVOLUSM227NFWA","FRED/TTLCON","FRED/TTLCONS","FRED/U4RATE","FRED/U4RATENSA","FRED/U6RATE","FRED/U6RATENSA","FRED/UEMPMED","FRED/UKTOT","FRED/ULCBS","FRED/ULCMFG","FRED/ULCNFB","FRED/UNRATE","FRED/USAGDPDEFAISMEI","FRED/USAGDPDEFQISMEI","FRED/USAGFCFADSMEI","FRED/USAGFCFQDSMEI","FRED/USAGFCFQDSNAQ","FRED/USARECDM","FRED/USARGDPC","FRED/USASACRAISMEI","FRED/USASACRMISMEI","FRED/USASACRQISMEI","FRED/USPRIV","FRED/USRECD","FRED/USRECDM","FRED/USSLIND","FRED/USSTHPI","FRED/WCOILBRENTEU","FRED/WCOILWTICO","FRED/WHLSLRIRNSA","FRED/WHLSLRIRSA")
 
# Pull Data
rawData <- Quandl(c(toString(responseVariable), indicatorVariables),
	start_date = toString(start), end_date = toString(end), type = "zoo",
	collapse = toString(frequency), transform = "normalize")
 
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
future <- futurePrediction(completeY, completeData, length)
