# Install Microsoft Open R from https://mran.microsoft.com/documents/rro/installation/
brew update # If you don't have homebrew, get it from here (http://brew.sh/)
brew install hadoop # Install Hadoop
brew install apache-spark # Install Spark
brew install postgres # Install Postgres
R -q -e "install.packages("SparkR","acepack","assertthat","base","BH","biganalytics","biglm","bigmemory","bigmemory.sri","bitops","Boom","BoomSpikeSlab","boot","brew","bsts","caTools","CausalImpact","checkpoint","chron","class","cluster","codetools","colorspace","compiler","curl","data.table","datasets","DBI","deployrRserve","devtools","dichromat","digest","dplyr","evaluate","foreach","forecast","foreign","Formula","fracdiff","functional","ggplot2","git2r","graphics","grDevices","grid","gridExtra","gtable","Hmisc","httr","iterators","jsonlite","KernSmooth","labeling","lattice","latticeExtra","lazyeval","magrittr","MASS","Matrix","memoise","methods","mgcv","mime","munsell","nlme","nnet","openssl","parallel","plyr","plyrmr","quadprog","Quandl","R.methodsS3","R6","RColorBrewer","Rcpp","RcppArmadillo","reshape2","rhdfs","rJava","rjson","RJSONIO","rmr2","roxygen2","rpart","rstudioapi","rversions","scales","spatial","splines","stats","stats4","stringi","stringr","survival","tcltk","timeDate","tools","tseries","utils","whisker","withr","xml2","xts","zoo")"