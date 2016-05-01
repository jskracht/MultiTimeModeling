library(timeSeries)

read = function(x){
  series <- readSeries(x, header=TRUE, sep=",", format="%Y-%m-%d", col.names=c("DATE", x)) 
  print(x)
  return(series)
}

multmerge = function(mypath){
  filenames = list.files(path=mypath, recursive = TRUE, pattern="*.csv")
  datalist = lapply(filenames, read)
  Reduce(function(x,y) {merge(x,y,by.x=1,by.y=1)}, datalist)
}
  
setwd("/Users/Jesh/Documents/Project/FRED/R/E/C")
mymergeddata = multmerge(getwd())