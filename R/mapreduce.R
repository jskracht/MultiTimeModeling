Sys.setenv("HADOOP_CMD"="/bin/hadoop")
Sys.setenv("HADOOP_HOME"="/etc/hadoop")
Sys.setenv("HADOOP_STREAMING"="/usr/share/java/hadoop/hadoop-streaming.jar")
Sys.setenv("JAVA_HOME"="/etc/alternatives/java_sdk")
Sys.setenv("RSCRIPT"="/usr/bin/Rscript")

library(rhdfs)
library(rmr2)

wordcount = 
  function(
    input, 
    output = NULL, 
    pattern = " "){
    wc.map = 
      function(., lines) {
        keyval(
          unlist(
            strsplit(
              x = lines,
              split = pattern)),
          1)}
    wc.reduce =
      function(word, counts ) {
        keyval(word, sum(counts))}

    mapreduce(
      input = input,
      output = output,
      map = wc.map,
      reduce = wc.reduce,
      combine = TRUE)}

text = capture.output(license())
out = list()
for(be in c("local", "hadoop")) {
  rmr.options(backend = be)
  out[[be]] = from.dfs(wordcount(to.dfs(keyval(NULL, text)), pattern = " +"))}
stopifnot(rmr2:::kv.cmp(out$hadoop, out$local))
