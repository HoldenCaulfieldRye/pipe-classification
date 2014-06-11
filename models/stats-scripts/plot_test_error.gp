#set logscale xy
set term png
set output "test_error_time_series.png"
set title "time series of the test error"
plot 	"time_series.txt" using 1:2 title 'logprob'\
with linespoints
