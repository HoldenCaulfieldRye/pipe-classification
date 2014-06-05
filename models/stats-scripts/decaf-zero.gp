#set logscale xy
set term png
set output "logprob.png"
set title "cross-entropy between predicted and target classification\
pmfs, estimated over 86 test batches, calculated every 20(?)\
iterations of SGD"
plot 	"time_series.txt" using 1:2 title 'decaf-net zero-initialised'\
with linespoints
# set terminal postscript portrait enhanced mono dashed lw 1 'Helvetica'\
#  14
# set output 'plot.ps'
