#!/bin/bash
echo "nohup ./scp_graphic02_here.sh \"upload\" >> scp_output.txt 2>&1 &"
echo "is the way to execute this"
mkdir /data/ad6813
mkdir /data/ad6813/pipe-data
mkdir /data/ad6813/pipe-data/Redbox
mkdir /data/ad6813/pipe-data/Redbox/batches
mkdir /data/ad6813/pipe-data/Redbox/batches/clamp_detection
mkdir /data/ad6813/my-nets
mkdir /data/ad6813/my-nets/saves/
cd /data/ad6813/pipe-data/Redbox/batches/clamp_detection
echo "copying batches from graphic02 to local data storage..."
scp graphic02.doc.ic.ac.uk:/data2/ad6813/pipe-data/Redbox/batches/clamp_detection/* .
echo "copying ConvNet 23.05.49 from graphic02 to local data storage..."
cd /data/ad6813/my-nets/saves/
scp graphic02.doc.ic.ac.uk:/data2/ad6813/my-nets/saves/ConvNet__2014-06-17_23.05.49 .
echo "WARNING: modify options.cfg to look up in /data, not /data2"
