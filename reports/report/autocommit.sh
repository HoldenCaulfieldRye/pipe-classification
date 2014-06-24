#!/bin/bash

if ps -ef | grep -v grep | grep "texmaker report.tex" ; then
    cd /home/alex/Git/pipe-classification
    git commit -a -m "auto-commit progress on report"
    git push origin master
    exit 0
else
    exit 0
fi
