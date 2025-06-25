#!/bin/bash

if [ $# -lt 2 ]
then
        echo "Usage: $0 <query file> <res file>"
        exit
fi

mvn exec:java -Dexec.mainClass="experiments.QPPOnPreRetrievedResults" -Dexec.args="$1 $2"
