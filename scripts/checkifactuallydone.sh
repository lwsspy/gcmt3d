#!/bin/bash

usage="
\n
------\n
Usage:\n
------\n
\n
checkifactuallydone.sh <DATABASE> <CMTSOLUTIONS> <LABEL>\n
\n
"

# Number of expected input arguments
Nexpect=3

if [ -z $1 ] || [ "${1}" == "-h" ] || [ "${1}" == "--help" ] || [ $# -ne $Nexpect ]
then
    echo -e $usage
    exit
fi

# From here (directory)
DATABASE=$1 

# These IDs (directory)
CMTSOLUTIONS=$2

# With this Label (string)
LABEL=$3


for cmt in $(ls $CMTSOLUTIONS)
do
    icmtfile="${DATABASE}/${cmt}/${cmt}_${LABEL}"

    if [ ! -e  "$icmtfile" ]
    then 
	echo "${cmt}"
    fi

done