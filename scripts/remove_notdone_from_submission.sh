#!/bin/bash

usage="
\n
------\n
Usage:\n
------\n
\n
remove_unfinished_inversions.sh <cmtsolution dir> <submitted dir> <done dir> \n
\n
--------\n
Example:\n
--------\n
\n
remove_unfinished_inversions.sh cmtsolutions submitted done \n
\n
"

Nexpect=3

if [ -z $1 ] || [ "${1}" == "-h" ] || [ "${1}" == "--help" ] || [ $# -ne $Nexpect ]
then
    echo -e $usage
    exit
fi

CMTS=$1
SUBMIT=$2
DONE=$3


for cmt in $(ls $CMTS)
do
    if [ -e "$SUBMIT/$cmt" ]
    then
	
	if [ ! -e "$DONE/$cmt" ]
	then
	    rm "$SUBMIT/$cmt"
	fi
    fi
done
