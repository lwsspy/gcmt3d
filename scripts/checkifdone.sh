#!/bin/bash

# Input argument is the data base in question
# Output is a list of events in that database that do not have
# the data, but do have a directory in the database
usage="
\n
------\n
Usage:\n
------\n
\n
checkifdone.sh <DONEDIR> <EVENTDR> \n
\n
--------\n
Example:\n
--------\n
\n
checkifdone.sh done cmtsolutions \n
\n
"

# Number of expected input arguments
Nexpect=2

if [ -z $1 ] || [ "${1}" == "-h" ] || [ "${1}" == "--help" ] || [ $# -ne $Nexpect ]
then
    echo -e $usage
    exit
fi

DONEDIR=$1
EVENTDIR=$2

# Get event list
EVENTS=$(ls $EVENTDIR)


for cmt in $EVENTS
do 
    # Print if the donedir does not contain the cmt solution
    if [ ! -e "$DONEDIR/${cmt}" ]
    then 
	echo $cmt
    fi
done