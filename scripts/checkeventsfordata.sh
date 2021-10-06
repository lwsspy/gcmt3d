#!/bin/bash

# Input argument is the data base in question
# Output is a list of events in that database that do not have
# the data, but do have a directory in the database
# Number of expected input arguments

usage="
\n
------\n
Usage:\n
------\n
\n
checkeventsfordata.sh <DATABASE> [<EVENTDIR>] \n
\n
EVENTDIR is optional -- if not provided the function will check\n
all events in the database.\n
\n
"

Nexpect=2

if [ -z $1 ] || [ "${1}" == "-h" ] || [ "${1}" == "--help" ] || [ $# -gt $Nexpect ]
then
    echo -e $usage
    exit
fi

DATABASE=$1
EVENTDIR=$2


# If an event directory is provided, then check only the 
# events in that directory
if [ -z $EVENTDIR ]
then
    EVENTS=$(ls $DATABASE)
else
    EVENTS=$(ls $EVENTDIR)
fi

for cmt in $EVENTS
do 
    # Print event if the directory doesn't even contain the data dir
    if [ ! -e "$DATABASE/$cmt/data" ]
    then 
	echo $cmt
    else 
	# Print the event if the data dir doesnt contain the waveform dir
	if [ -z "$(ls -A $DATABASE/$cmt/data)" ]
	then 
	    echo "$cmt"
	else 
	    # Print the event if the waveform dir is not empty
	    if [ -z "$(ls -A $DATABASE/$cmt/data/waveforms)" ]
	    then 
		echo $cmt
	    fi
	fi
    fi
done