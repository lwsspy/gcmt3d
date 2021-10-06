#!/bin/bash

# Short script that checks whether the files in submitted where actually finished
# and if they weren't will remove them, so that they can be resubmitted!

usage="
\n
------\n
Usage:\n
------\n
\n
remove_unfinished_inversions.sh <submitted dir> <done dir> \n
\n
--------\n
Example:\n
--------\n
\n
remove_unfinished_inversions.sh submitted done \n
\n
"

# Number of expected input arguments
Nexpect=2

if [ -z $1 ] || [ "${1}" == "-h" ] || [ "${1}" == "--help" ] || [ $# -ne $Nexpect ]
then
    echo -e $usage
    exit
fi

submdir=$1
donedir=$2

cmts=() 

for cmt in $(ls $submdir)
do 
    counter=0

    for cmtdone in $(ls $donedir)
    do 
	if [ "$cmt" == "$cmtdone" ]
	then 
	    counter=$(($counter + 1))
	    echo "$cmt done"
	fi
    done
    
    if [[ $counter == 0 ]]
    then 
	echo "$cmt not done"
	rm -f $submdir/$cmt
    
    fi

done
