#!/bin/bash



usage="
\n
------\n
Usage:\n
------\n
\n
create_invdir.sh <INVERSION-DIRECTORY>\n
\n
"

# Number of expected input arguments
Nexpect=1

if [ -z $1 ] || [ "${1}" == "-h" ] || [ "${1}" == "--help" ] || [ $# -ne $Nexpect ]
then
    echo -e $usage
    exit
fi

if [ -d $1 ]
then
    echo 'Directory already exists. Nothing is done.'
    exit
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Necessary files
INVDIR=$1
CMTSDIR=${INVDIR}/cmtsolutions
LOGSDIR=${INVDIR}/logs
SUBSDIR=${INVDIR}/submitted
DONEDIR=${INVDIR}/done
FAILDIR=${INVDIR}/fail
PARAMDIR=${INVDIR}/parameters

# Data to be copied from repo
CPDATADIR=${SCRIPT_DIR}/data
CATALOG_PARAMS=${CPDATADIR}/catalog_params.yml
INPUT=${CPDATADIR}/input.yml
INVERSIONSCRIPT=${CPDATADIR}/invertcmt.lsf
INVERSIONCHUNKSCRIPT=${CPDATADIR}/invert_chunky_job.lsf

# Create all the dirs
mkdir $INVDIR
mkdir $CMTSDIR
mkdir $LOGSDIR
mkdir $SUBSDIR
mkdir $DONEDIR
mkdir $FAILDIR
mkdir $PARAMDIR

# Copy data
cp $CATALOG_PARAMS $PARAMDIR/
cp $INPUT $PARAMDIR/
cp $INVERSIONSCRIPT $INVDIR/
cp $INVERSIONCHUNKSCRIPT $INVDIR/


important="
\n
IMPORTANT:\n
\n
There are still some things left to do:\n
\n
1.\t You have to modify the 'parameter/input.yml' for your own need.\n\n
2.\t Also you will possibly have to add 'parameter/process.yml'\n
\t for an example checkout 'lwsspy/gcmt3d/process.yml'\n
\t It contains all parameters needed to perform the waveform processing and \n
\t windowing.\n\n
3.\t For job submission using the 'invert_cmt.lsf' or 'invert_chunky_job.lsf',\n
\t you will have to modify multiple things in the script.\n
\t - submissiondir should point at -> INVERSION-DIRECTORY\n
\t - change node numbers depending on the number of parameters and in the case\n
\t   of the chunky job also the total number of nodes and chunk size.\n\n
4.\t You will have to get some CMTSOLUTIONS to be put into the 'cmtsolution'\n
\t directory\n\n
5.\t Then depending on whether you are running from compute nodes or not, you may\n
\t want to download data first and then do the inversion.\n
\t See 'download_data' flag in 'parameters/input.yml' and the command-line\n
\t argument for 'gcmt3d-invert-cmt' '-d' for download only.\n

"

echo -e $important