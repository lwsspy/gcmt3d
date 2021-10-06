#!/bin/bash



usage="
\n
------\n
Usage:\n
------\n
\n
create_downloaddir.sh <INVERSION-DIRECTORY>\n
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
DDIR=$1
CMTSDIR=${DDIR}/cmtsolutions
LOGSDIR=${DDIR}/logs
PARAMDIR=${DDIR}/parameters
DOWNLOADEDDIR=${DDIR}/downloaded
DOWNLOADFAILDIR=${DDIR}/downloadfail


# Data to be copied from repo
CPDATADIR=${SCRIPT_DIR}/data
CATALOG_PARAMS=${CPDATADIR}/catalog_params.yml
INPUT=${CPDATADIR}/input.yml
SCRIPT=${CPDATADIR}/download_data.sh

# Create all the dirs
mkdir $DDIR
mkdir $CMTSDIR
mkdir $LOGSDIR
mkdir $DOWNLOADEDDIR
mkdir $DOWNLOADFAILDIR
mkdir $PARAMDIR

# Copy data
cp $CATALOG_PARAMS $PARAMDIR/
cp $INPUT $PARAMDIR/
cp $SCRIPT $DDIR/
