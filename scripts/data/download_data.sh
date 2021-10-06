#!/bin/bash
#SBATCH -A GEO111 
#SBATCH --job-name=data-download # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --mem-per-cpu=50G        # memory per cpu-core (4G is default)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=lsawade@princeton.edu

limit_jobs()
{
   while true; do
      if [ "$(jobs -p | wc -l)" -lt "$1" ]; then break; fi
      sleep 1
   done
}

# just in case
ulimit -s unlimited

# load things
module purge
module load python/3.7.0-anaconda3-2018.12 
. /sw/dtn/python/3.7/anaconda3/2018.12/etc/profile.d/conda.sh
conda activate lwsspy

# Run inversion exectubale
submissiondir="/ccs/home/lsawade/DATA_DATABASE"

cmtdir="${submissiondir}/cmtsolutions"
submitteddir="${submissiondir}/submitted"
downloadeddir="${submissiondir}/downloaded"
downloadfaildir="${submissiondir}/downloadfail"
donedir="${submissiondir}/done"
faildir="${submissiondir}/fail"
inputfile="${submissiondir}/parameters/input.yml"
logs="${submissiondir}/logs"
cmtsolutions=$(ls $cmtdir)

# to be filled arrays
pids=()
cmts=()

# Maximum number of downloads
MAXDL=100000
JOBLIMIT=5

# Add a counter
counter=0

for cmt in $cmtsolutions
do
    # Get CMT solution
    cmtsolution=$cmtdir/${cmt}

    # Run inversion
    echo Downloading data for $cmtsolution
    invert-cmt $cmtsolution -i $inputfile -d &> ${logs}/${cmt}.log  & 

    # store PID of process
    pids+=("$!")

    # Store cmt 
    cmts+=("${cmt}")
    
    counter=$((counter+1))

    # Put in a jobs limit
    if [[ $(($counter % $JOBLIMIT)) == 0 ]]
    then
	
	echo Finishing a set of downloads

        # Check status
	for i in ${!pids[@]}; do

	    if wait ${pids[$i]}; then

		echo ${cmts[$i]} SUCCESS

		# Copy cmt to done 
		cp $cmtsolution $downloadeddir/${cmts[$i]}

	    else
		echo ${cmts[$i]} FAIL

		# Copy cmt to done 
		cp $cmtsolution $downloadfaildir/${cmts[$i]}

	    fi

	done
	
	pids=()
	cmts=()

    fi

    if [[ "$counter" -gt "$MAXDL" ]]; then
	break
    fi

    
done


echo Finishing a set of downloads

# Check status
for i in ${!pids[@]}; do

    if wait ${pids[$i]}; then
	echo  ${cmts[$i]} SUCCESS

	# Copy cmt to done 
	cp $cmtsolution $downloadeddir/${cmts[$i]}

    else
	echo ${cmts[$i]} FAIL

        # Copy cmt to done 
        cp $cmtsolution $downloadfaildir/${cmts[$i]}

    fi

done