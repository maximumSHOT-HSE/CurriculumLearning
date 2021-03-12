#!/bin/bash

TOTAL_SIZE=$1
BATCH_SIZE=$2
SAVE_DIR=$3
DATASET=$4
PART=$5

start=0

while (($start < $TOTAL_SIZE))
do
    let "end=start+BATCH_SIZE"
    if (($end > $TOTAL_SIZE))
    then
        let "end=TOTAL_SIZE"
    fi
    echo "[$start, $end)"

    sbatch run_part.sh $start $end "${SAVE_DIR}/${PART}_${start}_${end}" $DATASET $PART
    
    let "start += BATCH_SIZE"
done

