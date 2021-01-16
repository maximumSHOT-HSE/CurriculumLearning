#!/bin/bash

TOTAL_SIZE=73948816
BATCH_SIZE=500000

start=0

while (($start < $TOTAL_SIZE))
do
    let "end=start+BATCH_SIZE"
    if (($end > $TOTAL_SIZE))
    then
        let "end=TOTAL_SIZE"
    fi
    echo "[$start, $end)"

    sbatch run_part_train.sh $start $end "save_stat/save_stat_train_$start_$end" train 

    let "start += BATCH_SIZE"
done
