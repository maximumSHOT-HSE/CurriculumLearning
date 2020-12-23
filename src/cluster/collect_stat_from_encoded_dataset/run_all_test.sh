#!/bin/bash

TOTAL_SIZE=162274
BATCH_SIZE=50000

start=0

while (($start < $TOTAL_SIZE))
do
    let "end=start+BATCH_SIZE"
    if (($end > $TOTAL_SIZE))
    then
        let "end=TOTAL_SIZE"
    fi
    echo "[$start, $end)"

    sbatch run_part_train.sh $start $end "save_stat/save_stat_test_$start_$end" test

    let "start += BATCH_SIZE"
done
