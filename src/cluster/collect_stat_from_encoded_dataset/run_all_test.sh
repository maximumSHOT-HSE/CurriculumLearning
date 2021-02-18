#!/bin/bash

TOTAL_SIZE=273433
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

    sbatch run_part_train.sh $start $end "save_stat_3M_512/save_stat_test_$start_$end" test /home/aomelchenko/datasets/wiki40b_en_3M_tokenized512

    let "start += BATCH_SIZE"
done
