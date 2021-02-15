#!/bin/bash

TOTAL_SIZE=4930406
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

    sbatch run_part_train.sh $start $end "save_stat_3M_512/save_stat_train_$start_$end" train /home/aomelchenko/datasets/wiki40b_en_3M_tokenized512

    let "start += BATCH_SIZE"
done
