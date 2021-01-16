#!/bin/bash

start=$1
end=$2

while (($start <= $end))
do
    scancel $start
    let "start++"
done

