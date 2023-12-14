#!/bin/bash

total_time=0
runs=10

for i in $(seq 1 $runs)
do
    start_time=$(date +%s%N)
    mpirun -n 8 python parallel.py
    end_time=$(date +%s%N)
    elapsed_time=`expr $end_time - $start_time`
    total_time=$(($total_time + $elapsed_time))
done

average_time=$(($total_time / $runs))
average_time_in_seconds=$(echo "scale=9; $average_time/1000000000" | bc)

echo "Average time: $average_time_in_seconds seconds"