#!/usr/bin/env bash


for ntasks in 32 16 8 4
do
	for n in 100 1000 10000
	do
		ntasks=$ntasks ntasks_per_node=$(($ntasks/2)) ntasks_per_socket=$(($ntasks/4)) n=$n j2 --format=env hw2_part3_hassan.j2 > hw2_part3_hassan_$ntasks\_$n.sh
	done
done
