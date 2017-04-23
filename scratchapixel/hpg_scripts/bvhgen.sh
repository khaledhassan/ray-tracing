#!/usr/bin/env bash


for cores in 1 2 3 4 6 8 10 12 16 20 24 28
do
	cores=$cores prog_dir="/home/khaledjhassan/ray-tracing/scratchapixel/bvh" j2 --format=env bvh.j2 > bvh\_$cores.sh
done