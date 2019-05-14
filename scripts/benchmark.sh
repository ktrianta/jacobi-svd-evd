#!/usr/bin/env bash
#
# Usage: Run from the project root directory. Use without any arguments to run all benchmarks. If you give the path to
#        one of the benchmark executables, all the benchmarks for that specific executable will be run only.

run_all () {
    bin=$1
    resources_dir=""
    results_dir=""
    if [[ $bin == *"svd"* ]];
    then
        resources_dir=perf/resources/svd
        results_dir=results/svd
    else
        resources_dir=perf/resources/evd
        results_dir=results/evd
    fi

    benchmark=$(basename $bin)
    output="${results_dir}/$benchmark"
    rm "$output"  # delete previous benchmark output
    for input in `ls -v ${resources_dir}/* | head -8`; do
        [ -f "$bin" ] && [ -x "$bin" ] && "$bin" < "$input" >> "$output"
    done
}

param=$1
if [ $# -eq 0 ];
then
    for bin in bin/benchmark/svd/*; do
        run_all $bin
    done

    for bin in bin/benchmark/evd/*; do
        run_all $bin
    done
else
    run_all $param
fi
