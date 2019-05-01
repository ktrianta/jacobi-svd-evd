#!/usr/bin/env sh
#
# Usage: Run from the project root directory. Use without any arguments to run all benchmarks. If you give the path to
#        one of the benchmark executables, all the benchmarks for that specific executable will be run only.

run_all () {
    bin=$1
    dir=""
    if [[ $bin == *"svd"* ]];
    then
        dir=perf/resources/svd
    else
        dir=perf/resources/evd
    fi
    for input in `ls -v ${dir}/* | head -4`; do
        [ -f "$bin" ] && [ -x "$bin" ] && "$bin" < "$input"
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
