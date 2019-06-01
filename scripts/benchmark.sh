#!/usr/bin/env bash
#
# Usage: Run from the project root directory. Use without any arguments to
#        run all benchmarks. If you give the path to one of the benchmark
#        executables, all the benchmarks for that specific executable will
#        be run only.
#
# Timeout: Each benchmark executable is executed for every benchmark input.
#          There is a hardcoded timeout that is applied to every such execution.
#          Thus, if there is one benchmark executable, four benchmark inputs
#          and a timeout of 1 hour for every execution the whole benchmarking
#          process will take at most 4 hours.

timeout_duration=5  # timeout of 1 hour (3600 seconds)

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

    if [ -f "$output" ];
    then
        # If there is already a benchmark datafile for this benchmark, rename
        # it by introducing the current datetime to the end of its filename.
        dt=$(date '+%s')
        renamed_output="${output}_${dt}"
        mv "$output" "$renamed_output"
        echo "Renamed old benchmark results '$output' to '$renamed_output'."
    fi

    for input in `ls -v ${resources_dir}/*`; do
        # `ls -v ${resources_dir}/*` is used to sort the benchmarks inputs
        # according to input size, as we would like to start from the smallest
        # benchmark and increase the size in every step.
        [ -f "$bin" ] && [ -x "$bin" ] &&
            timeout $timeout_duration "$bin" < "$input" >> "$output"

        # Check if the benchmark timed out
        if [ $? -eq 124 ];
        then
            # Break to save time as the rest of the benchmarks are bigger
            # and thus will not terminate before the timeout.
            echo "Timed out"
            break
        fi
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
