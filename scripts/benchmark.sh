#!/usr/bin/env sh

for bin in bin/benchmark/svd/*; do
    for input in `ls -v perf/resources/svd/* | head -4`; do
        [ -f "$bin" ] && [ -x "$bin" ] && "$bin" < "$input"
    done
done

for bin in bin/benchmark/evd/*; do
    for input in `ls -v perf/resources/evd/* | head -4`; do
        [ -f "$bin" ] && [ -x "$bin" ] && "$bin" < "$input"
    done
done
