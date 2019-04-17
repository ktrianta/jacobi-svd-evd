#!/usr/bin/env sh

for bin in bin/benchmark/*; do
    [ -f "$bin" ] && [ -x "$bin" ] && "$bin"
done
