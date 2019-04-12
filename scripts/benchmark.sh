#!/usr/bin/env sh

filenames=`find perf -type f -name CMakeLists.txt | xargs cat | grep -i add_executable | cut -d' ' -f1 | cut -d '(' -f2 | tr '\n' ' '`

for f in ${filenames}; do
    path=`find build -name ${f}`
    ./${path}
done
