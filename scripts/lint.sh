#!/usr/bin/env bash

files=$(find . -name '*.hpp' -or -name '*.cpp' | grep -viE '\.\/build\/')
cpplint ${files}
