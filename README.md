# SVD and Symmetric EVD Using Jacobi Rotations

## Cloning the project

This repository uses git lfs to version large files used for benchmarking purposes.
If you want to download these files install git lfs before cloning this repository.
If you try to clone this repository and encounter the following message
`This repository is over its data quota.` it means that the GitHub git lfs bandwidth
limit has been reached and you should clone the repository without the lfs files by
executing `export GIT_LFS_SKIP_SMUDGE=1` before cloning the repository.
The project also contains the Eigen template library as a git submodule.

```bash
export GIT_LFS_SKIP_SMUDGE=1 # ingore git lfs files when cloning
git clone ...
cd jacobi-svd-evd
git submodule update --init  # pull project git submodules
```

## Building the project

Out of source build in build directory

```bash
mkdir build && cd build
cmake ..  # Configures the build type to Release
          # cmake .. -DCMAKE_BUILD_TYPE=Debug to set build for debugging

# Note that if the CMAKE_BUILD_TYPE option is used when running cmake, its value is cached for
# subsequent runs. To clear it delete CMakeCache.txt or set CMAKE_BUILD_TYPE to the wanted value.

make install
cd ..

# If there is any issue clean the build directory and try again
rm -rf build/*
```

## Testing the project

After the project has been build

```bash
cd build
ctest

# Verbose output
ctest -V

# Verbose output with color
GTEST_COLOR=1 ctest -V
```

## Formatting, Linting and Git Hooks
We only accept properly formatted code that passes all the checks performed by cpplint. In order to make this process
seamless, you can follow the below steps:

1. Install clang-format and cpplint programs on your local machine.
2. Copy `scripts/pre-commit.hook` file to `.git/hooks/pre-commit`
3. Now, whenever you attempt to commit your changes, git will automatically format your code with clang-format, and
then run cpplint to check for any linting errors. If any found, commit is aborted, and you will need to address all the
linting errors.

In any case, you can also manually format and lint the code by running
```bash
scripts/format.sh
scripts/lint.sh
```
bash scripts from the project root directory.

## Debugging Utilities

A C++ [pretty printing header library](https://github.com/louisdx/cxx-prettyprint)
is being used to print debugging information for all types that implement `operator<<`.
The library has been extended to handle our custom types, such as the `matrix_t` and
`vector_t` types.

The template variadic `debug` function allows printing of arbitrarily many parameters in one go.
This function also accepts the desired floating-point precision as a template argument.

Usage examples:

```C++
std::vector<double> matrix_data(10000);
std::vector<double> vector_data(200);
struct matrix_t matrix_x = {&matrix_data[0], 50, 200};
struct vector_t vector_v = {&vector_data[0], 200};

debug("Matrix X", matrix_x);                        // default 5 precision
debug<10>("Matrix X", matrix_x);                    // 10 precision
debug("Matrix X", matrix_x, "Vector v", vector_v);  // multiple arguments
```

When running the tests using `ctest -V`, lots of output is created, and finding the debug
output may be problematic. For this reason, all debugging functions print `//` as the first
two characters of every line they print. Therefore, finding your debug output can be done by
`ctest -V | grep //`.

## Generating Benchmarking Inputs
There are two python scripts for generating benchmarking inputs `scripts/evd_benchmark_data.py` and
`scripts/svd_benchmark_data.py`. In the following example we generate two matrices with double
precision floating point numbers, one symmetric of size 1024 by 1024 for our evd benchmark and one
of size 1024 by 2048 for our svd benchmark.

```bash
python evd_benchmark_data.py 1024 > evd_benchmark_1024x1024.txt
python svd_benchmark_data.py 1024 2048 > svd_benchmark_1024x2048.txt
```

## Performance Benchmarking
Performance benchmarks are installed in `bin/benchmark/evd` and `bin/benchmark/svd` directories through `make install`.
To run the performance benchmarks execute execute the script `scripts/benchmark.sh` from the base project directory.

## Performance Plots
Performance plots are currently obtained using `scripts/plot_perf.py`.
To obtain a plot, you need to have at least two files:

1. An `input_sizes` file that contains the size of the input in each line.
2. A performance `perf1` file that contains the performance value corresponding to each line in `input_sizes`.
3. Possibly more similar performance files such as `perf2`, `perf3`, etc.

To get a plot that contains all these performance curves, you can run

```bash
python scripts/plot_perf.py input_sizes perf1 perf2 perf3
```

To save the resulting plot as a .eps file, you can remove the comment at the last line of the script.

## Autotuning
This automated infrastructure searches for the optimal block size and the optimal unrolling
amount for the vectorized subprocedures using grid search. Search for the parameters by running
the commands given below:

For SVD:
```bash
python autotuning/svd/try_params.py &> output.txt
```

For Symmetric EVD:
```bash
python autotuning/evd/try_params.py &> output.txt
```
