# SVD and Symmetric EVD Using Jacobi Rotations

## Building the project

Out of source build in build directory

```bash
mkdir build && cd build
cmake ..

make && make install
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
2. Copy ```scripts/pre-commit.hook``` file to ```.git/hooks/pre-commit```
3. Now, whenever you attempt to commit your changes, git will automatically format your code with clang-format, and
then run cpplint to check for any linting errors. If any found, commit is aborted, and you will need to address all the
linting errors.

In any case, you can also manually format and lint the code by running
```
scripts/format.sh
scripts/lint.sh
```
bash scripts from the project root directory.

## Debugging Utilities

A C++ [pretty printing header library](https://github.com/louisdx/cxx-prettyprint)
is being used to print debugging information. The library has been extended to handle
our custom types, such as the matrix and vector types.

Usage examples:

```C++
debug("Matrix X", matrix_x);                        // default 5 precision
debug<10>("Matrix X", matrix_x);                    // 10 precision
debug("Matrix X", matrix_x, "Vector v", vector_v);  // multiple arguments
```
