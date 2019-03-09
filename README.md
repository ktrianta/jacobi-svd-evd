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
cmake test
```
