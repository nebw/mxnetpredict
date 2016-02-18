# mxnetpredict

A simple C++ wrapper for MXNet predictors.

## Build

Make sure you have OpenCV and CUDA installed.

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

The binaries should be now in the `build/source/` directory.

## Run the Tests

Built the code and then run in you build directory:

```
$ ctest
```
