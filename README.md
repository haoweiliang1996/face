# install dlib on ubuntu

dlib only depend on boost and boost-python

when on ubuntu 16.04, just use type:
```bash
sudo apt-get install libboost-python-dev cmake;
pip install scikit-image
pip install dlib
```
to install it

use jupyter-notebook to see the face recognition code
```bash
pip installl jupyter
jupyter notebook knn.ipynb
```
# About the result
there about 7000 pic for about 600 persons

assume 566 persons are in our face database while 56 persons(about 1000 pics) not


use first two pics of the 566 persons to train knn classifier.

when it comes to the others pics plus 56 persons not in face database(total about 5800) ,100 of them are used to choose threshold of knn using logistic regression,while the rest(5700) are used to test the model

we can get  accuracy: 90~92% in our test dataset



# dlib C++ library

Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. See [http://dlib.net](http://dlib.net) for the main project documentation and API reference.



## Compiling dlib C++ example programs

Go into the examples folder and type:

```bash
mkdir build; cd build; cmake .. ; cmake --build .
```

That will build all the examples.
If you have a CPU that supports AVX instructions then turn them on like this:

```bash
mkdir build; cd build; cmake .. -DUSE_AVX_INSTRUCTIONS=1; cmake --build .
```

Doing so will make some things run faster.

## Compiling your own C++ programs that use dlib

The examples folder has a [CMake tutorial](https://github.com/davisking/dlib/blob/master/examples/CMakeLists.txt) that tells you what to do.  There are also additional instructions on the [dlib web site](http://dlib.net/compile.html).

## Compiling dlib Python API

Before you can run the Python example programs you must compile dlib. Type:

```bash
python setup.py install
```

or type

```bash
python setup.py install --yes USE_AVX_INSTRUCTIONS
```

if you have a CPU that supports AVX instructions, since this makes some things run faster.  Note that you need to have boost-python installed to compile the Python API.



## Running the unit test suite

Type the following to compile and run the dlib unit test suite:

```bash
cd dlib/test
mkdir build
cd build
cmake ..
cmake --build . --config Release
./dtest --runall
```

Note that on windows your compiler might put the test executable in a subfolder called `Release`. If that's the case then you have to go to that folder before running the test.

This library is licensed under the Boost Software License, which can be found in [dlib/LICENSE.txt](https://github.com/davisking/dlib/blob/master/dlib/LICENSE.txt).  The long and short of the license is that you can use dlib however you like, even in closed source commercial software.

## dlib sponsors

This research is based in part upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA) under contract number 2014-14071600010. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of ODNI, IARPA, or the U.S. Government.
