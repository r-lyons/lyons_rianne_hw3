# lyons_rianne_hw3

For assignment 4:
The testing data is located in the test/ directory, and the necessary models and vocabulary files for loading are located in the central directory (lyons_rianne_hw3), as well as the Dockerfile and the source code (in asgn4.py).
To build the Dockerfile after cloning this repository: docker build -t lyons_rianne_hw4 .
Building the image should copy in all the files from the current directory, install numpy, cython, and dynet, and includes the run command for the code to test the first and second iterations.
To run the code: docker run lyons_rianne_hw4


For the baseline code:
The testing data is located in src/main/resources, and the code is in src/main/scala/RunBaselines.scala
To run: In the main project directory, at the command line, enter sbt run
