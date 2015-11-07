CXX = g++
CXXFLAGS = -std=c++11 -Wall -pedantic -DDEBUG

.PHONY: all test clean

all: bin/IMU.o

test: test/run
	./test/run | tee test/run.out

bin/IMU.o: src/IMU.cpp include/IMU.h include/TRAXTypes.h
	$(CXX) -c src/IMU.cpp -o bin/IMU.o

test/run: src/test.cpp include/test.h bin/IMU.o
	$(CXX) src/test.cpp bin/IMU.o -o test/run

clean:
	rm -rf test/* bin/*

