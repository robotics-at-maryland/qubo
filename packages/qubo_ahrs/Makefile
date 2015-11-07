CXX = g++
CXXFLAGS = -std=c++11 -Wall -pedantic

.PHONY: all debug test clean

release: CXXFLAGS += -O3

release: all

debug: CXXFLAGS += -DDEBUG

debug: all

all: bin/IMU.o bin/config

test: test/run
	rm -rf test/run.log
	stdbuf -o0 ./test/run | tee test/run.log

bin/IMU.o: src/IMU.cpp include/IMU.h include/TRAXTypes.h
	$(CXX) $(CXXFLAGS) -c src/IMU.cpp -o bin/IMU.o 

bin/config: src/config.cpp bin/IMU.o
	$(CXX) $(CXXFLAGS) src/config.cpp bin/IMU.o -o bin/config

test/run: src/test.cpp bin/IMU.o
	$(CXX) $(CXXFLAGS) src/test.cpp bin/IMU.o -o test/run

clean:
	rm -rf test/* bin/*

