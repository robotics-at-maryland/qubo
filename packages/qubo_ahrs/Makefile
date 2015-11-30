CXX = g++
CXXFLAGS = -std=c++11 -Wall -pedantic
HEADS = include/IMU.h include/API_Types.h include/TRAX_Types.h
OBJS = bin/IMU.o bin/util.o

.PHONY: all debug test clean

.INTERMEDIATE: bin/util.o

release: CXXFLAGS += -O3

release: all

debug: CXXFLAGS += -DDEBUG

debug: all

all: bin/IMU.o bin/config bin/baudrate

test: bin/test
	./bin/test /dev/ttyUSB0 115200

bin:
	mkdir bin

bin/IMU.o: src/IMU.cpp $(HEADS) bin
	$(CXX) $(CXXFLAGS) -o $@ $< -c 

bin/util.o: src/util.cpp include/util.h $(HEADS) bin
	$(CXX) $(CXXFLAGS) -o $@ $< -c

bin/config: src/config.cpp $(OBJS) bin
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

bin/baudrate: src/baudrate.cpp $(OBJS) bin
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

bin/test: src/test.cpp $(OBJS) bin
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

clean:
	rm -rf bin/*

