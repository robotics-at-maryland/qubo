CXX = g++
CXXFLAGS = -std=c++11 -Wall -pedantic
OBJS = bin/IMU.o bin/util.o

.PHONY: all debug test clean

.INTERMEDIATE: bin/util.o

release: CXXFLAGS += -O3

release: all

debug: CXXFLAGS += -DDEBUG

debug: all

all: bin/IMU.o bin/config bin/baudrate

test: bin/test
	./bin/test /dev/ttyUSB0 38400

valgrind: bin/test
	valgrind --leak-check=yes ./bin/test /dev/ttyUSB0 38400

bin/IMU.o: src/IMU.cpp include/IMU.h include/TRAXTypes.h
	$(CXX) $(CXXFLAGS) -o $@ $< -c 

bin/util.o: src/util.cpp include/util.h include/IMU.h
	$(CXX) $(CXXFLAGS) -o $@ $< -c

bin/config: src/config.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

bin/baudrate: src/baudrate.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

bin/test: src/test.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

clean:
	rm -rf bin/*

