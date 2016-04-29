CXX = g++
CXXFLAGS = -std=c++11 -Wall -pedantic -O3
HEADS = include/DVL.h include/types.h include/statics.h include/functions.h
SOURCES = src/DVL.cpp src/impl.cpp src/statics.cpp
OBJS = bin/DVL.o bin/util.o

.PHONY: all test clean

.INTERMEDIATE: bin/util.o

all: bin bin/DVL.o bin/config bin/baudrate

test: all bin/test
	./bin/test /dev/ttyUSB0 115200

bin:
	mkdir bin

bin/DVL.o: src/DVL.cpp $(HEADS) $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $< -c 

bin/util.o: src/util.cpp include/util.h $(HEADS)
	$(CXX) $(CXXFLAGS) -o $@ $< -c

bin/config: src/config.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

bin/baudrate: src/baudrate.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

bin/test: src/test.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

clean:
	rm -rf bin

