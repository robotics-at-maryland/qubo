CXX = clang
CXXFLAGS = -std=gnu99 -Wall  -O3  -I$(INC_DIR)
#HEADS = include/sensorapi.h include/dvlapi.h include/imuapi.h include/buscodes.h
INC_DIR = include/
#OBJS = bin/sensorapi.o

.PHONY: all clean



all: bin/sensorapi.o bin/dvlapi.o bin/imuapi.o

bin:
	mkdir bin

bin/sensorapi.o: src/sensorapi.c
	$(CXX) $(CXXFLAGS) -o $@ $< -c

bin/dvlapi.o: src/dvlapi.c 
	$(CXX) $(CXXFLAGS) -o $@ $< -c	

bin/imuapi.o: src/imuapi.c 
	$(CXX) $(CXXFLAGS) -o $@ $< -c

clean:
	rm -rf bin
