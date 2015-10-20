CC = gcc

_PHONY: clean

clean:
	rm -rf test imuapi.o test.out

imuapi.o: imuapi.c imuapi.h
	$(CC) -c -o imuapi.o imuapi.c

test: test.c imuapi.o
	$(CC) -o test test.c imuapi.o
	./test

