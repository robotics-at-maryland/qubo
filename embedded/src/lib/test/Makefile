CFLAGS = -g
CC = gcc

PROGS = test_status_sensor 

all: $(PROGS)

test_status_sensor: test_status_sensor.o status_sensor.o

test_status_sensor.o: test_status_sensor.c status_sensor.h

status_sensor.o: status_sensor.c status_sensor.h

clean: 
	rm -f *.o $(PROGS) a.out

run:
	make && ./test_status_sensor
