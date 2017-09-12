#!/bin/bash

TARGET=qubo_arduino.ino
PORT=/dev/ttyACM0

# Install path of arduino executable
#ARDUINO=~/arduino/arduino

./arduino_headless.sh --verify --board arduino:avr:uno $TARGET
./arduino_headless.sh --upload --board arduino:avr:uno --port $PORT $TARGET
