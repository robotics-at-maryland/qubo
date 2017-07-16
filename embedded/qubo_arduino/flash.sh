#!/bin/bash

TARGET=qubo_arduino.ino
PORT=/dev/ttyACM0

# Install path of arduino executable
ARDUINO=~/arduino/arduino

$ARDUINO --verify --board arduino:avr:uno $TARGET
$ARDUINO --upload --board arduino:avr:uno --port $PORT $TARGET
