#!/bin/bash

SCREEN=3

Xvfb :$SCREEN -nolisten tcp -screen :$SCREEN 1280x800x24 &

xvfb="$!"

DISPLAY=:$SCREEN arduino $@

kill -9 $xvfb
