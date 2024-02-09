#!/usr/bin/env bash

make clean build

echo "" > output.txt
./bin/LightDarkGrayscale.exe -b 45 -d 60 >> output.txt