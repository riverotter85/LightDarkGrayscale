COMPILER=nvcc
COMPILER_FLAGS += $(shell pkg-config --cflags --libs opencv4)

.PHONY: clean build run

build:
	$(COMPILER) src/LightDarkGrayscale.cu --std c++17 -o bin/LightDarkGrayscale.exe -Wno-deprecated-gpu-targets $(COMPILER_FLAGS) -I./lib -I/usr/local/cuda/include -lcuda

clean:
	rm -f ./bin/LightDarkGrayscale.exe

run:
	./bin/LightDarkGrayscale.exe

all: clean build run