
BIFROST_DIR  ?= ../..
BIFROST_LIB  ?= $(BIFROST_DIR)/lib/libbifrost.so
BIFROST_HDR  ?= $(BIFROST_DIR)/include # TODO: Not sure that this works as intended
VERSION_FILE ?= $(BIFROST_DIR)/src/plugin.version
CUDA_DIR  = /usr/local/cuda
CUDA_LIBS = -L$(CUDA_DIR)/lib64 -lcudart
INCLUDE      += -I$(BIFROST_DIR)/include -I$(CUDA_DIR)/include
CXX_FLAGS    += -g -Wall -O3 -std=c++11 -fPIC -fopenmp
NVCC_FLAGS   += -g -Xcompiler "-Wall -fPIC" -O3 -std=c++11

all: RecvUDP.so Depacketize.so FileWrite.so

%.o: %.cpp $(wildcard %.hpp) Makefile
	g++ -c -o $@ $(CXX_FLAGS) $(INCLUDE) $<

RecvUDP.so: RecvUDP.o RecvUDP.hpp $(BIFROST_LIB) $(BIFROST_HDR)
	g++ -shared -o $@ -Wl,--version-script=$(VERSION_FILE),-soname,$@.1 $(CXX_FLAGS) $(INCLUDE) -fopenmp $<

Depacketize.so: Depacketize.o Depacketize_create.o Depacketize.hpp $(BIFROST_LIB) $(BIFROST_HDR)
	g++ -shared -o $@ -Wl,--version-script=$(VERSION_FILE),-soname,$@.1 $(CXX_FLAGS) $(INCLUDE) -fopenmp Depacketize.o Depacketize_create.o

FileWrite.so: FileWrite.o FileWrite.hpp $(BIFROST_LIB) $(BIFROST_HDR)
	g++ -shared -o $@ -Wl,--version-script=$(VERSION_FILE),-soname,$@.1 $(CXX_FLAGS) $(INCLUDE) $<

clean:
	rm -f RecvUDP.so Depacketize.so
	rm -f *.o
.PHONY: clean

