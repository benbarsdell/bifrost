
LIB_DIR  ?= ../lib
BIN_DIR  ?= ../bin
INC_DIR  ?= ../include
LIB_NAME ?= libbifrost
SO_EXT   ?= .so
MAJOR    ?= 1
MINOR    ?= 0.0
SO_FILE   := $(LIB_NAME)$(SO_EXT).$(MAJOR).$(MINOR)
SO_NAME   := $(LIB_DIR)/$(SO_FILE)
CXX_FLAGS  = -g -Wall -O3 -std=c++11 -fPIC -fopenmp
NVCC_FLAGS = -g -Xcompiler "-Wall -fPIC" -O3 -std=c++11
CUDA_DIR  ?= /usr/local/cuda
CUDA_LIBS ?= -L$(CUDA_DIR)/lib64 -lcudart
INCLUDE   = -I$(CUDA_DIR)/include -I.
#ZMQ_LIBS     ?= -lzmq.
#LIB          ?= $(ZMQ_LIBS)
#ZMQ_LIB_DIR ?= /usr/local/lib
LIB     += -ldl -lzmq $(CUDA_LIBS)

OBJECTS ?= Pipeline.o Task.o
HEADERS ?= *.hpp *.h

all: $(SO_NAME) $(BIN_DIR)/bifrost_launch

%.o: %.cpp %.hpp #Makefile
	g++ -c -o $@ $(CXX_FLAGS) $(INCLUDE) $<

$(LIB_DIR):
	mkdir -p $@
$(INC_DIR):
	mkdir -p $@
$(INC_DIR)/bifrost:
	mkdir -p $@
$(INC_DIR)/bifrost/cuda:
	mkdir -p $@

$(SO_NAME): $(OBJECTS) $(HEADERS) Makefile | $(LIB_DIR) $(INC_DIR) $(INC_DIR)/bifrost $(INC_DIR)/bifrost/cuda
	g++ -shared -o $@ -Wl,-soname,$(LIB_NAME)$(SO_EXT).$(MAJOR) $(OBJECTS)
	ln -s -f $(SO_NAME) $(LIB_DIR)/$(LIB_NAME)$(SO_EXT).$(MAJOR)
	ln -s -f $(SO_NAME) $(LIB_DIR)/$(LIB_NAME)$(SO_EXT)
	cp $(HEADERS) $(INC_DIR)/bifrost/
	cp cuda/*.h* $(INC_DIR)/bifrost/cuda/

#-Wl,-unresolved-symbols=ignore-in-shared-libs

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/bifrost_launch: bifrost_launch.cpp Makefile | $(BIN_DIR)
	g++ -o $@ $(CXX_FLAGS) $(INCLUDE) $< -L$(LIB_DIR) -lbifrost $(LIB)

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(LIB_DIR)
	rm -rf $(INC_DIR)
	rm -f *.o
.PHONY: clean
