
/*
  TODO: Having shutdown events propagate down the pipeline (in
          lock-step with data via RingBuffer) would allow
          partially-filled buffers (e.g., pending data) to be flushed
          downstream and finished off before everything is closed.
          This is mostly relevant to batch processing.

  [DONE]TODO: Stats tracking and broadcast for all tasks
          In particular, track wait time as fraction of total time (i.e., load %)
          Need to be able to subscribe to broadcasts by name
            Use a single PUB socket per pipeline and use topic filtering
              Topic identifiers: "%s.%s" % (_name, data_name)


  TODO: Support some sort of launch policy definition?
          E.g., shared core-mask between all tasks,
                task-specific core-idx,
                task-specific core-counts, dedicated=true/false,
                round-robin through core-mask, etc.

  [cpu_core_group (default=0)]
  cpu_core_count (default=1)
  cpu_core_dedicated (default=true)
  // Bind to set of cores to ensure containment
  
  cpu_cores:   [2] // Which CPU core of this process's set to use
  gpu_devices: [1] // Which GPU of this process's set to use
  gpu_devices: ["$gpu_devices.gtx750", "$gpu_devices[1]"]
  
  Process specifies:
    gpu_devices: [0,1,"06:00.0"]
    cpu_cores:   [1, 2, 3, 4, 5]
  
  [DONE]** TODO: Work out how to manage memory spaces
             How/where to convert between string names and allocator spaces?
               Build everything into cuda::allocator?
           Work out what to do when !still_valid() occurs (write.seek?)
           Work out what to do when a read skips frames (write.seek to match?)
           Work out what to do when an open read or write times out (undesired partial buffer?)
  
  [DONE]** TODO: Change packet capture system:
             RecvUDP should copy first _header_size of each packet to _header_output
               Should also use WriteBlock outputs, as sequence doesn't matter
             Depacketize should call decode_packet_header(_header_input) --> dst,src_offset,size
               and then spawn asynchronous copy operations, sync'ing at end of input block.

  TODO: Spatial filtering task:
          Input: [time, chan, stand, pol, cpx] fixed8.7

  TODO: Generic tasks to implement:
          N-ary transform: outputs[] = f(inputs[])
          MapReduce:       outputs[] = reduce_by_key(sort(transform(inputs[])))
          Einsum
          FFT
            cufftPlanMany + callbacks:
              in/out_nbit,ndim,sizes,in/out_capacities,in/out_elstrides,nbatch(or gulp_size?)
              TODO: How to deal with, e.g., separately transforming interleaved X/Y pols?
                      Could specify a second task with an element offset of 1 that
                        outputs to the exact same buffer?
                        Not a bad capability to have(!), but complicated (at this point)
                      OR, specify ninterleave as an added feature?
                        Multiply in/out_elstrides by ninterleave to get actual elstrides
                        This would be much easier and could also be useful more generally
          Convolve
          Signal generation (white noise, point/gaussian sources, pulsars etc.)
          Disk read/write

  Typical data formats:
    LEDA:   time,chan,ant,pol,cpx,4bit
    Parkes: time,chan,2bit
    Beam:   beam,chan,time,cpx,8bit
    Corr:   time,chan,bsln,cpx,32bit

  Heimdall pipeline:
      time,chan,2/8bit
    Transpose
      chan,time,2/8bit
    Dedisperse
      dm,time,32bit
    Matched filter
      dm,filter,time,32bit
    Detect peaks
      
  SETI pipeline:
      Time,time,chan,cpx,8bit
    Transpose
      Time,chan,time,cpx,8bit
    C2C FFT
      Time,chan,subchan,cpx,8bit
    Power spectrum
      Time,finechan,16bit
    Divide out smoothed baseline
      Time,finechan,16bit
    Extract peaks
      peak: time;chan;power;baseline;strength;nnearby;etc.
  
  TODO: Task creation: Recursively go up chain of __input__ references to create tasks
                         in order of connection.
                         ACTUALLY, just create all outputs first, then all inputs

 */

#pragma once

#include "Object.hpp"
#include "RingBuffer.hpp"
#include "DynLib.hpp"
#include "ZMQSocket.hpp"
#include "Log.hpp"
#include "cuda/allocator.hpp"
#include "cuda/copier.hpp"

#include <string>
#include <map>
#include <stdexcept>
#include <memory>
#include <fstream>

class Task;

class Pipeline {
public:
	enum {
		DEFAULT_BROADCAST_PORT = 7777
	};
	typedef cuda::allocator<uint8_t> allocator_type;
	typedef cuda::copier<uint8_t>    copier_type;
	//typedef RingBuffer<uint8_t,allocator_type,copier_type> ring_type;
	typedef RingBuffer               ring_type;
	typedef std::shared_ptr<Task>    task_pointer;
	typedef Logger<Pipeline>         log_type;
	
	Pipeline(int io_threads=3);
	std::string name() const { return _name; }
	void load(std::string filename, Object process_attrs=Object());
	inline Object&       params()       { return lookup_object(_definition.get<Object>(), "__pipeline__"); }
	inline Object const& params() const { return lookup_object(_definition.get<Object>(), "__pipeline__"); }
	task_pointer  create_task(Value       definition,
	                          bool        substitute_refs=true);
	//void          launch_task(std::string name);
	//void            stop_task(std::string name);
	//void         destroy_task(std::string name);
	
	ring_type* create_ring(std::string name,
	                       space_type  space);
	ring_type* create_ring(std::string name,
	                       std::string space);
	//template<class R>
	//std::shared_ptr<R> get_ring(std::string name) {
	ring_type* get_ring(std::string name);
	void launch();
	void shutdown();
	void wait();
	inline log_type&       log()            { return _log; }
	inline log_type const& log()      const { return _log; }
	void broadcast(std::string topic,
	               Object      metadata,
	               char const* data=0,
	               size_t      size=0) const;
	zmq::context_t& zmq() { return _zmq_ctx; }
	//inline PUBSocket&       broadcast_socket()       { return _pubsock; }
	//inline PUBSocket const& broadcast_socket() const { return _pubsock; }
private:
	void init();
	//typedef std::pair<task_pointer,cpu_core_type>  task_entry_type;
	//typedef std::map<std::string, task_entry_type> task_map_type;
	typedef std::map<std::string, task_pointer>  task_map_type;
	typedef std::map<std::string, ring_type>     ring_map_type;
	typedef std::map<std::string, DynLib>        lib_map_type;
	std::string   _name;
	// IMPORTANT: _libs must be listed before _tasks so that it is
	//              destructed *after* _tasks. This is because the memory
	//              holding the Tasks' virtual tables is released when the
	//              corresponding dynamic library is unloaded.
	lib_map_type      _libs;
	ring_map_type     _rings;
	task_map_type     _tasks;
	Value             _definition;
	log_type          _log;
	zmq::context_t    _zmq_ctx;
	mutable PUBSocket _pubsock;
};
