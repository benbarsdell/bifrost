
#pragma once

#include <bifrost/ConsumerTask.hpp>
#include <bifrost/Socket.hpp>

#include <iostream> // Debugging only
#include <map>
#include <string>
#include <atomic>

typedef size_t           size_type;
typedef sockaddr_storage addr_type;
typedef std::tuple<>     input_types;
typedef std::tuple<char,      // payloads
                   char,      // headers
                   size_type, // sizes
                   addr_type  // sources
                   >     output_types;

class RecvUDP
	: public ConsumerTask2<input_types,output_types> {
	typedef  ConsumerTask2<input_types,output_types> super_type;
public:
	enum {
		// Conservative defaults to allow laziness
		DEFAULT_PAYLOAD_SIZE_MAX =    16384,
		////DEFAULT_HEADER_SIZE_MAX =      256,
		//DEFAULT_GULP_SIZE_MIN   = 67108864,
		//DEFAULT_BUFFER_FACTOR   =        3
	};
	RecvUDP(Pipeline*     pipeline,
	        const Object* definition);
  virtual void open();
	virtual void init();
	virtual void process();
	virtual void shutdown();
private:
	typedef Socket                        socket_type;
	typedef std::vector<socket_type>      socket_array;
	typedef std::map<std::string,ssize_t> stats_map;
	socket_array _sockets;
	stats_map    _stats;
	int          _ncore;
};
