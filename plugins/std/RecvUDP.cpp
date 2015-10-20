
#include "RecvUDP.hpp"

#include <bifrost/utils.hpp>
#include <bifrost/affinity.hpp>

#include <omp.h>

Task* create(Pipeline*     pipeline,
             const Object* definition) {
	return new RecvUDP(pipeline, definition);
}

RecvUDP::RecvUDP(Pipeline*     pipeline,
                 const Object* definition)
	: super_type({},                                       // inputs
	             {"payloads","headers","sizes","sources"}, // outputs
	             pipeline, definition) {
	
	// TODO: Consider accepting a full payload shape parameter that is
	//         multiplied out and added to the header size to derive
	//         the packet size.
	//         The shape could then be passed on to the data output
	//         ** The only downside is the loss of flexibility in being
	//              able to dynamically decide which part of the packet
	//              is the payload.
	
	size_t payload_size_max = lookup_integer(params(), "payload_size_max",
	                                         DEFAULT_PAYLOAD_SIZE_MAX);
	size_t header_size      = lookup_integer(params(), "header_size");
	this->get_output_ring("payloads")->set_shape({payload_size_max});
	this->get_output_ring("headers" )->set_shape({header_size});
}
void RecvUDP::init() {
	super_type::init();
	
	// Bind processing threads to CPU cores
	// TODO: Just use cpu_cores.size() instead of separate ncore?
	// TODO: Abstract this out into a base class (or something) somehow?
	_ncore = lookup_integer(params(), "ncore", 1);
	auto cpu_cores = lookup_list<int>(params(), "cpu_cores", {});
	//_ncore = std::min(_ncore, (int)cpu_cores.size());
	omp_set_num_threads(_ncore);
#pragma omp parallel for schedule(static, 1)
	for( int core=0; core<_ncore; ++core ) {
		int tid = omp_get_thread_num();
		if( tid == 0 ) {
			this->log_info("Using %i cores to process packets",
			               omp_get_num_threads());
		}
		if( !cpu_cores.empty() ) {
			bind_to_core(cpu_cores[tid % cpu_cores.size()]);
		}
	}
	
	// Initialise sockets
	std::string addr = lookup_string(params(), "address");
	int         port = lookup_integer(params(),"port");
#pragma omp parallel for schedule(static, 1)
	for( int core=0; core<_ncore; ++core ) {
		// Note: The code here is not thread safe, but calling it
		//         in a parallel section ensures any potential
		//         memory<->core correspondances will be applied.
		#pragma omp critical
		{
		_sockets.emplace_back(SOCK_DGRAM);
		_sockets.back().bind(Socket::address(addr, port));
		}
	}
	this->log_info("Listening on udp://%s:%i", addr.c_str(), port);
	
	// Initialise stats
	stats_map& stats = _stats;
	stats["nrecv_bytes"] = 0;
	stats["nrecv"]       = 0;
	stats["ndrop"]       = 0;
}
void RecvUDP::process() {
	stats_map& stats = _stats;
	
	auto& payloads_output = this->get_output<0>();
	auto& headers_output  = this->get_output<1>();
	auto& sizes_output    = this->get_output<2>();
	auto& addrs_output    = this->get_output<3>();
	
	size_t payload_size = payloads_output.frame_size();
	size_t header_size  = headers_output.frame_size();
	
	// See here for more info: https://lwn.net/Articles/542629/
	//   Basically, packets are distributed between the sockets
	//     based on their (src,dst)*(addr,port) 4-tuple hash,
	//     which allows capture to be multithreaded.
	// Note: Because packets are distributed between the sockets
	//         based on their (src,dst)*(addr,port) 4-tuple, this
	//         non-pipelined approach to parallelisation is fine.
	//         If packets were instead distributed based on their
	//           TOA then pipelining would be required in order to
	//           remain efficient. One issue with pipelining is
	//           when to do the stats broadcast.
#pragma omp parallel for schedule(static, 1)
	for( int core=0; core<_ncore; ++core ) {
		int tid = omp_get_thread_num();
		RingWriteBlock<char>      pkt_pylds;
		RingWriteBlock<size_type> pkt_sizes;
		RingWriteBlock<addr_type> pkt_addrs;
		RingWriteBlock<char>      pkt_hdrs;
		// Note: Must ensure the opened blocks correspond to each other
#pragma omp critical
		{
			pkt_pylds = payloads_output.open();
			pkt_hdrs  = headers_output.open();
			pkt_sizes = sizes_output.open();
			pkt_addrs = addrs_output.open();
		}
		size_t block_nframe = pkt_pylds.nframe();
		std::cout << "Waiting for block" << std::endl;
		std::cout << pkt_pylds.tail() << ", " << pkt_pylds.head() << std::endl;
		std::cout << pkt_pylds.ring()->tail() << ", " << pkt_pylds.ring()->head() << std::endl;
		size_t npkt = _sockets[tid].recv_block(block_nframe,
		                                       &pkt_hdrs[0],  0, &header_size,
		                                       &pkt_pylds[0], 0, &payload_size,
		                                       &pkt_sizes[0],
		                                       &pkt_addrs[0]);
		std::cout << "Received block of " << npkt << " packets" << std::endl;
		
		stats.at("nrecv_bytes") += _sockets[tid].get_recv_size();
		stats.at("nrecv")       += npkt;
		stats.at("ndrop")       += _sockets[tid].get_drop_count();
		
		// Fill in possibly-unused end of block
		for( size_t p=npkt; p<block_nframe; ++p ) {
			pkt_sizes[p] = 0;
			// Note: Ends of data, address and header blocks are left
			//         uninitialised, so size array should always be checked.
		}
	} // End for each core
	
	this->broadcast("stats", make_Object(stats));
	//std::cout << "end of RecvUDP::process" << std::endl;
}
void RecvUDP::shutdown() {
	Task::shutdown();
	//this->log_debug("RecvUDP::shutdown");
	for( auto& socket : _sockets ) {
		try {
			socket.shutdown();
		}
		catch( Socket::Error ) {}
	}
}
