
#pragma once

#include "PacketSequence.hpp"

#include <bifrost/ConsumerTask.hpp>

#include <deque>
#include <atomic>
#include <sys/socket.h> // For sockaddr_storage

typedef size_t           size_type;
typedef sockaddr_storage addr_type;

typedef std::tuple<char,      // payloads
                   char,      // headers
                   size_type, // sizes
                   addr_type  // sources
                   > input_types;
typedef std::tuple<char       // data
                   > output_types;

class Depacketize
	: public ConsumerTask2<input_types,output_types> {
	typedef  ConsumerTask2<input_types,output_types> super_type;
public:
	enum {
		DEFAULT_GULP_SIZE_MIN  = 67108864,
		DEFAULT_SCATTER_FACTOR = 3,
		DEFAULT_FILL_CHAR      = 0
	};
	Depacketize(Pipeline*     pipeline,
	            const Object* definition);
	virtual ~Depacketize() {}
	virtual void init();
	virtual void open();
	virtual void process();
	virtual void overwritten();
	// Subclasses override this to compute the last three arguments
	virtual void decode_packet(char const* header,
	                           size_type   size,
	                           addr_type   src_addr,
	                           ssize_t*    payload_offset,
	                           ssize_t*    payload_destination,
	                           ssize_t*    payload_size);
private:
	void clear_output_blocks();
	void open_output_block();
	template<typename Sequence>
	void copy_to_output_block(Sequence const& pkt_seq, int blk_idx);
	void release_output_block();
	
	struct PacketInfo {
		ssize_t     idx;
		char const* ptr;
	};
	PacketSequence<ssize_t,PacketInfo> _packet_sequence;
	PacketSequence<ssize_t,PacketInfo> _new_packet_sequence;
	struct CopyInfo {
		char const* src;
		char*       dst;
		size_t      size;
	};
	std::vector<CopyInfo> _copy_list;
	struct FillInfo {
		char*  ptr;
		size_t size;
	};
	std::vector<FillInfo> _fill_list;
	
	std::deque<RingWriteBlock<char> > _data_output_blocks;
	//std::deque<RingWriteBlock<char> > _mask_output_blocks;
	std::deque<size_t>                _nbyte_blocks;
	size_t _scatter_factor;
	char   _fill_char;
	
	typedef std::map<std::string,std::atomic<size_t> > atomic_stats_map;
	//typedef std::map<std::string,size_t>                      stats_map;
	mutable atomic_stats_map _stats;
	//atomic_stats_map _stats;
};
