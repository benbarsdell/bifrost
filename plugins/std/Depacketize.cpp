
/*
  Depacketizer that takes an unordered sequence of packets as input,
    decodes the packet headers into destination offsets, and scatters
    the payloads to these locations in the output stream.
    Missing bytes are tracked and filled-in with a specified byte.
    The payload streams are only touched once by either a
      memcpy or a memset operation, and these operations are
      parallelised with OpenMP.

  TODO: See if can get it working using CUDA space (be careful with cudaSetDevice in threads)

5x 8ub  roach_id, GbE_id(tuning), nsubband(11), nchan_per_sb(109), sb_id(0-10)
1x 16ub first_channel_of_pkt
1x 8ub  empty
1x 64ub counter

//cpx,chan,pol,stand,
144 chans, 32 inputs, 2 cpx

*/


// **** TODO: Replace the current try/construct/catch 


#include "Depacketize.hpp"

#include <bifrost/affinity.hpp>
#include <bifrost/range.hpp>

#include <omp.h>

Depacketize::Depacketize(Pipeline*     pipeline,
                         const Object* definition)
	: super_type({"payloads","headers","sizes","sources"}, // inputs
	             {}, // input shapes
	             {"data"},                                 // outputs
	             pipeline, definition) { 
	auto input_space = this->get_input_ring("payloads")->space();
	this->get_output_ring("data")->set_space(input_space);
	// Note: Output stream is, by design, a pure byte stream (i.e., shapeless)
	//         This allows the original data stream to have been packetised
	//           in any arbitrary way (e.g., variable-size packets).
}
void Depacketize::init() {
	super_type::init();
	
	_scatter_factor = lookup_integer(params(), "scatter_factor",
	                                 DEFAULT_SCATTER_FACTOR);
	// Note: We must have at least two output blocks open to allow packets
	//         to span the boundary between blocks.
	_scatter_factor = std::max(_scatter_factor, (size_t)2);
	_fill_char      = lookup_integer(params(), "fill_char",
	                                 DEFAULT_FILL_CHAR);
	
	auto& data_output = this->get_output<0>();
	// Note: this task is a special case due to the inputs and output
	//   not being in lock-step, so we need to request the size again manually.
	// TODO: Allow specifying gulp_nframe instead
	//       Allow specifying separate input vs. output gulp sizes?
	//         ** Simply input/output_gulp_size and input/output_buffer_factor?
	// Gulps specified by memory size upper limit
	size_t gulp_size_min = lookup_integer(params(), "gulp_size",
	                                      DEFAULT_GULP_SIZE_MIN);
	data_output.request_size_bytes(gulp_size_min,
	                               gulp_size_min*_scatter_factor);
	
	// TODO: Rename _stats --> stats
	//stats_map _stats;
	_stats["nrecv"]              = 0;
	_stats["nrecv_bytes"]        = 0;
	_stats["ngood_bytes"]        = 0;
	_stats["nignored"]           = 0;
	_stats["nignored_bytes"]     = 0;
	_stats["nlate"]              = 0;
	_stats["nlate_bytes"]        = 0;
	_stats["nrepeat"]            = 0;
	_stats["nrepeat_bytes"]      = 0;
	_stats["noverwritten_bytes"] = 0;
	_stats["npending_bytes"]     = 0;
	_stats["nmissing_bytes"]     = 0;
	/*
	// TODO: Just use cpu_cores.size() instead of separate ncore?
	int ncore = lookup_integer(params(), "ncore", 1);
	auto cpu_cores = lookup_list<int>(params(), "cpu_cores", {});
	//ncore = std::min(ncore, (int)cpu_cores.size());
	omp_set_num_threads(ncore);
#pragma omp parallel for schedule(static, 1)
	for( int core=0; core<ncore; ++core ) {
		if( core == 0 ) {
			this->log().info("Using %i cores to process packets\n",
			               omp_get_num_threads());
		}
		if( !cpu_cores.empty() ) {
			bind_to_core(cpu_cores[core % cpu_cores.size()]);
		}
	}
	//if( !cpu_cores.empty() ) {
	//  bind_to_core(cpu_cores[0]);
	//}
	*/
}

// Default decoder (implements simplest possible case)
void Depacketize::decode_packet(char const* header,
                                size_type   packet_size,
                                addr_type   src_addr,
                                ssize_t*    payload_offset,
                                ssize_t*    payload_destination,
                                ssize_t*    payload_size) {
	struct DefaultHeader {
		uint64_t destination_offset; // Note: Assumed big-endian
	};
	DefaultHeader hdr = *(DefaultHeader*)header;
	*payload_offset      = 0;
	*payload_destination = be64toh(hdr.destination_offset);
	*payload_size        = packet_size - sizeof(DefaultHeader);
}
void Depacketize::clear_output_blocks() {
	_data_output_blocks.clear();
	//_mask_output_blocks.clear();
	_packet_sequence.clear();
}
void Depacketize::open_output_block() {
	auto& data_output = this->get_output<0>();
	_data_output_blocks.push_back(data_output.open());
	//_mask_output_blocks.push_back(_mask_output.open());
	//_nbyte_blocks.push_back(0);
}
template<typename Sequence>
void Depacketize::copy_to_output_block(Sequence const& pkt_seq,
                                       int blk_idx) {
	auto& data_output = this->get_output<0>();
	ssize_t blk_size = data_output.size_bytes();
	ssize_t blk_beg  = data_output.head() + blk_idx*blk_size;
	char*   blk_ptr  = &_data_output_blocks[blk_idx][0];
	//char*  mask_ptr  = &_mask_output_blocks[blk_idx][0];
	// Find where packets overlap with the block
	_copy_list.clear();
	size_t npending_bytes = 0;
	for( auto const& pkt : make_range(pkt_seq.query(blk_beg, blk_size)) ){
		ssize_t pkt_dst     = pkt.first;
		ssize_t pkt_size    = pkt.second.first;
		PacketInfo pkt_info = pkt.second.second;
		char const* pkt_ptr = pkt_info.ptr;
		// Calculate overlap of packet and block
		ssize_t segment_beg  = std::max(pkt_dst, blk_beg);
		ssize_t segment_end  = std::min(pkt_dst + pkt_size,
		                                blk_beg + blk_size);
		size_t segment_size  = segment_end - segment_beg;
		ssize_t pkt_offset   = segment_beg - pkt_dst;
		ssize_t blk_offset   = segment_beg - blk_beg;
		char const* src_ptr  = pkt_ptr + pkt_offset;
		char*       dst_ptr  = blk_ptr + blk_offset;
		_copy_list.push_back(CopyInfo({src_ptr,dst_ptr,segment_size}));
		npending_bytes += segment_size;
	}
	_stats.at("npending_bytes") += npending_bytes;
	space_type data_space = data_output.space();
	//space_type mask_space = _mask_output.space();
	// Copy the packet overlaps into the block
#pragma omp parallel for
	for( int c=0; c<(int)_copy_list.size(); ++c ) {
		char const* __restrict__ src  = _copy_list[c].src;
		char*       __restrict__ dst  = _copy_list[c].dst;
		size_t                   size = _copy_list[c].size;
		// Copy packet payload to destination
		copy(src, src + size, dst, data_space, data_space);
		// TODO: Update output mask. Use separate fill_list array.
	}
}
void Depacketize::release_output_block() {
	auto& data_output = this->get_output<0>();
	size_t  blk_size = data_output.size_bytes();
	ssize_t blk_beg  = data_output.head();
	ssize_t blk_end  = blk_beg + blk_size;
	char*   blk_ptr  = &_data_output_blocks[0][0];
	// Find gaps in the block
	_fill_list.clear();
	size_t nmissing_bytes = 0;
	auto pkt_range = _packet_sequence.query(blk_beg, blk_size);
	//this->log().debug("Npkt in block: %lu\n",
	//                  std::distance(pkt_range.first, pkt_range.second));
	if( pkt_range.first == pkt_range.second ) {
		// No packets at all in block
		_fill_list.push_back(FillInfo({blk_ptr, blk_size}));
		nmissing_bytes = blk_size;
	}
	else {
		// Check for gaps before first packet and after last packet
		auto first_pkt = pkt_range.first;
		auto  last_pkt = pkt_range.second;
		--last_pkt;
		ssize_t first_beg  = first_pkt->first;
		ssize_t  last_beg  =  last_pkt->first;
		ssize_t  last_size =  last_pkt->second.first;
		ssize_t  last_end  =  last_beg + last_size;
		if( first_beg > blk_beg ) {
			char*  ptr  = blk_ptr;
			size_t size = first_beg - blk_beg;
			_fill_list.push_back(FillInfo({ptr, size}));
			nmissing_bytes += size;
		}
		if( last_end < blk_end ) {
			char*  ptr  = blk_ptr + (last_end - blk_beg);
			size_t size = blk_end - last_end;
			_fill_list.push_back(FillInfo({ptr, size}));
			nmissing_bytes += size;
		}
		// Find gaps between packets in the block
		auto cur_pkt = pkt_range.first;
		auto nxt_pkt = cur_pkt;
		++nxt_pkt;
		for( ; nxt_pkt!=pkt_range.second; ++cur_pkt,++nxt_pkt ) {
			ssize_t cur_beg  = cur_pkt->first;
			ssize_t cur_size = cur_pkt->second.first;
			ssize_t cur_end  = cur_beg + cur_size;
			ssize_t nxt_beg  = nxt_pkt->first;
			if( cur_end < nxt_beg ) {
				char*  ptr  = blk_ptr + (cur_end - blk_beg);
				size_t size = nxt_beg - cur_end;
				_fill_list.push_back(FillInfo({ptr, size}));
				nmissing_bytes += size;
			}
		}
	}
	size_t ngood_bytes = blk_size - nmissing_bytes;
	_stats.at("nmissing_bytes") += nmissing_bytes;
	_stats.at("ngood_bytes")    += ngood_bytes;
	_stats.at("npending_bytes") -= ngood_bytes;
	// Fill in the gaps
	space_type data_space = data_output.space();
#pragma omp parallel for
	for( int f=0; f<(int)_fill_list.size(); ++f ) {
		char*  ptr  = _fill_list[f].ptr;
		size_t size = _fill_list[f].size;
		fill(ptr, ptr + size, _fill_char, data_space);
	}
	// Finally, pop off the block
	_packet_sequence.erase_before(blk_end);
	_data_output_blocks.pop_front();
}
void Depacketize::open() {
  // TODO: Just use cpu_cores.size() instead of separate ncore?
	int ncore = lookup_integer(params(), "ncore", 1);
	auto cpu_cores = lookup_list<int>(params(), "cpu_cores", {});
	//ncore = std::min(ncore, (int)cpu_cores.size());
	omp_set_num_threads(ncore);
#pragma omp parallel for schedule(static, 1)
	for( int core=0; core<ncore; ++core ) {
		if( core == 0 ) {
			this->log().info("Using %i cores to process packets\n",
			               omp_get_num_threads());
		}
		if( !cpu_cores.empty() ) {
			bind_to_core(cpu_cores[core % cpu_cores.size()]);
		}
	}
	bind_to_core(cpu_cores[0]);

	super_type::open();
	this->clear_output_blocks();
	for( size_t b=0; b<_scatter_factor; ++b ) {
		this->open_output_block();
	}
}
void Depacketize::process() {
	auto& data_output = this->get_output<0>();
	auto const& pyld_input  = this->get_input<0>();
	auto const& hdr_input   = this->get_input<1>();
	auto const& size_input  = this->get_input<2>();
	auto const& addr_input  = this->get_input<3>();
	ssize_t nframe_in = pyld_input.nframe();
	
	//std::cout << "*** " << pyld_input.frame_size() << ", " << pyld_input.nframe() << std::endl;
	//std::cout << "*** " << data_output.frame_size() << ", " << data_output.nframe() << std::endl;
	
	_new_packet_sequence.clear();
	
	// Decode each packet header
	for( ssize_t pkt_idx=0; pkt_idx<nframe_in; ++pkt_idx ) {
		const char* pkt_ptr  = &pyld_input[pkt_idx];
		const char* pkt_hdr  =  &hdr_input[pkt_idx];
		size_type   pkt_size =  size_input[pkt_idx];
		addr_type   pkt_addr =  addr_input[pkt_idx];
		if( pkt_size == 0 ) {
			continue; // Skip empty entries
		}
		++_stats.at("nrecv");
		_stats.at("nrecv_bytes") += pkt_size;
		
		ssize_t payload_src;  // Byte offset from start of packet data
		ssize_t payload_dst;  // Byte offset from start of output stream
		ssize_t payload_size; // No. bytes comprising payload
		this->decode_packet(pkt_hdr, pkt_size, pkt_addr,
		                    &payload_src, &payload_dst, &payload_size);
		if( payload_size == 0 ) {
			++_stats.at("nignored");
			_stats.at("nignored_bytes") += pkt_size;
			continue;
		}
		if( payload_dst + payload_size < data_output.head() ) {
			++_stats.at("nlate");
			_stats.at("nlate_bytes") += payload_size;
			continue;
		}
		const char* payload_ptr = pkt_ptr + payload_src;
		if( !_new_packet_sequence.insert(payload_dst,
		                                 payload_size,
		                                 PacketInfo({pkt_idx,
					                                 payload_ptr})).second ) {
			++_stats.at("nrepeat");
			_stats.at("nrepeat_bytes") += payload_size;
		}
	}
	if( !_new_packet_sequence.empty() ) {
		// Copy new packets into currently-open blocks
		ssize_t nblock = _data_output_blocks.size();
		for( int b=0; b<(int)nblock; ++b ) {
			//this->process_output_block(b);
			this->copy_to_output_block(_new_packet_sequence, b);
		}
		_packet_sequence.insert(_new_packet_sequence.begin(),
		                        _new_packet_sequence.end());
		// Advance output blocks until all packets have been processed
		ssize_t pkt_span_end = _packet_sequence.span().second;
		ssize_t blk_size     = data_output.size_bytes();
		while( pkt_span_end > data_output.head() + nblock*blk_size ) {
			this->release_output_block();
			this->open_output_block();
			//this->process_output_block(nblock-1);
			ssize_t new_blk = nblock-1;
			this->copy_to_output_block(_packet_sequence, new_blk);
		}
	}
	// Update and broadcast stats
	Object stats_obj = make_Object_of<size_t>(_stats);
	stats_obj["head"]         = Value(data_output.head());
	stats_obj["reserve_head"] = Value(data_output.reserve_head());
	stats_obj["tail"]         = Value(data_output.tail());
	this->broadcast("stats", stats_obj);
	//std::cout << "Done broadcast" << std::endl;
}
void Depacketize::overwritten() {
	std::cout << this->name() << "::overwritten()" << std::endl;
	
	auto& data_output = this->get_output<0>();
	auto& pyld_input  = this->get_input<0>();
	auto& hdr_input   = this->get_input<1>();
	auto& size_input  = this->get_input<2>();
	auto& addr_input  = this->get_input<3>();
	
	ssize_t blk_size = data_output.size_bytes();
	_stats.at("noverwritten_bytes") += blk_size;
	
	// ** TODO: Fill in overwritten blocks
	
	// Skip ahead to the latest data
	ssize_t  cur_pkt = pyld_input.frame0();
	ssize_t head_pkt = pyld_input.head();
	ssize_t npkt_advance = head_pkt - cur_pkt;
	
	pyld_input += npkt_advance;
	hdr_input  += npkt_advance;
	size_input += npkt_advance;
	addr_input += npkt_advance;
}
