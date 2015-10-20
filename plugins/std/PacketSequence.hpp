
/*
  A container for tracking a sequence of packets with O(logN) range queries
    This is similar to an interval tree but much simpler due to assumption of
      non-overlapping intervals.


  for each packet:
    pkt_dst, pkt_size, pkt_idx = decode_packet(...);
    If not spans.insert(pkt_dst, pkt_size, pkt_idx):
      ++nduplicate;
  //spans.start();
  //spans.extent();
  //Cycle blocks as much as needed without going past the start of the packet span
  For each open block:
    packets = spans.query(block_start, block_size)
    Parallel for each packet:
      Copy overlapping part of packet into block
  While packet span end is > block span end:
    packets = spans.query(block0_start, block0_size)
    For packet in packets:
      If packet.end < next_packet.begin:
        missing_spans.push_back(packet.end, next_packet.begin);
        nmissing_bytes += next_packet.begin - packet.end;
    Parallel for each missing span:
      Fill overlapping part of span in block with fill_char
    spans.erase(any span that ends <= output.head+block_size);
    // erase_before(...)
    Pop off block0
    Open new block
    process(new_block)

TODO: Depacketize: Separate into process_block(block_idx) and release_block()

 */

#pragma once

#include <map>

// Note: PacketSequence::iterator type is such that:
//         iter->first         = start
//         iter->second.first  = size
//         iter->second.second = value
template<typename KeyType, typename MappedType, typename DeltaType=KeyType>
class PacketSequence : public std::map<KeyType,
                                       std::pair<DeltaType,
                                                 MappedType> > {
	typedef std::map<KeyType,
	                 std::pair<DeltaType,
	                           MappedType> > super_type;
public:
	typedef KeyType    key_type;
	typedef MappedType mapped_type;
	typedef DeltaType  delta_type;
	typedef typename super_type::iterator       iterator;
	typedef typename super_type::const_iterator const_iterator;
private:
	// TODO: Find a way to avoid code duplication with these two versions
	// Returns first entry that ends > point
	iterator upper_bound_end(key_type point) {
		iterator ret = this->lower_bound(point);
		// Must check if the previous element ends > point
		if( ret != this->begin() ) {
			iterator prev = ret;
			--prev;
			key_type prev_end = prev->first + prev->second.first;
			if( prev_end > point ) {
				ret = prev;
			}
		}
		return ret;
	}
	const_iterator upper_bound_end(key_type point) const {
		const_iterator ret = this->lower_bound(point);
		// Must check if the previous element ends > point
		if( ret != this->begin() ) {
			const_iterator prev = ret;
			--prev;
			key_type prev_end = prev->first + prev->second.first;
			if( prev_end > point ) {
				ret = prev;
			}
		}
		return ret;
	}
public:
	using super_type::insert;
	// Inserts a single entry
	std::pair<iterator,bool> insert(key_type    start,
	                                delta_type  size,
	                                mapped_type value) {
		return super_type::insert(std::make_pair(start,std::make_pair(size,value)));
	}
	// Returns all entries that intersect with the specified key range
	std::pair<iterator,iterator> query(key_type   start,
	                                   delta_type size=delta_type(0)) {
		return std::make_pair(this->upper_bound_end(start),
		                      this->lower_bound(start+size));
	}
	std::pair<const_iterator,
	          const_iterator> query(key_type   start,
	                                delta_type size=delta_type(0)) const {
		return std::make_pair(this->upper_bound_end(start),
		                      this->lower_bound(start+size));
	}
	// Erases all elements that end <= point and returns the new begin
	iterator erase_before(key_type point) {
		this->erase(this->begin(), this->upper_bound_end(point));
		return this->begin();
	}
	// Returns the range (start,end) spanned by all elements in the container
	std::pair<key_type,key_type> span() const {
		if( this->empty() ) {
			throw std::out_of_range("Request for span of empty sequence");
		}
		const_iterator back_iter = this->end();
		--back_iter;
		return std::make_pair(this->begin()->first,
		                      back_iter->first + back_iter->second.first);
	}
};
