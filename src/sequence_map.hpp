
/*
  A (thread-safe) container representing a sequence with ad-hoc
    insertion and orderly consumption.
    Inserted values are assumed to apply up until the next inserted
      value in the sequence; i.e., insertions represent 'updates'.
  Ben Barsdell (2015)
  BSD 3-Clause license

  // TODO: Better name for this?

  // TODO: What if made the initial value -int_max and then
  //         did reads with absolute sequence values?
  //         This would avoid needing to know how to initialise

 */

#pragma once

#include <map>
#include <stdexcept>
#include <mutex>

template<typename Key, typename Value>
class sequence_map {
	typedef std::map<Key, Value> base_type;
public:
	typedef typename base_type::key_type       key_type;
	typedef typename base_type::value_type   value_type;
	typedef typename base_type::mapped_type mapped_type;
	
	// Insert into the sequence
	// Note: If empty, sets beginning of sequence
	//       If not empty and s < beginning of sequence, throws out_of_range
	//       Beginning of sequence is incremeted by pop()
	/*
	mapped_type& operator[](key_type const& s) {
		std::lock_guard<std::mutex> lock(_mutex);
		return this->get(s);
	}
	*/
	// Insert at given point in sequence
	void insert(key_type when, mapped_type value) {
		std::lock_guard<std::mutex> lock(_mutex);
		if( !_map.empty() && when < _map.begin()->first ) {
			throw std::out_of_range("Late sequence entry");
		}
		this->get(when) = value;
	}
	//// Insert 'immediately' (i.e., at next pop())
	//void insert(mapped_type value) { return this->insert(_head, value); }
	// Pop next value from sequence
	// Note: It is safe to use the returned reference until the next pop()
	mapped_type& pop(key_type when) {
		std::lock_guard<std::mutex> lock(_mutex);
		if( _map.empty() ) {
			throw std::out_of_range("Sequence is empty");
		}
		//typename base_type::iterator head_iter = this->find(_head++);
		typename base_type::iterator head_iter = this->find(when);
		// Erase all entries older than the required one
		_map.erase(_map.begin(), head_iter);
		return head_iter->second;
	}
	// Initialise/reset the sequence
	// Note: This is used to set the starting key and value of the sequence
	//mapped_type& reset(key_type const& s0) {
	void reset(key_type when0, mapped_type value) {
		std::lock_guard<std::mutex> lock(_mutex);
		_map.clear();
		this->get(when0) = value;
	}
	//key_type head() const {
	//	std::lock_guard<std::mutex> lock(_mutex);
	//	return _head;
	//}
private:
	mapped_type& get(key_type when) {
		//if( _map.empty() ) {
		//	_head = when;
		//}
		//else if( when < _head ) {
		//	throw std::out_of_range("Late sequence entry");
		//}
		return _map[when];
	}
	// Find the most recent entry at or before 'when'
	typename base_type::iterator       find(key_type const& when) {
		return --_map.upper_bound(when);
	}
	typename base_type::const_iterator find(key_type const& when) const {
		return --_map.upper_bound(when);
	}
	base_type _map;
	//key_type  _head;
	mutable std::mutex _mutex;
};
