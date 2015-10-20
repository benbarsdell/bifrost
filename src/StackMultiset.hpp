
/*
  Ordered multiset object with fixed-size storage allocation on the stack
  Ben Barsdell (2015)
  Apache v2 license
  
  Note: This implementation has the following operation complexities:
    insert: linear
    erase:  linear
    find,count,lower_bound,upper_bound,equal_range: logarithmic
    all others: constant
  Note also that the iterators support random access
*/

#pragma once

#include <algorithm>
#include <stdexcept>

template<typename T, size_t MAX_SIZE_>
class StackMultiset {
public:
	enum { MAX_SIZE = MAX_SIZE_ };
	typedef T        value_type;
	typedef T*       iterator;
	typedef T const* const_iterator;
	typedef size_t   size_type;
	
	StackMultiset();
	template<typename Iterator>
	StackMultiset(Iterator first, Iterator last);
	void swap(StackMultiset& other);
	
	iterator insert(value_type const& val);
	void erase(iterator pos);
	void erase(iterator first, iterator last);
	void clear();
	
	inline bool           empty()    const { return _size == 0; }
	inline bool           full()     const { return _size == MAX_SIZE; }
	inline size_type      size()     const { return _size; }
	inline size_type      max_size() const { return MAX_SIZE; }
	inline       iterator begin()          { return &_data[0]; }
	inline const_iterator begin()    const { return &_data[0]; }
	inline       iterator end()            { return &_data[_size]; }
	inline const_iterator end()      const { return &_data[_size]; }
	
	const_iterator find       (value_type const& val) const;
	iterator       find       (value_type const& val);
	size_t         count      (value_type const& val) const;
	const_iterator lower_bound(value_type const& val) const;
	iterator       lower_bound(value_type const& val);
	const_iterator upper_bound(value_type const& val) const;
	iterator       upper_bound(value_type const& val);
	std::pair<const_iterator,const_iterator> equal_range(value_type const& val) const;
	std::pair<      iterator,      iterator> equal_range(value_type const& val);
private:
	size_type  _size;
	value_type _data[MAX_SIZE];
};

template<typename T, size_t MAX_SIZE>
StackMultiset<T,MAX_SIZE>::StackMultiset() : _size(0) {}
template<typename T, size_t MAX_SIZE>
template<typename Iterator>
StackMultiset<T,MAX_SIZE>::StackMultiset(Iterator first, Iterator last)
	: _size(last-first) {
	std::copy(first, last, this->begin());
}
template<typename T, size_t MAX_SIZE>
void StackMultiset<T,MAX_SIZE>::swap(StackMultiset& other) {
	if( &other == this ) {
		return;
	}
	std::swap_ranges(this->begin(), this->begin()+std::max(_size, other._size),
	                 other.begin());
	std::swap(_size, other._size);
}
template<typename T, size_t MAX_SIZE>
typename StackMultiset<T,MAX_SIZE>::iterator
StackMultiset<T,MAX_SIZE>::insert(value_type const& val) {
	if( this->full() ) {
		throw std::range_error("StackMultiset: Cannot insert, "
		                       "container is full");
	}
	// Insertion into sorted array (linear complexity to move existing data)
	iterator pos = this->upper_bound(val);
	//iterator end = this->end();
	std::copy_backward(pos, this->end(), this->end()+1);
	*pos = val;
	++_size;
	return pos;
}
template<typename T, size_t MAX_SIZE>
void StackMultiset<T,MAX_SIZE>::erase(iterator pos) {
	return this->erase(pos, pos+1);
}
template<typename T, size_t MAX_SIZE>
void StackMultiset<T,MAX_SIZE>::erase(iterator first, iterator last) {
	// Linear complexity to move data
	std::copy(last, this->end(), first);
	_size -= last - first;
}
template<typename T, size_t MAX_SIZE>
void StackMultiset<T,MAX_SIZE>::clear() {
	_size = 0;
}
template<typename T, size_t MAX_SIZE>
typename StackMultiset<T,MAX_SIZE>::const_iterator StackMultiset<T,MAX_SIZE>::find(value_type const& val) const {
	iterator pos = this->lower_bound(val);
	return (*pos == val) ? pos : this->end();
}
template<typename T, size_t MAX_SIZE>
typename StackMultiset<T,MAX_SIZE>::iterator StackMultiset<T,MAX_SIZE>::find(value_type const& val) {
	iterator pos = this->lower_bound(val);
	return (*pos == val) ? pos : this->end();
}
template<typename T, size_t MAX_SIZE>
typename StackMultiset<T,MAX_SIZE>::size_type StackMultiset<T,MAX_SIZE>::count(value_type const& val) const {
	return this->upper_bound(val) - this->lower_bound(val);
}
template<typename T, size_t MAX_SIZE>
typename StackMultiset<T,MAX_SIZE>::const_iterator StackMultiset<T,MAX_SIZE>::lower_bound(value_type const& val) const {
	return std::lower_bound(this->begin(), this->end(), val);
}
template<typename T, size_t MAX_SIZE>
typename StackMultiset<T,MAX_SIZE>::iterator StackMultiset<T,MAX_SIZE>::lower_bound(value_type const& val) {
	return std::lower_bound(this->begin(), this->end(), val);
}
template<typename T, size_t MAX_SIZE>
typename StackMultiset<T,MAX_SIZE>::const_iterator StackMultiset<T,MAX_SIZE>::upper_bound(value_type const& val) const {
	return std::upper_bound(this->begin(), this->end(), val);
}
template<typename T, size_t MAX_SIZE>
typename StackMultiset<T,MAX_SIZE>::iterator StackMultiset<T,MAX_SIZE>::upper_bound(value_type const& val) {
	return std::upper_bound(this->begin(), this->end(), val);
}
template<typename T, size_t MAX_SIZE>
std::pair<typename StackMultiset<T,MAX_SIZE>::const_iterator,
          typename StackMultiset<T,MAX_SIZE>::const_iterator>
StackMultiset<T,MAX_SIZE>::equal_range(value_type const& val) const {
	return std::equal_range(this->begin(), this->end(), val);
}
template<typename T, size_t MAX_SIZE>
std::pair<typename StackMultiset<T,MAX_SIZE>::iterator,
          typename StackMultiset<T,MAX_SIZE>::iterator>
StackMultiset<T,MAX_SIZE>::equal_range(value_type const& val) {
	return std::equal_range(this->begin(), this->end(), val);
}
