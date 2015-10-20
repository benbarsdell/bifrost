
// TODO: Add checks for out_of_range
// TODO: Pull common functionality out into base class shared by StackMultiset

#pragma once

#include <algorithm>
#include <stdexcept>

template<typename T, size_t MAX_SIZE_>
class StackVector {
public:
	enum { MAX_SIZE = MAX_SIZE_ };
	typedef T        value_type;
	typedef T*       iterator;
	typedef T const* const_iterator;
	typedef size_t   size_type;
	
	StackVector();
	explicit StackVector(size_type n, value_type const& value=value_type());
	template<typename Iterator>
	StackVector(Iterator first, Iterator last);
	void swap(StackVector& other);
	
	inline bool           empty()    const { return _size == 0; }
	inline bool           full()     const { return _size == MAX_SIZE; }
	inline size_type      size()     const { return _size; }
	inline size_type      max_size() const { return MAX_SIZE; }
	inline size_type      capacity() const { return MAX_SIZE; }
	inline       iterator begin()          { return &_data[0]; }
	inline const_iterator begin()    const { return &_data[0]; }
	inline       iterator end()            { return &_data[_size]; }
	inline const_iterator end()      const { return &_data[_size]; }
	inline bool operator==(StackVector other) const {
		return (this->size() == other.size() &&
		        std::equal(this->begin(), this->end(), other->begin()));
	}
	inline bool operator!=(StackVector other) const {
		return !(*this == other);
	}
	
	void resize(size_type n, value_type const& value);
	void assign(size_type n, value_type const& value);
	template<typename Iterator>
	void assign(Iterator first, Iterator last);
	void push_back(value_type value);
	void pop_back();
	
	inline value_type const& operator[](size_type i) const { return *(this->begin()+i); }
	inline value_type&       operator[](size_type i)       { return *(this->begin()+i); }
	inline value_type const& front() const { return *(this->begin()); }
	inline value_type&       front()       { return *(this->begin()); }
	inline value_type const& back()  const { return *(--this->end()); }
	inline value_type&       back()        { return *(--this->end()); }
	inline void              clear()       { _size = 0; }
private:
	size_type  _size;
	value_type _data[MAX_SIZE];
};

template<typename T, size_t MAX_SIZE>
StackVector<T, MAX_SIZE>::StackVector() : _size(0) {}
template<typename T, size_t MAX_SIZE>
StackVector<T, MAX_SIZE>::StackVector(size_type n,
                                      value_type const& value)
	: _size(n) {
	this->assign(n, value);
}
template<typename T, size_t MAX_SIZE>
template<typename Iterator>
StackVector<T, MAX_SIZE>::StackVector(Iterator first, Iterator last)
	: _size(last-first) {
	this->assign(first, last);
}
template<typename T, size_t MAX_SIZE>
void StackVector<T,MAX_SIZE>::swap(StackVector& other) {
	if( &other == this ) {
		return;
	}
	std::swap_ranges(this->begin(), this->begin()+std::max(_size, other._size),
	                 other.begin());
	std::swap(_size, other._size);
}
template<typename T, size_t MAX_SIZE>
void StackVector<T, MAX_SIZE>::resize(size_type n, value_type const& value) {
	std::fill(this->end(), this->begin()+n, value);
	_size = n;
}
template<typename T, size_t MAX_SIZE>
void StackVector<T, MAX_SIZE>::assign(size_type n, value_type const& value) {
	return this->resize(n, value);
}
template<typename T, size_t MAX_SIZE>
template<typename Iterator>
void StackVector<T, MAX_SIZE>::assign(Iterator first, Iterator last) {
	std::copy(first, last, this->begin());
	_size = std::distance(first, last); //last - first;
}
template<typename T, size_t MAX_SIZE>
void StackVector<T, MAX_SIZE>::push_back(value_type value) {
	++_size;
	this->back() = value;
}
template<typename T, size_t MAX_SIZE>
void StackVector<T, MAX_SIZE>::pop_back() {
	--_size;
}
