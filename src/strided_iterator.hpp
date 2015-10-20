
#pragma once

#include <cassert>

template<typename T>
class strided_iterator {
public:
	typedef T        value_type;
	typedef T*       pointer;
	typedef T const* const_pointer;
	typedef T&       reference;
	typedef T const& const_reference;
	typedef strided_iterator self_type;
	strided_iterator() : _p(0), _stride(0) {}
	explicit strided_iterator(ssize_t stride) : _p(0), _stride(stride) {}
	strided_iterator(pointer p, ssize_t stride) : _p(p), _stride(stride) {}
	bool operator==(self_type const& it) const { return _p == it._p && _stride == it._stride; }
	bool operator!=(self_type const& it) const { return !(*this == it); }
	bool operator< (self_type const& it) const { assert( _stride == it._stride); return _p <  it._p; }
	bool operator<=(self_type const& it) const { assert( _stride == it._stride); return _p <= it._p; }
	bool operator> (self_type const& it) const { assert( _stride == it._stride); return _p >  it._p; }
	bool operator>=(self_type const& it) const { assert( _stride == it._stride); return _p >= it._p; }
	reference       operator*()        { return *_p; }
	const_reference operator*() const  { return *_p; }
	pointer         operator->()       { return _p; }
	const_pointer   operator->() const { return _p; }
	reference       operator[](ssize_t n)       { return *(*this + n); }	
	const_reference operator[](ssize_t n) const { return *(*this + n); }
	// Prefix
	self_type& operator++() { return *this += 1; }
	self_type& operator--() { return *this -= 1; }
	// Postfix
	self_type  operator++(int) { self_type tmp(*this); *this += 1; return tmp; }
	self_type  operator--(int) { self_type tmp(*this); *this -= 1; return tmp; }
	self_type& operator+=(ssize_t n) { _p += n*_stride; return *this; }
	self_type& operator-=(ssize_t n) { _p -= n*_stride; return *this; }
	ssize_t           operator-(self_type const& it) { assert( _stride == it._stride); return (_p - it._p) / _stride; }
	void swap(self_type& it) {
		std::swap(_p, it._p);
		std::swap(_stride, it._stride);
	}
	self_type& set_stride(ssize_t stride) { _stride = stride; return *this; }
	self_type& operator=(pointer p) { _p = p; return *this; }
	ssize_t       stride() const { return _stride; }
	pointer       data()       { return _p; }
	const_pointer data() const { return _p; }
	operator bool() const { return _p != 0; }
private:
	pointer _p;
	ssize_t _stride;
};
template<typename T>
strided_iterator<T> operator+(strided_iterator<T> const& a, ssize_t n) {
	strided_iterator<T> tmp(a);
	return tmp += n;
}
template<typename T>
strided_iterator<T> operator-(strided_iterator<T> const& a, ssize_t n) {
	strided_iterator<T> tmp(a);
	return tmp -= n;
}
template<typename T>
strided_iterator<T> operator+(ssize_t n, strided_iterator<T> const& a) {
	strided_iterator<T> tmp(a);
	return tmp += n;
}
template<typename T>
strided_iterator<T> operator-(ssize_t n, strided_iterator<T> const& a) {
	strided_iterator<T> tmp(a);
	return tmp -= n;
}
