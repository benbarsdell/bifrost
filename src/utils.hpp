
#pragma once

#include <stdexcept>
#include <iostream>
//#include <ctime>
#include "time_portable.h" // For cross-platform clock_gettime

// Note: Seconds must be the last thing in format
inline std::string get_current_utc_string(std::string format="%Y-%m-%dT%H:%M:%S") {
	timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	char buffer[64] = {};
	size_t end = strftime(buffer, 64, format.c_str(), gmtime(&ts.tv_sec));
	sprintf(&buffer[end], ".%09li", ts.tv_nsec);
	return buffer;
}
inline long get_current_clock_ns() {
	timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec*1000000000l + ts.tv_nsec;
}

// Divide n by d and round up to an integer
template<typename T> T div_ceil(T const& n, T const& d) {
	return (n-1)/d+1;
}
// Round x up to a multiple of n
template<typename T> T  ciel_to(T const& x, T const& n)  {
	return div_ceil(x,n)*n;
}

template<typename Container>
typename Container::value_type product_of(Container vals) {
	typename Container::value_type result = 1;
	for( auto x : vals ) {
		result *= x;
	}
	return result;
}
template<typename Iter>
typename Iter::value_type product_of(Iter first, Iter last) {
	typename Iter::value_type result = 1;
	while( first != last ) {
		result *= *(first++);
	}
	return result;
}

template<typename T>
bool product_would_overflow(T a, T b) {
	return (a != 0 &&
	        std::numeric_limits<T>::max() / abs(a) < abs(b));
}

template<typename T>
void partition_balanced(T nitem, T npart, T part_idx,
                        T& part_nitem, T& part_offset) {
	T rem       = nitem % npart;
	part_nitem  = nitem / npart + (part_idx < rem);
	part_offset = ((part_idx < rem) ?
	               part_idx*part_nitem :
	               (rem*(part_nitem+1) + (part_idx-rem)*part_nitem));
}

template<typename Container>
Container broadcast_shapes(Container const& a, Container const& b) {
	size_t na = a.size();
	size_t nb = b.size();
	size_t nc  = std::max(na, nb);
	Container c(nc);
	for( size_t i=0; i<nc; ++i ) {
		size_t sa = (i < na) ? a[na-1-i] : 1;
		size_t sb = (i < nb) ? b[nb-1-i] : 1;
		if( sa == 1 ) {
			c[nc-1-i] = sb;
		}
		else if( sb == 1 ) {
			c[nc-1-i] = sa;
		}
		else if( sa == sb ) {
			c[nc-1-i] = sa;
		}
		else {
			throw std::invalid_argument("Shapes cannot be broadcast");
		}
	}
	return c;
}
/*
template<typename Container>
bool can_broadcast_shape_to(Container const& src, Container const& dst) {
	size_t ns = src.size();
	size_t nd = dst.size();
	if( ns > nd ) {
		return false;
	}
	for( size_t i=0; i<nd; ++i ) {
		// TODO: Implement this
	}
	return true;
}
*/

inline void print_shape(std::vector<size_t> const& s) {
	std::cout << "(";
	for( auto const& d : s ) {
		std::cout << d << ",";
	}
	std::cout << ")";
}

inline int64_t date2mjds(const tm *date) {
	int second = date->tm_sec;
	int minute = date->tm_min;
	int hour   = date->tm_hour;
	// WARNING: Result for leap seconds is degenerate with 1st sec of next day
	//            I.e., date2mjds(23:59:60) == date2mjds(00:00:00)
	//            This must be deal with outside of this function
	//bool is_leap_sec = (hour = 23 && minute == 59 && && second == 60);
	int day    = date->tm_mday;
	int month  = 1    + date->tm_mon;
	int year   = 1900 + date->tm_year;
	int a = (14 - month) / 12;
	int y = year + 4800 - a;
	int m = month + 12*a - 3;
	int64_t jd = ((uint64_t)day + (153*m+2)/5 +
	              365*y + y/4 - y/100 + y/400 - 32045);
	int64_t jd_secs  = second + 60*(minute + 60*((hour-12) + 24*jd));
	int64_t mjd_secs = jd_secs - (int64_t)(2400000.5*24*60*60);
	return mjd_secs;
}

/*
// TODO: This is not actually a very good idea
template<typename T>
class CountingIterator {
public:
	typedef T value_type;
	typedef CountingIterator self_type;
	CountingIterator(value_type idx) : _idx(idx) {}
	// HACK This is the absolute bare minimum implementation, and may not
	//        work under all compilers.
	value_type operator*()  const { return _idx; }
	self_type& operator++()       { ++_idx; return *this; }
	bool operator!=(self_type const& it) { return _idx != it._idx; }
private:
	value_type _idx;
};
template<typename SizeType=size_t>
class XRange {
public:
	typedef SizeType                    size_type;
	typedef CountingIterator<size_type> iterator;
	typedef CountingIterator<size_type> const_iterator;
	XRange(size_type size) : _size(size) {}
	iterator begin() const { return 0; }
	iterator end()   const { return _size; }
private:
	size_type _size;
};
inline XRange<size_t> xrange(size_t size) { return XRange<size_t>(size); }
template<class Sizeable>
inline XRange<typename Sizeable::size_type> xrange(const Sizeable& x) {
	return XRange<typename Sizeable::size_type>(x.size());
}
*/
/*
struct pre_increment {
	template<typename T>
	inline T& operator()(T& x) { return ++x; }
};
*/
// Tuple algorithms
// TODO: These implementations are a bit messy and may not be fully generic
// ----------------
// Converts type tuple<A...> to tuple<F<A>...>
// This uses decltype to exploit function type inference in a class template
template<template<typename...> class V, typename TypeList>
class tuple_of {
	struct InferType {
		template<template<typename...> class TupleType, typename... Ts>
		std::tuple<V<Ts>...> operator()(TupleType<Ts...> const& t);
	};
public:
	typedef typename std::result_of<InferType(TypeList)>::type type;
};

template<int I>
struct integer_value {
	enum { value = I };
};

// Note: C++14 has index_sequence built in
template<int ...>
struct index_sequence {};
template<int N, int ...S>
struct gen_index_sequence : gen_index_sequence<N-1, N-1, S...> {};
template<int ...S>
struct gen_index_sequence<0, S...> {
	typedef index_sequence<S...> type;
	typedef std::tuple<integer_value<S>...> tuple_type;
};
template<typename... Ts>
typename gen_index_sequence<sizeof...(Ts)>::type
make_index_sequence(std::tuple<Ts...> const& ) {
	return typename gen_index_sequence<sizeof...(Ts)>::type();
}
/*
template<int... I>
typename gen_index_sequence<sizeof...(I)>::type
make_index_sequence() {
	return typename gen_index_sequence<sizeof...(I)>::type();
}
*/
template<int N>
typename gen_index_sequence<N>::type
make_index_sequence() {
	return typename gen_index_sequence<N>::type();
}

template<typename... Ts>
typename gen_index_sequence<sizeof...(Ts)>::tuple_type
make_index_tuple(std::tuple<Ts...> const& ) {
	return typename gen_index_sequence<sizeof...(Ts)>::tuple_type();
}

// Transforms each element of a tuple
// We can 'unpack' a tuple argument by using an index_sequence argument
template<int... I, typename UnaryFunction, typename... Ts>
std::tuple<typename std::result_of<UnaryFunction(Ts&)>::type...>
transform_impl(std::tuple<Ts...>& t, UnaryFunction f, index_sequence<I...> ) {
	typedef std::tuple<typename std::result_of<UnaryFunction(Ts&)>::type...> result_type;
	return result_type(f(std::get<I>(t))...);
}
template<typename UnaryFunction, typename... Ts>
std::tuple<typename std::result_of<UnaryFunction(Ts&)>::type...>
transform(std::tuple<Ts...>& t, UnaryFunction f) {
	return transform_impl(t, f, make_index_sequence(t));
}
template<int... I, typename UnaryFunction, typename... Ts>
std::tuple<typename std::result_of<UnaryFunction(Ts const&)>::type...>
transform_impl(std::tuple<Ts...> const& t, UnaryFunction f, index_sequence<I...> ) {
	typedef std::tuple<typename std::result_of<UnaryFunction(Ts const&)>::type...> result_type;
	return result_type(f(std::get<I>(t))...);
}
template<typename UnaryFunction, typename... Ts>
std::tuple<typename std::result_of<UnaryFunction(Ts const&)>::type...>
transform(std::tuple<Ts...> const& t, UnaryFunction f) {
	return transform_impl(t, f, make_index_sequence(t));
}
/*
// TODO: This probably works, but is a bit of a crazy implementation
namespace detail {
template<int... Is>
struct seq { };
template<int N, int... Is>
struct gen_seq : gen_seq<N - 1, N - 1, Is...> { };
template<int... Is>
struct gen_seq<0, Is...> : seq<Is...> { };

template<typename T, typename F, int... Is>
void for_each(T&& t, F f, seq<Is...>) {
	// Note: (xxx, 0)  exploits the comma operator to force evaluation
	//       { ... }   exploits in-order evaluation of initializer lists
	//       (void)xxx exploits void cast to avoid unused variable warning
	auto dummy = { ( f(std::get<Is>(t)), 0 )... };
	(void)dummy;
	// TODO: Can't quite get the above to work; this is a worse version
	//auto dummy = std::make_tuple( (f(std::get<Is>(t)), 0)... );
	//(void)dummy;
}
}
template<typename... Ts, typename F>
void for_each(std::tuple<Ts...>& t, F f) {
	detail::for_each(t, f, detail::gen_seq<sizeof...(Ts)>());
	//detail::for_each(t, f, make_index_sequence(t));
}
*/
template<int N> struct for_each_impl {
	template<class Tuple, class UnaryFunction>
	inline void operator()(Tuple& t, UnaryFunction f) const {
		for_each_impl<N-1>()(t, f);
		f(std::get<N-1>(t));
	}
};
template<> struct for_each_impl<0> {
	template<class Tuple, class UnaryFunction>
	inline void operator()(Tuple& t, UnaryFunction f) const {}
};
template<class Tuple, class UnaryFunction>
inline void for_each(Tuple& t, UnaryFunction f) {
	return for_each_impl<std::tuple_size<Tuple>::value>()(t, f);
}

/*
template<int... I, typename UnaryFunction, typename... Ts>
void for_each_impl(std::tuple<Ts...>& t, UnaryFunction f, index_sequence<I...> ) {
	auto dummy = std::make_tuple(f(std::get<I>(t))...);
}
template<typename UnaryFunction, typename... Ts>
void for_each(std::tuple<Ts...>& t, UnaryFunction f) {
	return for_each_impl(t, f, make_index_sequence(t));
}
*/
template<typename ResultType, typename BinaryFunction,
         typename... Ts>
ResultType
reduce_impl(std::tuple<Ts...> const& t, ResultType init, BinaryFunction f,
            index_sequence<> ) {
	// Base case
	return init;
}
template<int I0, int... I, typename ResultType, typename BinaryFunction,
         typename... Ts>
ResultType
reduce_impl(std::tuple<Ts...> const& t, ResultType init, BinaryFunction f,
            index_sequence<I0, I...> ) {
	// Note: Starts at the end and works backwards
	return f(std::get<sizeof...(I)>(t),
	         reduce_impl(t, init, f, make_index_sequence<sizeof...(I)>()));
}
// Reduces (e.g., sums) tuple elements to a single value
template<typename ResultType, typename BinaryFunction, typename... Ts>
ResultType
reduce(std::tuple<Ts...> const& t, ResultType init, BinaryFunction f) {
	return reduce_impl(t, init, f, make_index_sequence(t));
}
// ----------------
