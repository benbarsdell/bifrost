
/*
  Utilities for dynamically-typed objects
  (Currently just some thin wrappers around the excellent picojson library)
 */
// TODO: Rename to Value.hpp?

#pragma once

#define PICOJSON_USE_INT64
#include "picojson.hpp"

#include <stdexcept>
#include <string>
#include <vector>

typedef picojson::object Object;
typedef picojson::array  List;
typedef picojson::value  Value;
typedef picojson::null   None;
//using picojson::parse;

// Note: This automatically adds quotes if parsing fails initially
inline Value parse_value(std::string s) {
	Value val;
	std::string err;
	picojson::parse(val, s.begin(), s.end(), &err);
	if( !err.empty() ) {
		val = Value(s);
		/*
		// Try adding quotes
		s = '"'+s+'"';
		err.clear();
		parse(val, s.begin(), s.end(), &err);
		if( !err.empty() ) {
			throw std::invalid_argument(err);
		}
		*/
	}
	return val;
}
template<typename Iter>
inline Value parse_value(Iter begin, Iter end) {
	Value val;
	std::string err;
	picojson::parse(val, begin, end, &err);
	if( !err.empty() ) {
		val = Value(std::string(begin, end));
	}
	return val;
}

class TypeError : public std::runtime_error {
	typedef std::runtime_error super_t;
public:
	virtual const char* what() const throw() {
		return super_t::what();
	}
public:
	TypeError(const std::string& what_arg)
		: super_t(what_arg) {}
};

template<typename T> struct identity_type { typedef T type; };

// TODO: Use these to allow convenient type flexibility in the below functions
//         Also allow implicit deduction from default_value
template<typename T> struct repr_type : public identity_type<T> {};
#define DEFINE_REPR_TYPE(rtype, type) \
	template<> struct repr_type<type> : public identity_type<rtype> {}
DEFINE_REPR_TYPE(int64_t, char);
DEFINE_REPR_TYPE(int64_t, wchar_t);
DEFINE_REPR_TYPE(int64_t, signed char);
DEFINE_REPR_TYPE(int64_t, unsigned char);
DEFINE_REPR_TYPE(int64_t, signed int);
DEFINE_REPR_TYPE(int64_t, unsigned int);
DEFINE_REPR_TYPE(int64_t, signed long);
DEFINE_REPR_TYPE(int64_t, unsigned long);
DEFINE_REPR_TYPE(int64_t, signed long long);
DEFINE_REPR_TYPE(int64_t, unsigned long long);
DEFINE_REPR_TYPE(double, float);
DEFINE_REPR_TYPE(std::string, const char*);
#undef DEFINE_REPR_TYPE

template<typename K, typename V, typename T=typename repr_type<V>::type>
inline Object make_Object(std::map<K,V> const& source) {
	Object obj;
	for( auto const& item : source ) {
		obj[item.first] = Value((T)(item.second));
	}
	return obj;
}
template<typename T, typename K, typename V>
inline Object make_Object_of(std::map<K,V> const& source) {
	return make_Object<K,V,typename repr_type<T>::type>(source);
}

// Convenient generic map lookup functions
template<typename M>
inline typename M::mapped_type& lookup(M& obj, std::string key) {
	typename M::iterator it = obj.find(key);
	if( it == obj.end() ) {
		throw std::out_of_range("Key not found: "+key);
	}
	return it->second;
}
template<typename M>
inline typename M::mapped_type const& lookup(M const& obj, std::string key) {
	typename M::const_iterator it = obj.find(key);
	if( it == obj.end() ) {
		throw std::out_of_range("Key not found: "+key);
	}
	return it->second;
}
template<typename M>
inline typename M::mapped_type const& lookup(M const& obj, std::string key,
                                             typename M::mapped_type const& default_value) {
	try {                        return lookup(obj, key); }
	catch( std::out_of_range ) { return default_value; }
}

// Convenient dynamically-typed map lookup functions
template<typename RT>
inline RT& _lookup(Object& obj, std::string key) {
	Value& val = lookup(obj, key);
	if( !val.is<RT>() ) {
		throw TypeError("Wrong type for key: "+key);
	}
	return val.get<RT>();
}
template<typename RT>
inline RT const& _lookup(Object const& obj, std::string key) {
	Value const& val = lookup(obj, key);
	if( !val.is<RT>() ) {
		throw TypeError("Wrong type for key: "+key);
	}
	return val.get<RT>();
}
#define DEFINE_LOOKUP_FUNCTION(name, type)	  \
	inline type&       lookup_##name(Object&       obj, std::string key) { return _lookup<type>(obj, key); } \
	inline type const& lookup_##name(Object const& obj, std::string key) { return _lookup<type>(obj, key); } \
	inline type const& lookup_##name(Object const& obj, std::string key, type const& default_value) { \
		try {                        return _lookup<type>(obj, key); } \
		catch( std::out_of_range ) { return default_value; } \
	}
DEFINE_LOOKUP_FUNCTION(object, Object)
DEFINE_LOOKUP_FUNCTION(string, std::string)
DEFINE_LOOKUP_FUNCTION(integer,int64_t)
DEFINE_LOOKUP_FUNCTION(list,   List)
DEFINE_LOOKUP_FUNCTION(bool,   bool)
DEFINE_LOOKUP_FUNCTION(float,  double)
#undef DEFINE_LOOKUP_FUNCTION
// Convenient homogeneous list lookup
template<typename T>
inline std::vector<T> lookup_list(Object const& obj, std::string key) {
	List vals = lookup_list(obj, key);
	typedef typename repr_type<T>::type RT;
	std::vector<T> ret;
	for( List::const_iterator it=vals.begin(); it!=vals.end(); ++it ) {
		Value val = *it;
		if( !val.is<RT>() ) {
			throw TypeError("Wrong type for list element: "+key);
		}
		ret.push_back(val.get<RT>());
	}
	return ret;
}
template<typename T>
inline std::vector<T> lookup_list(Object const& obj, std::string key,
                                  std::vector<T> default_value) {
	try {                        return lookup_list<T>(obj, key); }
	catch( std::out_of_range ) { return default_value; }
}
