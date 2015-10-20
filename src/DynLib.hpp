
/*
  Simple RAII wrapper around dlopen/dlsym/dlclose for loading dynamic libraries
 */

#pragma once

#include <string>
#include <stdexcept>
#include <dlfcn.h>

class DynLib {
public:
	DynLib() : _lib(0) {}
	explicit DynLib(std::string filename, int flags=RTLD_LAZY) : _lib(0) { this->open(filename, flags); }
	~DynLib() { this->close(); }
	DynLib(DynLib const& p) : _lib(0) { this->open(p._filename, p._flags); }
	DynLib& operator=(DynLib const& p) { if( &p != this ) { this->close(); this->open(p._filename, p._flags); } return *this; }
#if __cplusplus >= 201103L
	DynLib(DynLib&& p) : _lib(0) { this->replace(p); }
	DynLib& operator=(DynLib&& p) { if( &p != this ) { this->close(); this->replace(p); } return *this; }
#endif
	void swap(DynLib& p) {
		std::swap(_lib,      p._lib);
		std::swap(_filename, p._filename);
		std::swap(_flags,    p._flags);
	}
	void* symbol(std::string s) {
		if( !_lib ) {
			throw std::runtime_error("Dynamic library is not open");
		}
		void* ret = ::dlsym(_lib, s.c_str());
		if( !ret ) {
			//throw std::runtime_error("Failed to find symbol "+create_symbol);
			throw std::runtime_error(dlerror());
		}
		return ret;
	}
	void open(std::string filename, int flags=RTLD_LAZY) {
		this->close();
		_filename = filename;
		_flags    = flags;
		_lib      = ::dlopen(_filename.c_str(), _flags);
		if( !_lib ) {
			//throw std::runtime_error("Failed to load "+_filename);
			throw std::runtime_error(dlerror());
		}
	}
	void close() {
		if( _lib ) {
			::dlclose(_lib);
			_lib      = 0;
			_filename = "";
			_flags    = 0;
		}
	}
private:
#if __cplusplus >= 201103L
	void replace(DynLib& p) {
		_lib = p._lib; p._lib = 0;
		_filename = std::move(p._filename);
		_flags    = std::move(p._flags);
	}
#endif
	void*       _lib;
	std::string _filename;
	int         _flags;
};
