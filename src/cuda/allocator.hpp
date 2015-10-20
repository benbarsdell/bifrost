
#pragma once

#include <stdexcept>
#include <limits>
#include <cuda_runtime_api.h>

namespace cuda {

template<typename T>
class allocator {
public:
	enum space_type { SPACE_DEVICE, SPACE_HOST };
	typedef T                 value_type;
	typedef value_type*       pointer;
	typedef value_type const* const_pointer;
	typedef value_type&       reference;
	typedef value_type const& const_reference;
	typedef std::size_t       size_type;
	typedef std::ptrdiff_t    difference_type;
	template<typename U> struct rebind { typedef cuda::allocator<U> other; };
	/*
	inline explicit allocator(space_type space=SPACE_DEVICE,
	                          unsigned   flags=cudaHostAllocDefault)
		: _space(space), _flags(flags) {}
	*/
	//inline         ~allocator() {}
	//inline explicit allocator(cuda::allocator    const& a) _space(a._space), _flags(a._flags) {}
	//inline cuda::allocator& operator=(cuda::allocator const& a) {
	//	_space = a._space; _flags = a._flags; return *this;
	//}
	template<typename U>
	inline explicit allocator(cuda::allocator<U> const& a)
		: _space(a._space), _flags(a._flags) {}
	inline pointer       address(reference r)       const { return &r; }
	inline const_pointer address(const_reference r) const { return &r; }
	inline pointer allocate(size_type n,
	                        //typename cuda::allocator<void>::const_pointer =0) {
	                        void const* =0) {
		pointer ptr = 0;
		switch( _space ) {
		case SPACE_DEVICE: check_error( cudaMalloc(   (void**)&ptr, n*sizeof(value_type)) );         break;
		case SPACE_HOST:   check_error( cudaHostAlloc((void**)&ptr, n*sizeof(value_type), _flags) ); break;
		}
		return ptr;
	}
    inline void deallocate(pointer p, size_type) {
	    switch( _space ) {
	    case SPACE_DEVICE: cudaFree(p);     break;
		case SPACE_HOST:   cudaFreeHost(p); break;
		}
    }
    inline size_type max_size() const {
	    switch( _space ) {
	    case SPACE_DEVICE: {
		    int device;
		    check_error( cudaGetDevice(&device) );
		    cudaDeviceProp props;
		    check_error( cudaGetDeviceProperties(&props, device) );
		    return props.totalGlobalMem / sizeof(value_type);
	    }
		case SPACE_HOST: return std::numeric_limits<size_type>::max() / sizeof(value_type);
	    default: throw std::invalid_argument("Invalid space");
		}
    }
    inline void construct(pointer p, const_reference val) {
	    switch( _space ) {
	    case SPACE_DEVICE: {
		    // TODO: This will break if value_type has a non-trivial destructor
		    value_type tmp(val);
		    check_error( cudaMemcpy(p, &tmp, sizeof(tmp)) );
		    break;
	    }
	    case SPACE_HOST: new(p) T(val); break;
	    }
    }
    inline void destroy(pointer p) {
	    switch( _space ) {
	    case SPACE_DEVICE: {
		    // TODO: Any way/need to do this?
		    break;
	    }
	    case SPACE_HOST: p->~T(); break;
	    }
    }
	inline bool operator==(allocator const& a) const { return _space == a._space && _flags == a._flags; }
    inline bool operator!=(allocator const& a) const { return !operator==(a); }
	
	// Non-standard functionality
	inline explicit allocator(std::string space="system",
	                          unsigned    flags=cudaHostAllocDefault)
		: _flags(flags) {
		std::transform(space.begin(), space.end(), space.begin(), ::tolower);
		if( space == "host" ||
		    space == "system" ) {
			_space = SPACE_HOST;
		}
		else if( space == "device" ||
		         space == "cuda" ) {
			_space = SPACE_DEVICE;
		}
		else {
			throw std::invalid_argument("Unrecognised space: "+space);
		}
	}
	//space_type space() const { return _space; }
	std::string space() const {
		switch( _space ) {
		case SPACE_HOST:   return "system";
		case SPACE_DEVICE: return "cuda";
		default: throw std::invalid_argument("Unexpected internal space value!");
		}
	}
	
private:
	space_type _space;
	unsigned   _flags;
	void check_error(cudaError_t ret) const {
		if( ret == cudaErrorMemoryAllocation ) {
			throw std::bad_alloc(/*cudaGetErrorString(ret)*/);
		}
		else if( ret != cudaSuccess ) {
			throw std::runtime_error(cudaGetErrorString(ret));
		}
	}
};

} // namespace cuda
