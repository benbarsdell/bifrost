
#pragma once

#include <stdexcept>
#include <cuda_runtime_api.h>

#include "stream.hpp"

namespace cuda {

template<typename T>
struct copier {
	typedef T                 value_type;
	typedef value_type*       pointer;
	typedef value_type const* const_pointer;
	typedef std::size_t       size_type;
	//inline void operator()(pointer dst, const_pointer src, size_type n) const {
	inline void operator()(const_pointer first, const_pointer last, pointer result) const {
		cuda::independent_stream s;
		//cudaError_t ret = cudaMemcpyAsync(dst, src, n*sizeof(value_type),
		cudaError_t ret = cudaMemcpyAsync(result, first, (last-first)*sizeof(value_type),
		                                  cudaMemcpyDefault, s);
		if( ret != cudaSuccess ) {
			throw std::runtime_error(cudaGetErrorString(ret));
		}
	}
	bool operator==(copier c) const { return true; }
	bool operator!=(copier c) const { return !(*this == c); }
};

} // namespace cuda
