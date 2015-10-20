
#pragma once

#include "cuda/stream.hpp"

enum space_type {
	SPACE_SYSTEM = 0,
	SPACE_CUDA   = 1,
	SPACE_AUTO
};

inline void _check_cuda(cudaError_t ret) {
	if( ret != cudaSuccess ) {
		throw std::runtime_error(std::string("CUDA error: ") +
		                         cudaGetErrorString(ret));
	}
}

inline space_type get_pointer_space(void const* ptr) {
	cudaPointerAttributes ptr_attrs;
	_check_cuda( cudaPointerGetAttributes(&ptr_attrs, ptr) );
	switch( ptr_attrs.memoryType ) {
	case cudaMemoryTypeHost:   return SPACE_SYSTEM;
	case cudaMemoryTypeDevice: return SPACE_CUDA;
	default: throw std::invalid_argument("Pointer space is unknown!");
	}
}

template<typename T>
T* allocate(size_t size, space_type space) {
	T* data;
	size_t size_bytes = size*sizeof(T);
	switch( space ) {
	case SPACE_SYSTEM: {
		unsigned flags = cudaHostAllocDefault;
		_check_cuda( cudaHostAlloc((void**)&data, size_bytes, flags) );
		break;
	}
	case SPACE_CUDA: {
		_check_cuda( cudaMalloc((void**)&data, size_bytes) );
		break;
	}
	default: throw std::invalid_argument("Invalid memory space");
	}
	return data;
}
template<typename T>
void deallocate(T* data, space_type space) {
	switch( space ) {
	case SPACE_SYSTEM: cudaFreeHost(data); break;
	case SPACE_CUDA:   cudaFree(data); break;
	default: throw std::invalid_argument("Invalid memory space");
	}
}
template<typename T>
void copy(T const* first, T const* last, T* result,
          space_type src_space=SPACE_AUTO,
          space_type dst_space=SPACE_AUTO) {
	// Note: Must do this to avoid calling get_pointer_space on invalid pointer
	if( last - first == 0 ) {
		return;
	}
	// Note: Explicitly dispatching to ::memcpy was found to be much faster
	//         than using cudaMemcpyDefault.
	if( src_space == SPACE_AUTO ) {
		src_space = get_pointer_space(first);
	}
	if( dst_space == SPACE_AUTO ) {
		dst_space = get_pointer_space(result);
	}
	size_t size_bytes = (last-first)*sizeof(T);
	cudaMemcpyKind kind = cudaMemcpyDefault;
	switch( src_space ) {
	case SPACE_SYSTEM:
		switch( dst_space ) {
		case SPACE_SYSTEM:
			::memcpy(result, first, size_bytes);
			return;
		case SPACE_CUDA: kind = cudaMemcpyHostToDevice; break;
		default: throw std::invalid_argument("Invalid memory space");
		}
	case SPACE_CUDA:
		switch( dst_space ) {
		case SPACE_SYSTEM: kind = cudaMemcpyDeviceToHost; break;
		case SPACE_CUDA:   kind = cudaMemcpyDeviceToDevice; break;
		default: throw std::invalid_argument("Invalid memory space");
		}
	default: throw std::invalid_argument("Invalid memory space");
	}
	cuda::independent_stream s;
	_check_cuda( cudaMemcpyAsync(result, first, size_bytes, kind, s) );
}
inline void fill(char* first, char* last, char value,
          space_type space=SPACE_AUTO) {
	// Note: Must do this to avoid calling get_pointer_space on invalid pointer
	if( last - first == 0 ) {
		return;
	}
	if( space == SPACE_AUTO ) {
		space = get_pointer_space(first);
	}
	switch( space ) {
	case SPACE_SYSTEM: ::memset(first, value, last-first); break;
	case SPACE_CUDA:   _check_cuda( cudaMemset(first, value, last-first) ); break;
	default: throw std::invalid_argument("Unsupported space for fill");
	}
}
