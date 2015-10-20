
#pragma once

#include "value.hpp"

#include <cublas_v2.h>

//cublasCreate(&_cublas_handle);

namespace cuda {

const char* _cublasGetErrorString(cublasStatus_t status) {
	switch( status ) {
	case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS:          The operation completed successfully.";
	case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED:  The cuBLAS library was not initialized.";
	case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED:     Resource allocation failed inside the cuBLAS library.";
	case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE:    An unsupported value or parameter was passed to the function.";
	case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH:    The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.";
	case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR:    An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED: The GPU program failed to execute.";
	case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR:   An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.";
	case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED:    The functionality requested is not supported.";
	case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR:    The functionnality requested requires some license and an error was detected when trying to check the current licensing.";
	default: return "Unknown cublas error!";
	}
}

class cublas {
	cublasHandle_t         _obj;
	cuda::value<float>     _falpha;
	cuda::value<cuComplex> _calpha;
	cuda::value<float>     _fbeta;
	cuda::value<cuComplex> _cbeta;
	// Not copy-assignable
#if __cplusplus >= 201103L
	cublas(const cuda::cublas& other) = delete;
	cublas& operator=(const cuda::cublas& other) = delete;
#else
	cublas(const cuda::cublas& other);
	cublas& operator=(const cuda::cublas& other);
#endif
	void check_error(cublasStatus_t ret) const {
		if( ret != CUBLAS_STATUS_SUCCESS ) {
			throw std::runtime_error(_cublasGetErrorString(ret));
		}
	}
	void destroy() { if( _obj ) { cublasDestroy(_obj); _obj = 0; } }
public:
#if __cplusplus >= 201103L
	// Move semantics
	inline cublas(cuda::cublas&& other)
		: _obj(0) {
		this->swap(other);
	}
	inline cuda::cublas& operator=(cuda::cublas&& other) {
		this->destroy();
		this->swap(other);
		return *this;
	}
#endif
	inline explicit cublas()
		: _obj(0),
		  _falpha(1), _calpha(make_cuComplex(1, 0)),
		  _fbeta( 0), _cbeta( make_cuComplex(0, 0))
	{
		check_error( cublasCreate(&_obj) );
		cublasSetPointerMode(_obj, CUBLAS_POINTER_MODE_DEVICE);
	}
	inline ~cublas() { this->destroy(); }
	inline void swap(cuda::cublas& other) {
		std::swap(_obj, other._obj);
		_falpha.swap(other._falpha);
		_calpha.swap(other._calpha);
		_fbeta.swap(other._fbeta);
		_cbeta.swap(other._cbeta);
	}
	inline cudaStream_t stream() const {
		cudaStream_t val;
		check_error( cublasGetStream(_obj, &val) );
		return val;
	}
	inline void set_stream(cudaStream_t val) {
		check_error( cublasSetStream(_obj, val) );
	}
	inline int version() const {
		int val;
		check_error( cublasGetVersion(_obj, &val) );
		return val;
	}
	// WARNING: This uses synchronous cudaMemcpy, which will block the device
	inline void set_alpha(float val)     { _falpha = val; _calpha = make_cuComplex(val,0); }
	inline void set_alpha(cuComplex val) { _calpha = val; }
	// WARNING: This uses synchronous cudaMemcpy, which will block the device
	inline void set_beta(float val)      { _fbeta = val; _cbeta = make_cuComplex(val,0); }
	inline void set_beta(cuComplex val)  { _cbeta = val; }
	inline void set_matrix(int rows, int cols, int elemSize,
	                       void const* H, int lda,
	                       void*       D, int ldb) {
		cudaStream_t s = this->stream();
		if( s != 0 ) {
			check_error( cublasSetMatrixAsync(rows, cols, elemSize,
			                                  H, lda, D, ldb, s) );
		}
		else {
			check_error( cublasSetMatrix(rows, cols, elemSize,
			                             H, lda, D, ldb) );
		}
	}
	inline void get_matrix(int rows, int cols, int elemSize,
	                       void const* D, int lda,
	                       void*       H, int ldb) {
		cudaStream_t s = this->stream();
		if( s != 0 ) {
			check_error( cublasGetMatrixAsync(rows, cols, elemSize,
			                                  D, lda, H, ldb, s) );
		}
		else {
			check_error( cublasGetMatrix(rows, cols, elemSize,
			                             D, lda, H, ldb) );
		}
	}
	inline void synchronize() const {
		cudaStream_t s = this->stream();
		if( s != 0 ) {
			cudaStreamSynchronize(s);
		}
		else {
			cudaDeviceSynchronize();
		}
	}
	void gemm(cublasOperation_t transa, cublasOperation_t transb,
	          int m, int n, int k,
	          float const* A, int lda,
	          float const* B, int ldb,
	          float*       C, int ldc) {
		check_error( cublasSgemm(_obj, transa, transb, m, n, k,
		                         _falpha.data(),
		                         A, lda,
		                         B, ldb,
		                         _fbeta.data(),
		                         C, ldc) );
	}
	void gemm(cublasOperation_t transa, cublasOperation_t transb,
	          int m, int n, int k,
	          cuComplex const* A, int lda,
	          cuComplex const* B, int ldb,
	          cuComplex*       C, int ldc) {
		check_error( cublasCgemm(_obj, transa, transb, m, n, k,
		                         _calpha.data(),
		                         A, lda,
		                         B, ldb,
		                         _cbeta.data(),
		                         C, ldc) );
	}
	void herk(cublasFillMode_t uplo, cublasOperation_t trans,
	          int n, int k,
	          cuComplex const* A, int lda,
	          cuComplex*       C, int ldc) {
		check_error( cublasCherk(_obj, uplo, trans, n, k,
		                         _falpha.data(),
		                         A, lda,
		                         _fbeta.data(),
		                         C, ldc) );
	}
	inline operator const cublasHandle_t&() const { return _obj; }
};

} // namespace cuda
