
#pragma once

namespace cuda {

// TODO: Could actually implement copy/assign for this
template<typename T>
class value {
public:
	typedef T value_type;
	inline value() : _dptr(0) { this->allocate(); this->set(value_type()); }
	inline /*explicit*/ value(value_type val) : _dptr(0) {
		this->allocate(); this->set(val);
	}
	inline ~value() { this->destroy(); }
	inline void swap(value& other) { std::swap(_dptr, other._dptr); }
#if __cplusplus >= 201103L
	// Move semantics
	inline value(value&& other) : _dptr(0) { this->swap(other); }
	inline value& operator=(value&& other) {
		this->destroy();
		this->swap(other);
		return *this;
	}
#endif
	inline operator value_type() const { return this->get(); }
	inline value_type const* data() const { return _dptr; }
	inline value_type*       data()       { return _dptr; }
private:
#if __cplusplus >= 201103L
	value(const value& other) = delete;
	value& operator=(const value& other) = delete;
#else
	value(const value& other);
	value& operator=(const value& other);
#endif
	void check_error(cudaError_t ret) const {
		if( ret != cudaSuccess ) {
			throw std::runtime_error(cudaGetErrorString(ret));
		}
	}
	void allocate() {
		this->destroy();
		check_error( cudaMalloc((void**)&_dptr, sizeof(value_type)) );
	}
	void set(value_type const& val) {
		check_error( cudaMemcpy(_dptr, &val, sizeof(value_type),
		                        cudaMemcpyDefault) );
	}
	value_type get() const {
		value_type val;
		check_error( cudaMemcpy(&val, _dptr, sizeof(value_type),
		                        cudaMemcpyDefault) );
		return val;
	}
	void destroy() {
		if( _dptr ) {
			check_error( cudaFree(_dptr) );
			_dptr = 0;
		}
	}
	value_type* _dptr;
};

} // namespace cuda
