
/*
  A storage class for fixed-point values
  Ben Barsdell (2015)
  Apache v2 license
  
  Note: Assumes values are signed and symmetric
          E.g., 8-bit => [-127:127], not [-128:127]
  Note: Saturates on overflow
  
  Provides explicit conversion (quantization) from float,
    implicit conversion to float, and access to raw bits.
  Supports any nbits up to 32 (but storage rounds up to powers of two)
  
  TODO: Write unit tests!
 */

#pragma once

namespace detail {
template<int N, int S=1>
struct meta_round_up_pow2_impl {
	enum { value = (N >> S) ? meta_round_up_pow2_impl<(N|(N>>S))>::value : N };
};
template<int N>
struct meta_round_up_pow2 {
	enum { value = meta_round_up_pow2_impl<N-1>::value+1 };
};
template<int N> struct FixedStorage {
	typedef typename FixedStorage<meta_round_up_pow2<N>::value>::type type;
};
template<>      struct FixedStorage< 1> { typedef signed char  type; };
template<>      struct FixedStorage< 2> { typedef signed char  type; };
template<>      struct FixedStorage< 4> { typedef signed char  type; };
template<>      struct FixedStorage< 8> { typedef signed char  type; };
template<>      struct FixedStorage<16> { typedef signed short type; };
template<>      struct FixedStorage<32> { typedef signed int   type; };
} // namespace detail

template<int N, int F, typename T=typename detail::FixedStorage<N>::type>
struct __attribute__((aligned(sizeof(T)))) Fixed {
	typedef T value_type;
	inline __host__ __device__ Fixed() : x(0) {}
	inline __device__ explicit Fixed(float f) : x(quantize(f)) {}
	inline __host__ __device__ operator float() const {
		return (float)x * (1.f/(float)SCALE);
	}
	inline __host__ __device__ const T& bits()  const { return x; }
	inline __host__ __device__ T&       bits()        { return x; }
private:
	T x;
	enum {
		SCALE        = (1<<F)-1,
		MAXVAL_FLOAT = (1<<(N-F))-1,
		MINVAL_FLOAT = -MAXVAL_FLOAT
	};
#ifdef __CUDA_ARCH__
	static inline __device__ T quantize(float f) {
		// Saturate on overflow
		f  = min(f, (float)MAXVAL_FLOAT);
		f  = max(f, (float)MINVAL_FLOAT);
		f *= (float)SCALE;
		return __float2int_rn(f);
	}
#else
	static inline T quantize(float f) {
		// Saturate on overflow
		f  = std::min(f, (float)MAXVAL_FLOAT);
		f  = std::max(f, (float)MINVAL_FLOAT);
		f *= (float)SCALE;
		return std::round(f);
	}
#endif
};
