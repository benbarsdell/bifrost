
/*
  Arithmetic and storage classes for complex numbers
  Ben Barsdell (2015)
  Apache v2 license
  
  Provides the Complex type for fp32 arithmetic
    and the ComplexFixed<N,F> type for fixed-point storage.
 */

#pragma once

#include <cmath>
#include <iostream>

#include "Fixed.hpp"

using std::atan2;

typedef float Real;
typedef float Real32;

struct __attribute__((aligned(8))) Complex {
	union { float x, real, r; };
	union { float y, imag, i; };
	inline __host__ __device__ Complex() : x(0), y(0) {}
	inline explicit __host__ __device__ Complex(Real x_, Real y_=0) : x(x_), y(y_) {}
	
	inline __host__ __device__ Complex& operator+=(Complex c) { x += c.x; y += c.y; return *this; }
	inline __host__ __device__ Complex& operator-=(Complex c) { x -= c.x; y -= c.y; return *this; }
	inline __host__ __device__ Complex& operator*=(Complex c) {
		Complex tmp;
		tmp.x  = x*c.x;
		tmp.x -= y*c.y;
		tmp.y  = y*c.x;
		tmp.y += x*c.y;
		return *this = tmp;
	}
	inline __host__ __device__ Complex& operator/=(Complex c) {
		Real c_mag2;
		c_mag2  = c.x*c.x;
		c_mag2 += c.y*c.y;
		*this *= Complex(c.x, -c.y);
		return *this /= c_mag2;
	}
	inline __host__ __device__ Complex& operator*=(Real s) { x *= s; y *= s; return *this; }
	inline __host__ __device__ Complex& operator/=(Real s) { return *this *= 1/s; }
	inline __host__ __device__ Complex  operator+() const { return Complex(+x,+y); }
	inline __host__ __device__ Complex  operator-() const { return Complex(-x,-y); }
	inline __host__ __device__ Complex conj()  const { return Complex(x, -y); }
	inline __host__ __device__ Real    phase() const { return atan2(y, x); }
	inline __host__ __device__ Real    mag2()  const { Real a = x*x; a += y*y; return a; }
	inline __host__ __device__ Real    mag()   const { return sqrt(this->mag2()); }
	inline __host__ __device__ Real    abs()   const { return this->mag(); }
	inline __host__ __device__ Complex& mad(Complex a, Complex b) {
		x += a.x*b.x;
		x -= a.y*b.y;
		y += a.y*b.x;
		y += a.x*b.y;
		return *this;
	}
	inline __host__ __device__ Complex& msub(Complex a, Complex b) {
		x -= a.x*b.x;
		x += a.y*b.y;
		y -= a.y*b.x;
		y -= a.x*b.y;
		return *this;
	}
	inline __host__ __device__ bool operator==(Complex const& c) const { return (x==c.x) && (y==c.y); }
	inline __host__ __device__ bool operator!=(Complex const& c) const { return !(*this == c); }
	inline __host__ __device__ bool isReal(Real tol=1e-6) const {
		return y/x <= tol;
	}
};
typedef Complex Complex32;
//const Real Complex::tol = 1e-6;
//inline __host__ __device__
//Complex& mad(Complex& c, Complex a, Complex b) { return c.mad(a, b); }
//inline __host__ __device__
//void msub(Complex& c, Complex a, Complex b) { return c.msub(a, b); }
inline __host__ __device__
Complex operator+(Complex const& a, Complex const& b) { Complex c = a; return c += b; }
inline __host__ __device__
Complex operator-(Complex const& a, Complex const& b) { Complex c = a; return c -= b; }
inline __host__ __device__
Complex operator*(Complex const& a, Complex const& b) { Complex c = a; return c *= b; }
inline __host__ __device__
Complex operator/(Complex const& a, Complex const& b) { Complex c = a; return c /= b; }
inline __host__ __device__
Complex operator*(Complex const& a, Real const& b) { Complex c = a; return c *= b; }
inline __host__ __device__
Complex operator*(Real const& a, Complex const& b) { Complex c = b; return c *= a; }
inline __host__ __device__
Complex operator/(Complex const& a, Real const& b) { Complex c = a; return c /= b; }

inline std::ostream& operator<<(std::ostream& stream, Complex const& c) {
	stream << c.x << "," << c.y;
	return stream;
}

// Note: Contrary to Numpy, here we use the convention that Complex<N,F>
//         means N bits per _real_ value, not the whole structure.
//         E.g., Complex<32,F> ~ numpy.complex64
template<int N, int F, bool PACKED=(N<=4)>
struct __attribute__((aligned(2*sizeof(typename Fixed<N,F>::value_type))))
ComplexFixed {
	typedef Fixed<N,F> value_type;
	inline __host__ __device__ ComplexFixed() {}
	inline __device__ explicit ComplexFixed(float x_, float y_=0)
		: x(x_), y(y_) {}
	inline __device__ explicit ComplexFixed(Complex c)
		: x(c.x), y(c.y) {}
	inline __host__ __device__ operator Complex() const {
		return Complex(x, y);
	}
 private:
	Fixed<N,F> x, y;
};
template<int N, int F>
struct ComplexFixed<N,F,true> {
	typedef Fixed<N,F> value_type;
	inline __host__ __device__ ComplexFixed() {}
	inline __device__ explicit ComplexFixed(float x, float y=0)
		: xy(pack(x,y)) {}
	inline __device__ explicit ComplexFixed(Complex c)
		: xy(pack(c.x,c.y)) {}
	inline operator Complex() const { return unpack(xy); }
private:
	signed char xy;
	static inline __device__ int pack(float x, float y) {
		enum { MASK = (1<<N)-1 };
		return ((value_type(x) << N) |
		        (value_type(y) & MASK));
	}
	static inline __host__ __device__ Complex unpack(int xy) {
		return Complex(xy >> N,
		               xy << (32-N) >> (32-N));
	}
};
