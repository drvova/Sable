// x (any integer/float dtype, up to 4 dims, any strides) -> y = x mod P as int32, row-major.
#include "field.cuh"

template <typename T>
__device__ void f32_body(const T* __restrict__ x, long long n, long long n1, long long n2, long long n3,
                         long long s0, long long s1, long long s2, long long s3, int* __restrict__ y) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    long long r = i, i3 = r % n3; r /= n3;
    long long i2 = r % n2; r /= n2;
    long long i1 = r % n1, i0 = r / n1;
    long long v = (long long)x[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3] % (long long)MP;
    y[i] = (int)(v < 0 ? v + (long long)MP : v);
  }
}

#define F32K(name, T) \
  extern "C" __global__ void name(const T* x, long long n, long long n1, long long n2, long long n3, \
                                  long long s0, long long s1, long long s2, long long s3, int* y) { \
    f32_body<T>(x, n, n1, n2, n3, s0, s1, s2, s3, y); }
F32K(f32_i64, long long)
F32K(f32_i32, int)
F32K(f32_i16, short)
F32K(f32_i8, signed char)
F32K(f32_u8, unsigned char)
F32K(f32_f32, float)
F32K(f32_f64, double)
