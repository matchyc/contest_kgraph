#include "kgraph.h"
#include "kgraph-data.h"

namespace kgraph {

float float_l2sqr (float const *t1, float const *t2, unsigned dim) {
    float sum = 0;
    for (unsigned i = 0; i < dim; ++i) {
        float v = t1[i] - t2[i];
        sum += v * v;
    }
    return sum;
}

float float_l2sqr (float const *t1, unsigned dim) {
    float sum = 0;
    for (unsigned i = 0; i < dim; ++i) {
        sum += t1[i] * t1[i];
    }
    return sum;
}

float float_dot (float const *t1, float const *t2, unsigned dim) {
    float sum = 0;
    for (unsigned i = 0; i < dim; ++i) {
        sum += t1[i] * t2[i];
    }
    return sum;
}
}

#ifdef __GNUC__
#ifdef __AVX__
// #if 1
#include <immintrin.h>
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_load_ps(addr1);\
    tmp2 = _mm256_load_ps(addr2);\
    tmp1 = _mm256_sub_ps(tmp1, tmp2); \
    dest = _mm256_fmadd_ps(tmp1, tmp1, dest) 
namespace kgraph {

    float float_l2sqr_avx (float const *t1, float const *t2, unsigned dim) {
        __m256 sum;
        __m256 l0, l1, l2, l3;
        __m256 r0, r1, r2, r3;
        unsigned D = (dim + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
        unsigned DR = D % 32;
        unsigned DD = D - DR;
        const float *l = t1;
        const float *r = t2;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
        float ret = 0.0;
        sum = _mm256_load_ps(unpack);
        switch (DR) {
            case 24:
                AVX_L2SQR(e_l+16, e_r+16, sum, l2, r2);
            case 16:
                AVX_L2SQR(e_l+8, e_r+8, sum, l1, r1);
            case 8:
                AVX_L2SQR(e_l, e_r, sum, l0, r0);
        }
        for (unsigned i = 0; i < DD; i += 32, l += 32, r += 32) {
            AVX_L2SQR(l, r, sum, l0, r0);
            AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
            AVX_L2SQR(l + 16, r + 16, sum, l2, r2);
            AVX_L2SQR(l + 24, r + 24, sum, l3, r3);
        }
        _mm256_store_ps(unpack, sum);
        ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
            + unpack[4] + unpack[5] + unpack[6] + unpack[7];
        return ret;//sqrt(ret);
    }

    float avx512_l2_distance(float const * a, float const * b, unsigned n) {
        const int kFloatsPerVec = 16;
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        // Process 32 floats at once
        for (int i = 0; i + 2*kFloatsPerVec <= n; i += 2*kFloatsPerVec) {
            // Load two sets of 32 floats from a and b with aligned memory access
            __m512 a_vec1 = _mm512_load_ps(&a[i]);
            __m512 a_vec2 = _mm512_load_ps(&a[i + kFloatsPerVec]);
            __m512 b_vec1 = _mm512_load_ps(&b[i]);
            __m512 b_vec2 = _mm512_load_ps(&b[i + kFloatsPerVec]);

            __m512 diff1 = _mm512_sub_ps(a_vec1, b_vec1);
            sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
            __m512 diff2 = _mm512_sub_ps(a_vec2, b_vec2);
            sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
        }

        // Combine the two sum vectors to a single sum vector
        __m512 sum12 = _mm512_add_ps(sum1, sum2);

        float result = 0;
        // Sum the remaining floats in the sum vector using non-vectorized operations
        for (int j = 0; j < kFloatsPerVec; j++) {
            result += ((float*)&sum12)[j];
        }
        // Process the remaining elements with non-vectorized operations
        for (int i = n - (n % kFloatsPerVec); i < n; i++) {
            float diff = a[i] - b[i];
            result += diff * diff;
        }
        return result;
    }

    float avx512_l2_distance_opt(float const * a, float const * b, unsigned n) {
        const uint32_t kFloatsPerVec = 16;
        __m512 sum1 = _mm512_setzero_ps();
        // n = 112; // for debug, hard code, just replace with `n` is OK, 'n' is aligned dimension passed in. 

        for (uint32_t i = 0; i + kFloatsPerVec <= n; i += kFloatsPerVec) {
            // Load two sets of 32 floats from a and b with aligned memory access
            __m512 a_vec1 = _mm512_load_ps(&a[i]);
            __m512 b_vec1 = _mm512_load_ps(&b[i]);
            __m512 diff1 = _mm512_sub_ps(a_vec1, b_vec1);
            sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        }

        // Combine the two sum vectors to a single sum vector

        float result = 0;
        // Sum the remaining floats in the sum vector using non-vectorized operations
        for (int j = 0; j < kFloatsPerVec; j++) {
            result += ((float*)&sum1)[j];
        }
        return result;
    }
    static inline __m128 masked_read (int d, const float *x)
    {
        assert (0 <= d && d < 4);
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
        switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
        }
        return _mm_load_ps (buf);
        // cannot use AVX2 _mm_mask_set1_epi32
    }
    float avx512_l2_dist_v2 (const float *x, const float *y, unsigned d)
    {
        __m512 msum1 = _mm512_setzero_ps();
        // d = 100;
        while (d >= 16) {
            __m512 mx = _mm512_load_ps (x); x += 16;
            __m512 my = _mm512_load_ps (y); y += 16;
            const __m512 a_m_b1 = mx - my;
            msum1 += a_m_b1 * a_m_b1;
            d -= 16;
        }

        __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
        msum2 +=       _mm512_extractf32x8_ps(msum1, 0);

        while (d >= 8) {
            __m256 mx = _mm256_load_ps (x); x += 8;
            __m256 my = _mm256_load_ps (y); y += 8;
            const __m256 a_m_b1 = mx - my;
            msum2 += a_m_b1 * a_m_b1;
            d -= 8;
        }

        __m128 msum3 = _mm256_extractf128_ps(msum2, 1);
        msum3 +=       _mm256_extractf128_ps(msum2, 0);

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps (x); x += 4;
            __m128 my = _mm_loadu_ps (y); y += 4;
            const __m128 a_m_b1 = mx - my;
            msum3 += a_m_b1 * a_m_b1;
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read (d, x);
            __m128 my = masked_read (d, y);
            __m128 a_m_b1 = mx - my;
            msum3 += a_m_b1 * a_m_b1;
        }

        msum3 = _mm_hadd_ps (msum3, msum3);
        msum3 = _mm_hadd_ps (msum3, msum3);
        return  _mm_cvtss_f32 (msum3);
    }

    float avx2_l2_distance(const float* a, const float* b, unsigned dim) {
        __m256 sum = _mm256_setzero_ps(); // Initialize sum to 0
        unsigned i;
        dim = 104;
        for (i = 0; i + 7 < dim; i += 8) { // Process 8 floats at a time
            __m256 a_vec = _mm256_load_ps(&a[i]); // Load 8 floats from a
            __m256 b_vec = _mm256_load_ps(&b[i]); // Load 8 floats from b
            __m256 diff = _mm256_sub_ps(a_vec, b_vec); // Calculate difference
            sum = _mm256_fmadd_ps(diff, diff, sum); // Calculate sum of squares
        }
        float result = 0;
        // float temp[8] __attribute__((aligned(32)));
        // _mm256_store_ps(temp, sum);
        for (unsigned j = 0; j < 8; ++j) { // Reduce sum to a single float
            result += ((float*)&sum)[j];
        }
  
        return result; // Return square root of sum
    }

}
#endif
#ifdef __SSE2__
#include <xmmintrin.h>
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);\
    tmp2 = _mm_load_ps(addr2);\
    tmp1 = _mm_sub_ps(tmp1, tmp2); \
    tmp1 = _mm_mul_ps(tmp1, tmp1); \
    dest = _mm_add_ps(dest, tmp1); 
namespace kgraph {
float float_l2sqr_sse2 (float const *t1, float const *t2, unsigned dim) {
    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (dim + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = t1;
    const float *r = t2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm_load_ps(unpack);
    switch (DR) {
        case 12:
            SSE_L2SQR(e_l+8, e_r+8, sum, l2, r2);
        case 8:
            SSE_L2SQR(e_l+4, e_r+4, sum, l1, r1);
        case 4:
            SSE_L2SQR(e_l, e_r, sum, l0, r0);
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        SSE_L2SQR(l, r, sum, l0, r0);
        SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
        SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
        SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
    return ret;//sqrt(ret);
}

#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);\
    tmp2 = _mm_load_ps(addr2);\
    tmp1 = _mm_mul_ps(tmp1, tmp2); \
    dest = _mm_add_ps(dest, tmp1); 

float float_dot_sse2 (float const *t1, float const *t2, unsigned dim) {
    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (dim + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = t1;
    const float *r = t2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm_load_ps(unpack);
    switch (DR) {
        case 12:
            SSE_DOT(e_l+8, e_r+8, sum, l2, r2);
        case 8:
            SSE_DOT(e_l+4, e_r+4, sum, l1, r1);
        case 4:
            SSE_DOT(e_l, e_r, sum, l0, r0);
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        SSE_DOT(l, r, sum, l0, r0);
        SSE_DOT(l + 4, r + 4, sum, l1, r1);
        SSE_DOT(l + 8, r + 8, sum, l2, r2);
        SSE_DOT(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
    return ret;//sqrt(ret);
}

#define SSE_L2SQR_1(addr1, dest, tmp1) \
    tmp1 = _mm_load_ps(addr1);\
    tmp1 = _mm_mul_ps(tmp1, tmp1); \
    dest = _mm_add_ps(dest, tmp1); 

float float_l2sqr_sse2 (float const *t1, unsigned dim) {
    __m128 sum;
    __m128 l0, l1, l2, l3;
    unsigned D = (dim + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = t1;
    const float *e_l = l + DD;
    float unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm_load_ps(unpack);
    switch (DR) {
        case 12:
            SSE_L2SQR_1(e_l+8, sum, l2);
        case 8:
            SSE_L2SQR_1(e_l+4, sum, l1);
        case 4:
            SSE_L2SQR_1(e_l, sum, l0);
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16) {
        SSE_L2SQR_1(l, sum, l0);
        SSE_L2SQR_1(l + 4, sum, l1);
        SSE_L2SQR_1(l + 8, sum, l2);
        SSE_L2SQR_1(l + 12, sum, l3);
    }
    _mm_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
    return ret;//sqrt(ret);
}
}
/*
template <typename T>
void print_128 (__m128i v) {
    static unsigned constexpr L = 16 / sizeof(T);
    T unpack[L] __attribute__ ((aligned (16)));
    _mm_store_si128((__m128i *)unpack, v);
    cout << '(' << int(unpack[0]);
    for (unsigned i = 1; i < L; ++i) {
        cout << ',' << int(unpack[i]);
    }
    cout << ')';
}
*/

#define SSE_L2SQR_BYTE(addr1, addr2, sum, z) \
    do { \
        const __m128i o = _mm_load_si128((__m128i const *)(addr1));\
        const __m128i p = _mm_load_si128((__m128i const *)(addr2));\
        __m128i o1 = _mm_unpackhi_epi8(o,z); \
        __m128i p1 = _mm_unpackhi_epi8(p,z); \
        __m128i d = _mm_sub_epi16(o1, p1); \
        sum = _mm_add_epi32(sum, _mm_madd_epi16(d, d)); \
        o1 = _mm_unpacklo_epi8(o,z); \
        p1 = _mm_unpacklo_epi8(p,z); \
        d = _mm_sub_epi16(o1, p1); \
        sum = _mm_add_epi32(sum, _mm_madd_epi16(d, d)); \
    } while (false)
namespace kgraph {
float uint8_l2sqr_sse2 (uint8_t const *t1, uint8_t const *t2, unsigned dim) {
    unsigned D = (dim + 0xFU) & ~0xFU;   // actual dimension used in calculation, 0-padded
    unsigned DR = D % 64;           // process 32 dims per iteration
    unsigned DD = D - DR;
    const uint8_t *l = t1;
    const uint8_t *r = t2;
    const uint8_t *e_l = l + DD;
    const uint8_t *e_r = r + DD;
    int32_t unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};
    __m128i sum = _mm_load_si128((__m128i *)unpack);
    const __m128i z = sum;
    switch (DR) {
        case 48:
            SSE_L2SQR_BYTE(e_l+32, e_r+32, sum, z);
        case 32:
            SSE_L2SQR_BYTE(e_l+16, e_r+16, sum, z);
        case 16:
            SSE_L2SQR_BYTE(e_l, e_r, sum, z);
    }
    for (unsigned i = 0; i < DD; i += 64, l += 64, r += 64) {
        SSE_L2SQR_BYTE(l, r, sum, z);
        SSE_L2SQR_BYTE(l + 16, r + 16, sum, z);
        SSE_L2SQR_BYTE(l + 32, r + 32, sum, z);
        SSE_L2SQR_BYTE(l + 48, r + 48, sum, z);
    }
    _mm_store_si128((__m128i *)unpack, sum);
    int32_t ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
    return float(ret);//sqrt(ret);
}
}
#endif
#endif
