#include <cmath>
#include <iostream>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <vector>


float avx2_l2_distance(const float* a, const float* b, unsigned dim) {
    __m256 sum = _mm256_setzero_ps(); // Initialize sum to 0
    unsigned i;
    // dim = 104;
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
    // for (; i < dim; ++i) { // Process remaining floats
    //     float diff = a[i] - b[i];
    //     result += diff * diff;
    // }
    return result; // Return square root of sum
}

    float avx512_l2_distance_opt(float const * a, float const * b, unsigned n) {
    const uint32_t kFloatsPerVec = 16;
    __m512 sum1 = _mm512_setzero_ps();
    // n = 112;
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
        // result += ((float*)&sum12)[j];
        result += ((float*)&sum1)[j];
    }
    return result;
}

float normal_l2(float const * a, float const * b, unsigned dim) {
                    float r = 0;
            for (unsigned i = 0; i < dim; ++i) {
                float v = float(a[i]) - float(a[i]);
                v *= v;
                r += v;
            }
            return r;
}

int main(int argc, char** argv) {
    unsigned long N = 128;

    auto *a = static_cast<float*>(aligned_alloc(512, N*sizeof(float)));
    auto *b = static_cast<float*>(aligned_alloc(512, N*sizeof(float)));
    auto *c = static_cast<float*>(aligned_alloc(512, N*sizeof(float)));

    std::chrono::time_point<std::chrono::system_clock> start, end;
    for (unsigned long i = 0; i < N; ++i) {                                                                                                                                                                                   
        a[i] = 3141592.65358;           
        b[i] = 1234567.65358;                                                                                                                                                                            
    }
    std::vector<float*> va(100000);
    std::vector<float*> vb(100000);
    for(int i = 0; i < va.size(); ++i) {
        va[i] = a;
        vb[i] = b;
    }

    start = std::chrono::system_clock::now();
    for (int i = 0; i < 100000; i++)                                                                                                                                                                                                                                                                                                                                                                                              
        normal_l2(va[i], vb[i], N);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "normal l2 elapsed time: " << elapsed_seconds.count() << std::endl;   
    
    start = std::chrono::system_clock::now();
    for (int i = 0; i < 100000; i++)                                                                                                                                                                                                                                                                                                                                                                                              
        avx2_l2_distance(va[i], vb[i], N);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "avx2 elapsed time: " << elapsed_seconds.count()<< std::endl;   

    start = std::chrono::system_clock::now();
    for (int i = 0; i < 100000; i++)                                                                                                                                                                                                                                                                                                                                                                                              
        avx512_l2_distance_opt(va[i], vb[i], N);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "avx512 elapsed time: " << elapsed_seconds.count() << std::endl;   
    return 0;            
}