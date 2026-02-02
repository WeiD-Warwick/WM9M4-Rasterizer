#pragma once

#include "Macros.h"

#if OPT_AVX_SIMD

#include <immintrin.h>

struct EdgeFunction {
    __m256 a, b, c;

    EdgeFunction(float x1, float y1, float x2, float y2) {
        float fa = y1 - y2;
        float fb = x2 - x1;
        float fc = x1 * y2 - x2 * y1;
        a = _mm256_set1_ps(fa);
        b = _mm256_set1_ps(fb);
        c = _mm256_set1_ps(fc);
    }

    inline __m256 evaluate(__m256 vx, __m256 vy) const {
        __m256 ax = _mm256_mul_ps(a, vx);
        __m256 by = _mm256_mul_ps(b, vy);
        return _mm256_add_ps(_mm256_add_ps(ax, by), c);
    }

    inline __m256 stepX() const { return a; }
    inline __m256 stepY() const { return b; }
};

struct TriangleEdgeFunctions {
    EdgeFunction ef0; // (v1->v2)
    EdgeFunction ef1; // (v2->v0)
    EdgeFunction ef2; // (v0->v1)

    __m256 rowE0 = _mm256_setzero_ps();
    __m256 rowE1 = _mm256_setzero_ps();
    __m256 rowE2 = _mm256_setzero_ps();

    __m256 rowX = _mm256_setzero_ps();

    TriangleEdgeFunctions(const vec4& v0, const vec4& v1, const vec4& v2)
        : ef0(v1[0], v1[1], v2[0], v2[1])
        , ef1(v2[0], v2[1], v0[0], v0[1])
        , ef2(v0[0], v0[1], v1[0], v1[1]) {
    }


    inline void beginRow(int minX, int minY) {
        const __m256 vx0 = _mm256_set1_ps((float)minX + 0.5f);
        const __m256 vy = _mm256_set1_ps((float)minY + 0.5f);

        rowX = _mm256_add_ps(vx0, _mm256_setr_ps(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f));

        rowE0 = ef0.evaluate(rowX, vy);
        rowE1 = ef1.evaluate(rowX, vy);
        rowE2 = ef2.evaluate(rowX, vy);
    }

    inline void stepRow() {
        rowE0 = _mm256_add_ps(rowE0, ef0.stepY());
        rowE1 = _mm256_add_ps(rowE1, ef1.stepY());
        rowE2 = _mm256_add_ps(rowE2, ef2.stepY());
    }

    inline void getRowStart(__m256& e0, __m256& e1, __m256& e2) const {
        e0 = rowE0;
        e1 = rowE1;
        e2 = rowE2;
    }

    inline void step8Pixels(__m256& e0, __m256& e1, __m256& e2) const {
        const __m256 eight = _mm256_set1_ps(8.0f);

        e0 = _mm256_add_ps(e0, _mm256_mul_ps(ef0.stepX(), eight));
        e1 = _mm256_add_ps(e1, _mm256_mul_ps(ef1.stepX(), eight));
        e2 = _mm256_add_ps(e2, _mm256_mul_ps(ef2.stepX(), eight));
    }

    inline void stepRowX8() {
        rowX = _mm256_add_ps(rowX, _mm256_set1_ps(8.0f));
    }

    inline int insideMask(__m256 e0, __m256 e1, __m256 e2) const {
        const __m256 zero = _mm256_setzero_ps();
        const __m256 m0 = _mm256_cmp_ps(e0, zero, _CMP_GE_OQ);
        const __m256 m1 = _mm256_cmp_ps(e1, zero, _CMP_GE_OQ);
        const __m256 m2 = _mm256_cmp_ps(e2, zero, _CMP_GE_OQ);
        const __m256 m = _mm256_and_ps(_mm256_and_ps(m0, m1), m2);
        return _mm256_movemask_ps(m);
    }
};

#else

struct EdgeFunction {
    float a, b, c;

    EdgeFunction(float x1, float y1, float x2, float y2) {
        a = y1 - y2;
        b = x2 - x1;
        c = x1 * y2 - x2 * y1;
    }

    inline float evaluate(float x, float y) const {
        return a * x + b * y + c;
    }

    inline float stepX() const { return a; }
    inline float stepY() const { return b; }
};

struct TriangleEdgeFunctions {
    EdgeFunction ef0; // (v1->v2)
    EdgeFunction ef1; // (v2->v0)
    EdgeFunction ef2; // (v0->v1)

    float rowE0 = 0.f;
    float rowE1 = 0.f;
    float rowE2 = 0.f;

    TriangleEdgeFunctions(const vec4& v0, const vec4& v1, const vec4& v2)
        : ef0(v1[0], v1[1], v2[0], v2[1]),
          ef1(v2[0], v2[1], v0[0], v0[1]),
          ef2(v0[0], v0[1], v1[0], v1[1]) {}

    inline void beginRow(int minX, int minY) {
        const float fx = minX + 0.5f;
        const float fy = minY + 0.5f;
        rowE0 = ef0.evaluate(fx, fy);
        rowE1 = ef1.evaluate(fx, fy);
        rowE2 = ef2.evaluate(fx, fy);
    }

    inline void stepRow() {
        rowE0 += ef0.b;
        rowE1 += ef1.b;
        rowE2 += ef2.b;
    }

    inline void getRowStart(float& e0, float& e1, float& e2) const {
        e0 = rowE0;
        e1 = rowE1;
        e2 = rowE2;
    }

    inline void stepPixel(float& e0, float& e1, float& e2) const {
        e0 += ef0.a;
        e1 += ef1.a;
        e2 += ef2.a;
    }
};

#endif