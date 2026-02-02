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
    EdgeFunction ef0;
    EdgeFunction ef1;
    EdgeFunction ef2;

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

    inline bool isInside(float e, const EdgeFunction& ef) {
        if (e > 0) return true;
        if (e < 0) return false;

        return (ef.a > 0) || (ef.a == 0 && ef.b < 0);
    }
};

#endif