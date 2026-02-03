#pragma once
#include <vector>
#include <cmath>
#include <immintrin.h>

#include "renderer.h"
#include "light.h"
#include "colour.h"
#include "mesh.h"

#if OPT_EDGE_FUNCTION
#include "EdgeFunction.h"
#endif

struct Vec4SOA {
    std::vector<float> x, y, z, w;
    int size() const {
        return x.size();
    }

    void resize(int size) {
        x.resize(size);
        y.resize(size);
        z.resize(size);
        w.resize(size);
    }
};

struct ColorSOA {
    std::vector<float> r, g, b;
    int size() const {
        return r.size();
    }

    void resize(int size) {
        r.resize(size);
        g.resize(size);
        b.resize(size);
    }
};

struct VertexSOA {
    Vec4SOA p;     // position
    Vec4SOA n;     // normal
    ColorSOA c;    // color

    int size() const {
        return p.size();
    }

    void resize(int size) {
        p.resize(size);
        n.resize(size);
		c.resize(size);
    }
};

namespace avx2 {

    inline void mat4_mul_vec4(const float* m16, const Vec4SOA& in, int startIndex, int range, Vec4SOA& out) {

        const int end = startIndex + range;
        if (out.size() < end) out.resize(end);

        // Matrix rows
        __m256 m00 = _mm256_set1_ps(m16[0]);
        __m256 m01 = _mm256_set1_ps(m16[1]);
        __m256 m02 = _mm256_set1_ps(m16[2]);
        __m256 m03 = _mm256_set1_ps(m16[3]);

        __m256 m10 = _mm256_set1_ps(m16[4]);
        __m256 m11 = _mm256_set1_ps(m16[5]);
        __m256 m12 = _mm256_set1_ps(m16[6]);
        __m256 m13 = _mm256_set1_ps(m16[7]);

        __m256 m20 = _mm256_set1_ps(m16[8]);
        __m256 m21 = _mm256_set1_ps(m16[9]);
        __m256 m22 = _mm256_set1_ps(m16[10]);
        __m256 m23 = _mm256_set1_ps(m16[11]);

        __m256 m30 = _mm256_set1_ps(m16[12]);
        __m256 m31 = _mm256_set1_ps(m16[13]);
        __m256 m32 = _mm256_set1_ps(m16[14]);
        __m256 m33 = _mm256_set1_ps(m16[15]);

        int i = startIndex;
        for (; i + 8 <= end; i += 8) {
            __m256 vx = _mm256_loadu_ps(&in.x[i]);
            __m256 vy = _mm256_loadu_ps(&in.y[i]);
            __m256 vz = _mm256_loadu_ps(&in.z[i]);
            __m256 vw = _mm256_loadu_ps(&in.w[i]);

            // R0
            __m256 r0 = _mm256_add_ps(_mm256_mul_ps(m00, vx), _mm256_mul_ps(m01, vy));
            r0 = _mm256_add_ps(r0, _mm256_mul_ps(m02, vz));
            r0 = _mm256_add_ps(r0, _mm256_mul_ps(m03, vw));

            __m256 r1 = _mm256_add_ps(_mm256_mul_ps(m10, vx), _mm256_mul_ps(m11, vy));
            r1 = _mm256_add_ps(r1, _mm256_mul_ps(m12, vz));
            r1 = _mm256_add_ps(r1, _mm256_mul_ps(m13, vw));

            __m256 r2 = _mm256_add_ps(_mm256_mul_ps(m20, vx), _mm256_mul_ps(m21, vy));
            r2 = _mm256_add_ps(r2, _mm256_mul_ps(m22, vz));
            r2 = _mm256_add_ps(r2, _mm256_mul_ps(m23, vw));

            __m256 r3 = _mm256_add_ps(_mm256_mul_ps(m30, vx), _mm256_mul_ps(m31, vy));
            r3 = _mm256_add_ps(r3, _mm256_mul_ps(m32, vz));
            r3 = _mm256_add_ps(r3, _mm256_mul_ps(m33, vw));

            _mm256_storeu_ps(&out.x[i], r0);
            _mm256_storeu_ps(&out.y[i], r1);
            _mm256_storeu_ps(&out.z[i], r2);
            _mm256_storeu_ps(&out.w[i], r3);
        }

        for (; i < end; ++i) {
            float vx = in.x[i], vy = in.y[i], vz = in.z[i], vw = in.w[i];
            out.x[i] = m16[0] * vx + m16[1] * vy + m16[2] * vz + m16[3] * vw;
            out.y[i] = m16[4] * vx + m16[5] * vy + m16[6] * vz + m16[7] * vw;
            out.z[i] = m16[8] * vx + m16[9] * vy + m16[10] * vz + m16[11] * vw;
            out.w[i] = m16[12] * vx + m16[13] * vy + m16[14] * vz + m16[15] * vw;
        }
    }

    inline void divideW_and_ScreenMapping(
        Vec4SOA& p, int startIndex, int range, float halfW, float halfH, float H) {

        const int end = startIndex + range;

        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 vHalfW = _mm256_set1_ps(halfW);
        const __m256 vHalfH = _mm256_set1_ps(halfH);
        const __m256 vH = _mm256_set1_ps(H);

        int i = startIndex;
        for (; i + 8 <= end; i += 8) {
            __m256 x = _mm256_loadu_ps(&p.x[i]);
            __m256 y = _mm256_loadu_ps(&p.y[i]);
            __m256 z = _mm256_loadu_ps(&p.z[i]);
            __m256 w = _mm256_loadu_ps(&p.w[i]);

            // invW = 1/w
            __m256 invW = _mm256_div_ps(one, w);

            x = _mm256_mul_ps(x, invW);
            y = _mm256_mul_ps(y, invW);
            z = _mm256_mul_ps(z, invW);

            // screen mapping
            x = _mm256_mul_ps(_mm256_add_ps(x, one), vHalfW);
            y = _mm256_sub_ps(vH, _mm256_mul_ps(_mm256_add_ps(y, one), vHalfH));

            _mm256_storeu_ps(&p.x[i], x);
            _mm256_storeu_ps(&p.y[i], y);
            _mm256_storeu_ps(&p.z[i], z);
            _mm256_storeu_ps(&p.w[i], one); // w=1
        }

        for (; i < end; ++i) {
            float invW = 1.0f / p.w[i];
            float x = p.x[i] * invW;
            float y = p.y[i] * invW;
            float z = p.z[i] * invW;

            p.x[i] = (x + 1.0f) * halfW;
            p.y[i] = H - (y + 1.0f) * halfH;
            p.z[i] = z;
            p.w[i] = 1.0f;
        }
    }

    inline void normalize3(Vec4SOA& n, int startIndex, int range) {
        const int end = startIndex + range;

        const __m256 one = _mm256_set1_ps(1.0f);

        int i = startIndex;
        for (; i + 8 <= end; i += 8) {
            __m256 nx = _mm256_loadu_ps(&n.x[i]);
            __m256 ny = _mm256_loadu_ps(&n.y[i]);
            __m256 nz = _mm256_loadu_ps(&n.z[i]);

            __m256 length = _mm256_sqrt_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(nx, nx),
                    _mm256_add_ps(
                        _mm256_mul_ps(ny, ny),
                        _mm256_mul_ps(nz, nz))
            ));
            __m256 invLength = _mm256_div_ps(one, length);

            nx = _mm256_mul_ps(nx, invLength);
            ny = _mm256_mul_ps(ny, invLength);
            nz = _mm256_mul_ps(nz, invLength);

            _mm256_storeu_ps(&n.x[i], nx);
            _mm256_storeu_ps(&n.y[i], ny);
            _mm256_storeu_ps(&n.z[i], nz);
        }

        for (; i < end; ++i) {
            float x = n.x[i], y = n.y[i], z = n.z[i];
            float length = std::sqrt(x * x + y * y + z * z);
            float invLength = 1.0f / length;
            n.x[i] = x * invLength;
            n.y[i] = y * invLength;
            n.z[i] = z * invLength;
        }
    }

    struct LightSIMD {
        __m256 lX, lY, lZ;
        __m256 lR, lG, lB;
        __m256 ambR, ambG, ambB;
        __m256 kd, ka;

        LightSIMD(Light& light, float kd_, float ka_) {
            // light direction
            lX = _mm256_set1_ps(light.omega_i[0]);
            lY = _mm256_set1_ps(light.omega_i[1]);
            lZ = _mm256_set1_ps(light.omega_i[2]);

            // light colour
            lR = _mm256_set1_ps(light.L[colour::RED]);
            lG = _mm256_set1_ps(light.L[colour::GREEN]);
            lB = _mm256_set1_ps(light.L[colour::BLUE]);

            // ambient
            ambR = _mm256_set1_ps(light.ambient[colour::RED]);
            ambG = _mm256_set1_ps(light.ambient[colour::GREEN]);
            ambB = _mm256_set1_ps(light.ambient[colour::BLUE]);

            kd = _mm256_set1_ps(kd_);
            ka = _mm256_set1_ps(ka_);
        }
    };

    // Context for single triangle raster
    struct TriContext {
        __m256 v0z, v1z, v2z;
        __m256 n0x, n0y, n0z, n1x, n1y, n1z, n2x, n2y, n2z;
        __m256 r0, g0, b0, r1, g1, b1, r2, g2, b2;
        __m256 invArea;

        // load data from SOA
        void load(const VertexSOA& cache, unsigned int i0, unsigned int i1, unsigned int i2, float _invArea) {
            auto set_z = [&](unsigned int i) { return _mm256_set1_ps(cache.p.z[i]); };
            v0z = set_z(i0); v1z = set_z(i1); v2z = set_z(i2);

            n0x = _mm256_set1_ps(cache.n.x[i0]); n0y = _mm256_set1_ps(cache.n.y[i0]); n0z = _mm256_set1_ps(cache.n.z[i0]);
            n1x = _mm256_set1_ps(cache.n.x[i1]); n1y = _mm256_set1_ps(cache.n.y[i1]); n1z = _mm256_set1_ps(cache.n.z[i1]);
            n2x = _mm256_set1_ps(cache.n.x[i2]); n2y = _mm256_set1_ps(cache.n.y[i2]); n2z = _mm256_set1_ps(cache.n.z[i2]);

            r0 = _mm256_set1_ps(cache.c.r[i0]); g0 = _mm256_set1_ps(cache.c.g[i0]); b0 = _mm256_set1_ps(cache.c.b[i0]);
            r1 = _mm256_set1_ps(cache.c.r[i1]); g1 = _mm256_set1_ps(cache.c.g[i1]); b1 = _mm256_set1_ps(cache.c.b[i1]);
            r2 = _mm256_set1_ps(cache.c.r[i2]); g2 = _mm256_set1_ps(cache.c.g[i2]); b2 = _mm256_set1_ps(cache.c.b[i2]);

            invArea = _mm256_set1_ps(_invArea);
        }
    };

    // 8 bit SIMD Shader
    inline void shade_8_pixels(const __m256& e0, const __m256& e1, const __m256& e2,
        const TriContext& tc, const LightSIMD& lp,
        float* dOut, float* rOut, float* gOut, float* bOut) {
        __m256 alpha = _mm256_mul_ps(e0, tc.invArea);
        __m256 beta = _mm256_mul_ps(e1, tc.invArea);
        __m256 gamma = _mm256_mul_ps(e2, tc.invArea);

        // depth interpolate
        _mm256_store_ps(dOut, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(tc.v0z, alpha), _mm256_mul_ps(tc.v1z, beta)), _mm256_mul_ps(tc.v2z, gamma)));

        // normal interpolate and normalize
        __m256 nx = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(tc.n0x, alpha), _mm256_mul_ps(tc.n1x, beta)), _mm256_mul_ps(tc.n2x, gamma));
        __m256 ny = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(tc.n0y, alpha), _mm256_mul_ps(tc.n1y, beta)), _mm256_mul_ps(tc.n2y, gamma));
        __m256 nz = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(tc.n0z, alpha), _mm256_mul_ps(tc.n1z, beta)), _mm256_mul_ps(tc.n2z, gamma));

        __m256 invLen = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_add_ps(_mm256_mul_ps(ny, ny), _mm256_mul_ps(nz, nz)))));
        nx = _mm256_mul_ps(nx, invLen); ny = _mm256_mul_ps(ny, invLen); nz = _mm256_mul_ps(nz, invLen);

        // Dot(L, N)
        __m256 dot = _mm256_max_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(nx, lp.lX), _mm256_mul_ps(ny, lp.lY)), _mm256_mul_ps(nz, lp.lZ)), _mm256_setzero_ps());

        // calculate color
        auto calc = [&](const __m256& c0, const __m256& c1, const __m256& c2, const __m256& lCol, const __m256& amb) {
            __m256 color = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(c0, alpha), _mm256_mul_ps(c1, beta)), _mm256_mul_ps(c2, gamma));
            return _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(color, lp.kd), _mm256_mul_ps(lCol, dot)), amb), _mm256_set1_ps(1.0f));
            };

        _mm256_store_ps(rOut, calc(tc.r0, tc.r1, tc.r2, lp.lR, lp.ambR));
        _mm256_store_ps(gOut, calc(tc.g0, tc.g1, tc.g2, lp.lG, lp.ambG));
        _mm256_store_ps(bOut, calc(tc.b0, tc.b1, tc.b2, lp.lB, lp.ambB));
    }
}
