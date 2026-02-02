#pragma once
#include <vector>
#include <immintrin.h>

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
}
