#pragma once

#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <algorithm>
#include <cmath>

#include "Macros.h"

#if OPT_EDGE_FUNCTION
#include "EdgeFunction.h"
#endif

// Simple support class for a 2D vector
class vec2D {
public:
    float x, y;

    // Default constructor initializes both components to 0
    vec2D() { x = y = 0.f; };

    // Constructor initializes components with given values
    vec2D(float _x, float _y) : x(_x), y(_y) {}

    // Constructor initializes components from a vec4
    vec2D(vec4 v) {
        x = v[0];
        y = v[1];
    }

    // Display the vector components
    void display() { std::cout << x << '\t' << y << std::endl; }

    // Overloaded subtraction operator for vector subtraction
    vec2D operator- (vec2D& v) {
        vec2D q;
        q.x = x - v.x;
        q.y = y - v.y;
        return q;
    }
};

// Class representing a triangle for rendering purposes
class triangle {
    Vertex v[3];       // Vertices of the triangle
    float area;        // Area of the triangle
    colour col[3];     // Colors for each vertex of the triangle

#if OPT_INV_AREA
	float invArea;     // Inverse of the triangle area for optimization
#endif

#if OPT_BACKFACE_CULLING
	float signedArea;  // Signed area for backface culling
#endif

public:
    // Constructor initializes the triangle with three vertices
    // Input Variables:
    // - v1, v2, v3: Vertices defining the triangle
    triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;

        // Calculate the 2D area of the triangle
        vec2D e1 = vec2D(v[1].p - v[0].p);
        vec2D e2 = vec2D(v[2].p - v[0].p);

#if OPT_BACKFACE_CULLING
        signedArea = (e1.x * e2.y - e1.y * e2.x);
        area = std::fabs(signedArea);
#else
        area = std::fabs(e1.x * e2.y - e1.y * e2.x);
#endif

#if OPT_INV_AREA
        invArea = (area > 0.f) ? (1.0f / area) : 0.f;
#endif
    }

    // Helper function to compute the cross product for barycentric coordinates
    // Input Variables:
    // - v1, v2: Edges defining the vector
    // - p: Point for which coordinates are being calculated
    float getC(vec2D v1, vec2D v2, vec2D p) {
        vec2D e = v2 - v1;
        vec2D q = p - v1;
        return q.y * e.x - q.x * e.y;
    }

    // Compute barycentric coordinates for a given point
    // Input Variables:
    // - p: Point to check within the triangle
    // Output Variables:
    // - alpha, beta, gamma: Barycentric coordinates of the point
    // Returns true if the point is inside the triangle, false otherwise
    bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma) {
#if OPT_INV_AREA
        alpha = getC(vec2D(v[1].p), vec2D(v[2].p), p) * invArea;
        beta = getC(vec2D(v[2].p), vec2D(v[0].p), p) * invArea;
        gamma = getC(vec2D(v[0].p), vec2D(v[1].p), p) * invArea;
#else
		alpha = getC(vec2D(v[1].p), vec2D(v[2].p), p) / area;
        beta = getC(vec2D(v[2].p), vec2D(v[0].p), p) / area;
		gamma = getC(vec2D(v[0].p), vec2D(v[1].p), p) / area;
#endif

        if (alpha < 0.f || beta < 0.f || gamma < 0.f) return false;
        return true;
    }

    // Template function to interpolate values using barycentric coordinates
    // Input Variables:
    // - alpha, beta, gamma: Barycentric coordinates
    // - a1, a2, a3: Values to interpolate
    // Returns the interpolated value
    template <typename T>
    T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3) {
        return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
    }

    // Draw the triangle on the canvas
    // Input Variables:
    // - renderer: Renderer object for drawing
    // - L: Light object for shading calculations
    // - ka, kd: Ambient and diffuse lighting coefficients
#if OPT_EDGE_FUNCTION && OPT_AVX_SIMD
	// Edge function optimized drawing using AVX SIMD
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);
        if (area < 1.f) return;

        #if OPT_BACKFACE_CULLING
        if (signedArea <= 0.f) return;
        #endif

        const int minY = (int)(minV.y);
        const int maxY = (int)ceil(maxV.y);
        const int minX = (int)(minV.x);
        const int maxX = (int)ceil(maxV.x);

#if !OPT_LIGHT_PRENORMALIZE
        L.omega_i.normalise();
#endif

        const __m256 zero = _mm256_setzero_ps();
        const __m256 one = _mm256_set1_ps(1.0f);
#if OPT_INV_AREA
        const __m256 invAreaVec = _mm256_set1_ps(invArea);
#else
        const __m256 invAreaVec = _mm256_set1_ps(1.0f / area);
#endif

        const __m256 kdVec = _mm256_set1_ps(kd);
        const __m256 kaVec = _mm256_set1_ps(ka);

        const __m256 lightX = _mm256_set1_ps(L.omega_i[0]);
        const __m256 lightY = _mm256_set1_ps(L.omega_i[1]);
        const __m256 lightZ = _mm256_set1_ps(L.omega_i[2]);

        const __m256 lightR = _mm256_set1_ps(L.L[colour::RED]);
        const __m256 lightG = _mm256_set1_ps(L.L[colour::GREEN]);
        const __m256 lightB = _mm256_set1_ps(L.L[colour::BLUE]);

        const __m256 ambientR = _mm256_mul_ps(_mm256_set1_ps(L.ambient[colour::RED]), kaVec);
        const __m256 ambientG = _mm256_mul_ps(_mm256_set1_ps(L.ambient[colour::GREEN]), kaVec);
        const __m256 ambientB = _mm256_mul_ps(_mm256_set1_ps(L.ambient[colour::BLUE]), kaVec);

        const __m256 v0z = _mm256_set1_ps(v[0].p[2]);
        const __m256 v1z = _mm256_set1_ps(v[1].p[2]);
        const __m256 v2z = _mm256_set1_ps(v[2].p[2]);

        const __m256 n0x = _mm256_set1_ps(v[0].normal[0]);
        const __m256 n0y = _mm256_set1_ps(v[0].normal[1]);
        const __m256 n0z = _mm256_set1_ps(v[0].normal[2]);
        const __m256 n1x = _mm256_set1_ps(v[1].normal[0]);
        const __m256 n1y = _mm256_set1_ps(v[1].normal[1]);
        const __m256 n1z = _mm256_set1_ps(v[1].normal[2]);
        const __m256 n2x = _mm256_set1_ps(v[2].normal[0]);
        const __m256 n2y = _mm256_set1_ps(v[2].normal[1]);
        const __m256 n2z = _mm256_set1_ps(v[2].normal[2]);

        const __m256 c0r = _mm256_set1_ps(v[0].rgb[colour::RED]);
        const __m256 c0g = _mm256_set1_ps(v[0].rgb[colour::GREEN]);
        const __m256 c0b = _mm256_set1_ps(v[0].rgb[colour::BLUE]);
        const __m256 c1r = _mm256_set1_ps(v[1].rgb[colour::RED]);
        const __m256 c1g = _mm256_set1_ps(v[1].rgb[colour::GREEN]);
        const __m256 c1b = _mm256_set1_ps(v[1].rgb[colour::BLUE]);
        const __m256 c2r = _mm256_set1_ps(v[2].rgb[colour::RED]);
        const __m256 c2g = _mm256_set1_ps(v[2].rgb[colour::GREEN]);
        const __m256 c2b = _mm256_set1_ps(v[2].rgb[colour::BLUE]);

        TriangleEdgeFunctions edges(v[0].p, v[1].p, v[2].p);
        edges.beginRow(minX, minY);

        for (int y = minY; y < maxY; y++) {
            __m256 e0, e1, e2;
            edges.getRowStart(e0, e1, e2);

            for (int x = minX; x < maxX; x += 8) {
                const int mask = edges.insideMask(e0, e1, e2);
                if (mask) {

                    const __m256 alpha = _mm256_mul_ps(e0, invAreaVec);
                    const __m256 beta = _mm256_mul_ps(e1, invAreaVec);
                    const __m256 gamma = _mm256_mul_ps(e2, invAreaVec);

                    const __m256 depth = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(v0z, alpha), _mm256_mul_ps(v1z, beta)),
                        _mm256_mul_ps(v2z, gamma));

                    __m256 nx = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(n0x, alpha), _mm256_mul_ps(n1x, beta)),
                        _mm256_mul_ps(n2x, gamma));
                    __m256 ny = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(n0y, alpha), _mm256_mul_ps(n1y, beta)),
                        _mm256_mul_ps(n2y, gamma));
                    __m256 nz = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(n0z, alpha), _mm256_mul_ps(n1z, beta)),
                        _mm256_mul_ps(n2z, gamma));

                    const __m256 length = _mm256_sqrt_ps(
                        _mm256_add_ps(
                            _mm256_mul_ps(nx, nx),
                            _mm256_add_ps(_mm256_mul_ps(ny, ny), _mm256_mul_ps(nz, nz))));
                    const __m256 invLength = _mm256_div_ps(one, length);
                    nx = _mm256_mul_ps(nx, invLength);
                    ny = _mm256_mul_ps(ny, invLength);
                    nz = _mm256_mul_ps(nz, invLength);

                    __m256 dot = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(lightX, nx), _mm256_mul_ps(lightY, ny)),
                        _mm256_mul_ps(lightZ, nz));
                    dot = _mm256_max_ps(dot, zero);

                    const __m256 cR = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(c0r, alpha), _mm256_mul_ps(c1r, beta)),
                        _mm256_mul_ps(c2r, gamma));
                    const __m256 cG = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(c0g, alpha), _mm256_mul_ps(c1g, beta)),
                        _mm256_mul_ps(c2g, gamma));
                    const __m256 cB = _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(c0b, alpha), _mm256_mul_ps(c1b, beta)),
                        _mm256_mul_ps(c2b, gamma));

                    const __m256 litR = _mm256_mul_ps(_mm256_mul_ps(cR, kdVec), _mm256_mul_ps(lightR, dot));
                    const __m256 litG = _mm256_mul_ps(_mm256_mul_ps(cG, kdVec), _mm256_mul_ps(lightG, dot));
                    const __m256 litB = _mm256_mul_ps(_mm256_mul_ps(cB, kdVec), _mm256_mul_ps(lightB, dot));

                    const __m256 outR = _mm256_min_ps(_mm256_add_ps(litR, ambientR), one);
                    const __m256 outG = _mm256_min_ps(_mm256_add_ps(litG, ambientG), one);
                    const __m256 outB = _mm256_min_ps(_mm256_add_ps(litB, ambientB), one);

                    alignas(32) float depthArr[8];
                    alignas(32) float outRArr[8];
                    alignas(32) float outGArr[8];
                    alignas(32) float outBArr[8];
                    _mm256_store_ps(depthArr, depth);
                    _mm256_store_ps(outRArr, outR);
                    _mm256_store_ps(outGArr, outG);
                    _mm256_store_ps(outBArr, outB);

                    for (int i = 0; i < 8; i++) {
                        if ((mask & (1 << i)) == 0) continue;
                        const int px = x + i;
                        if (px >= maxX) continue;

                        const float d = depthArr[i];

                        #if OPT_EARLY_Z_TEST
                        if (renderer.zbuffer(px, y) <= d || d <= 0.001f) continue;
                        #endif

                        if (renderer.zbuffer(px, y) > d && d > 0.001f) {
                            unsigned char r = static_cast<unsigned char>(std::floor(outRArr[i] * 255.0f));
                            unsigned char g = static_cast<unsigned char>(std::floor(outGArr[i] * 255.0f));
                            unsigned char b = static_cast<unsigned char>(std::floor(outBArr[i] * 255.0f));
                            renderer.canvas.draw(px, y, r, g, b);
                            renderer.zbuffer(px, y) = d;
                        }
                    }
                }
                edges.step8Pixels(e0, e1, e2);
            }
            edges.stepRow();
        }
    }

#elif OPT_EDGE_FUNCTION && !OPT_AVX_SIMD
	// Edge function optimized drawing without AVX SIMD
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;

        // Get the screen-space bounds of the triangle
        getBoundsWindow(renderer.canvas, minV, maxV);

        // Skip very small triangles
        if (area < 1.f) return;

        #if OPT_BACKFACE_CULLING
        if (signedArea <= 0.f) return;
        #endif

        const int minY = (int)(minV.y);
        const int maxY = (int)ceil(maxV.y);
        const int minX = (int)(minV.x);
        const int maxX = (int)ceil(maxV.x);

        TriangleEdgeFunctions edges(v[0].p, v[1].p, v[2].p);
        edges.beginRow(minX, minY);

        for (int y = minY; y < maxY; y++) {
            float e0, e1, e2;
            edges.getRowStart(e0, e1, e2);

            for (int x = minX; x < maxX; x++) {
                // weights
                float alpha, beta, gamma;
                if (e0 >= 0.f && e1 >= 0.f && e2 >= 0.f) {
                    #if OPT_INV_AREA
                    alpha = e0 * invArea;
                    beta = e1 * invArea;
                    gamma = e2 * invArea;
                    #else
                    alpha = e0 / area;
                    beta = e1 / area;
                    gamma = e2 / area;
                    #endif


                    #if OPT_EARLY_Z_TEST
                    float depth = interpolate(alpha, beta, gamma, v[0].p[2], v[1].p[2], v[2].p[2]);
                    if (renderer.zbuffer(x, y) <= depth || depth <= 0.001f) continue;

                    // Interpolate color, depth, and normals
                    colour c = interpolate(alpha, beta, gamma, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();
                    #else

                    // Interpolate color, depth, and normals
                    colour c = interpolate(alpha, beta, gamma, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();
                    float depth = interpolate(alpha, beta, gamma, v[0].p[2], v[1].p[2], v[2].p[2]);
                    #endif

                    vec4 normal = interpolate(alpha, beta, gamma, v[0].normal, v[1].normal, v[2].normal);
                    normal.normalise();

                    // Perform Z-buffer test and apply shading
                    if (renderer.zbuffer(x, y) > depth && depth > 0.001f) {
                        // typical shader begin
                        
                        #if !OPT_LIGHT_PRENORMALIZE
                        L.omega_i.normalise();
                        #endif

                        float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka); // using kd instead of ka for ambient
                        // typical shader end
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
                edges.stepPixel(e0, e1, e2);
            }
            edges.stepRow();
        }
    }
#else
	// Standard drawing method without edge function optimization
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;

        // Get the screen-space bounds of the triangle
        getBoundsWindow(renderer.canvas, minV, maxV);

        // Skip very small triangles
        if (area < 1.f) return;

        #if OPT_BACKFACE_CULLING
        if (signedArea <= 0.f) return;
        #endif

        // Iterate over the bounding box and check each pixel
        for (int y = (int)(minV.y); y < (int)ceil(maxV.y); y++) {
            for (int x = (int)(minV.x); x < (int)ceil(maxV.x); x++) {
                float alpha, beta, gamma;
                // Check if the pixel lies inside the triangle
                if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma)) {

                    #if OPT_EARLY_Z_TEST
                    float depth = interpolate(alpha, beta, gamma, v[0].p[2], v[1].p[2], v[2].p[2]);
                    if (renderer.zbuffer(x, y) <= depth || depth <= 0.001f) continue;

                    // Interpolate color, depth, and normals
                    colour c = interpolate(alpha, beta, gamma, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();
                    #else

                    // Interpolate color, depth, and normals
                    colour c = interpolate(alpha, beta, gamma, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();
                    float depth = interpolate(alpha, beta, gamma, v[0].p[2], v[1].p[2], v[2].p[2]);
                    #endif

                    vec4 normal = interpolate(alpha, beta, gamma, v[0].normal, v[1].normal, v[2].normal);
                    normal.normalise();

                    // Perform Z-buffer test and apply shading
                    if (renderer.zbuffer(x, y) > depth && depth > 0.001f) {
                        // typical shader begin

                        #if !OPT_LIGHT_PRENORMALIZE
                        L.omega_i.normalise();
                        #endif

                        float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka); // using kd instead of ka for ambient
                        // typical shader end
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
            }
        }
    }
#endif

    // Compute the 2D bounds of the triangle
    // Output Variables:
    // - minV, maxV: Minimum and maximum bounds in 2D space
    void getBounds(vec2D& minV, vec2D& maxV) {
        minV = vec2D(v[0].p);
        maxV = vec2D(v[0].p);
        for (unsigned int i = 1; i < 3; i++) {
            minV.x = std::min(minV.x, v[i].p[0]);
            minV.y = std::min(minV.y, v[i].p[1]);
            maxV.x = std::max(maxV.x, v[i].p[0]);
            maxV.y = std::max(maxV.y, v[i].p[1]);
        }
    }

    // Compute the 2D bounds of the triangle, clipped to the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    // Output Variables:
    // - minV, maxV: Clipped minimum and maximum bounds
    void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D& minV, vec2D& maxV) {
        getBounds(minV, maxV);
        minV.x = std::max(minV.x, static_cast<float>(0));
        minV.y = std::max(minV.y, static_cast<float>(0));
        maxV.x = std::min(maxV.x, static_cast<float>(canvas.getWidth()));
        maxV.y = std::min(maxV.y, static_cast<float>(canvas.getHeight()));
    }

    // Debugging utility to display the triangle bounds on the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    void drawBounds(GamesEngineeringBase::Window& canvas) {
        vec2D minV, maxV;
        getBounds(minV, maxV);

        for (int y = (int)minV.y; y < (int)maxV.y; y++) {
            for (int x = (int)minV.x; x < (int)maxV.x; x++) {
                canvas.draw(x, y, 255, 0, 0);
            }
        }
    }

    // Debugging utility to display the coordinates of the triangle vertices
    void display() {
        for (unsigned int i = 0; i < 3; i++) {
            v[i].p.display();
        }
        std::cout << std::endl;
    }
};
