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

struct ScissorRect {
    int minX, minY;
    int maxX, maxY;
};


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
#if OPT_EDGE_FUNCTION && OPT_AVX_SIMD && OPT_INV_AREA && OPT_BACKFACE_CULLING
	// Edge function optimized drawing using AVX SIMD
#if OPT_MULTITHREAD
    static void drawMT(Renderer& renderer, VertexSOA& cache, triIndices& ind, avx2::LightSIMD& lp, const ScissorRect& sc) {
#else
    static void draw(Renderer& renderer, VertexSOA& cache, triIndices& ind, avx2::LightSIMD& lp) {
#endif
        // load coords
        float x0 = cache.p.x[ind.v[0]];
        float y0 = cache.p.y[ind.v[0]];
        float x1 = cache.p.x[ind.v[1]];
        float y1 = cache.p.y[ind.v[1]];
        float x2 = cache.p.x[ind.v[2]];
        float y2 = cache.p.y[ind.v[2]];

        // backface culling
        float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        if (area <= 0.f) return;
        float invArea = 1.0f / area;

        const int W = (int)renderer.canvas.getWidth();
        const int H = (int)renderer.canvas.getHeight();

        // calculate bbox
        int triMinX = (int)std::floor(std::min({ x0, x1, x2 }));
        int triMaxX = (int)std::ceil(std::max({ x0, x1, x2 }));
        int triMinY = (int)std::floor(std::min({ y0, y1, y2 }));
        int triMaxY = (int)std::ceil(std::max({ y0, y1, y2 }));

        // clamp to screen
        int minX = std::max(0, triMinX);
        int minY = std::max(0, triMinY);
        int maxX = std::min(W, triMaxX);
        int maxY = std::min(H, triMaxY);

#if OPT_MULTITHREAD
        minX = std::max(minX, sc.minX);
        minY = std::max(minY, sc.minY);
        maxX = std::min(maxX, sc.maxX);
        maxY = std::min(maxY, sc.maxY);
#endif

        if (minX >= maxX || minY >= maxY) return;

        // Triangle SIMD Context
        avx2::TriContext context;
        context.load(cache, ind.v[0], ind.v[1], ind.v[2], invArea);

        TriangleEdgeFunctions edges(vec4(x0, y0, 0, 1), vec4(x1, y1, 0, 1), vec4(x2, y2, 0, 1));
        edges.beginRow(minX, minY);

        alignas(32) float dArr[8], rArr[8], gArr[8], bArr[8];

        for (int y = minY; y < maxY; ++y) {
            __m256 e0, e1, e2;
            edges.getRowStart(e0, e1, e2);
            for (int x = minX; x < maxX; x += 8) {
                int mask = edges.insideMask(e0, e1, e2);
                if (mask) {
                    avx2::shade_8_pixels(e0, e1, e2, context, lp, dArr, rArr, gArr, bArr);
                    for (int i = 0; i < 8; ++i) {
                        int px = x + i;
                        if ((mask & (1 << i)) && px < maxX) {
                            float d = dArr[i];
                            if (d > 0.001f && d < renderer.zbuffer(px, y)) {
                                renderer.canvas.draw(px, y,
                                    (unsigned char)(rArr[i] * 255.0f),
                                    (unsigned char)(gArr[i] * 255.0f),
                                    (unsigned char)(bArr[i] * 255.0f));
                                renderer.zbuffer(px, y) = d;
                            }
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
                        
                        #if !OPT_LIGHT_PRE_NORMALIZE
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

                        #if !OPT_LIGHT_PRE_NORMALIZE
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
