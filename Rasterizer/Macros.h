#pragma once

// ===== Transform Optimisations =====
#define OPT_VERTEX_CACHE			1	// Cache transformed vertices to avoid redundant calculations

// ===== Pipeline Optimisations =====
#define OPT_LIGHT_PRENORMALIZE		1	// Pre-normalize light direction before per-pixel lighting calculations
#define OPT_EARLY_Z_TEST			1	// Enable early Z-test to discard occluded fragments before shading
#define OPT_BACKFACE_CULLING        1	// Enable backface culling to skip rendering of back-facing triangles
#define OPT_INV_AREA				1	// Use inverse area for barycentric coordinate calculations
#define OPT_EDGE_FUNCTION			1	// Use edge function for point-in-triangle tests

// ===== Compilation Optimisations =====
#define OPT_AVX_SIMD				0	// Enable AVX SIMD optimizations

// ===== Multithreading Optimisations =====