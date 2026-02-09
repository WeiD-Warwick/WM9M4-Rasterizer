#pragma once

// ===== Transform Optimisations =====
#define OPT_VERTEX_CACHE			1	// Cache transformed vertices to avoid redundant calculations


// ===== Pipeline Optimisations =====
#define OPT_LIGHT_PRE_NORMALIZE		1	// Pre-normalize light direction before per-pixel lighting calculations
#define OPT_EARLY_Z_TEST			1	// Enable early Z-test to discard occluded fragments before shading
#define OPT_BACKFACE_CULLING        1	// Enable backface culling to skip rendering of back-facing triangles
#define OPT_INV_AREA				1	// Use inverse area for barycentric coordinate calculations
#define OPT_EDGE_FUNCTION			1	// Use edge function for point-in-triangle tests

// ===== Compilation Optimisations =====
#define OPT_AVX_SIMD				1	// Enable AVX SIMD optimizations

// ===== Multithreading Optimisations =====
#define OPT_MULTITHREAD				1	// Enable multithreading support

// ===== For Scene 3 =====
#define OPT_RENDER_SCENE			1
#define OPT_FRUSTUM_CULLING			1

// ===== Variable Definitions =====
// 1024 768
// 
// Scene 1
// thread 1(1024 768)  1071
// thread 2(512  768)  1330
// thread 4(512  384)  1527
// thread 6(512  256)  1321
// thread 8(256 384)   1191
// 
// Scene 2
// thread 1(1024 768)  580
// thread 2 (512 768)  835
// thread 4 (512 384)  1075
// thread 6 (512 256)  1064
// thread 8 (256 384)  1013
// thread 12(256 256)  1030
//
// Scene 3
// thread 1(1024 768)  257
// thread 2 (512 768)  371
// thread 4 (512 384)  438
// thread 6 (512 256)  359
// thread 8 (256 384)  302
// thread 12(256 256)  176

#define MT_TILE_W 512
#define MT_TILE_H 384
#define THREAD_COUNT 8
#define MT_MAX_THREADS				22
#define TARGET_TOTAL_LOOPS			25000
#define WARMUP_LOOPS				5000

#define SCENE_SELECT				3