#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "matrix.h"
#include "colour.h"
#include "mesh.h"
#include "zbuffer.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"

#include "Macros.h"
#include "Profiler.h"
#include "ThreadPool.h"

#if OPT_MULTITHREAD

#if OPT_FRUSTUM_CULLING
float maxScaleFromMatrix(const matrix& m) {
    float sx = std::sqrt(m(0, 0) * m(0, 0) + m(0, 1) * m(0, 1) + m(0, 2) * m(0, 2));
    float sy = std::sqrt(m(1, 0) * m(1, 0) + m(1, 1) * m(1, 1) + m(1, 2) * m(1, 2));
    float sz = std::sqrt(m(2, 0) * m(2, 0) + m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2));
    return std::max({ sx, sy, sz });
}
#endif

struct AdaptiveMTConfig {
    int tileW = MT_TILE_W;
    int tileH = MT_TILE_H;
    int threadCount = THREAD_COUNT;
};

struct WorkloadStats {
    int totalTriangles = 0;
    int visibleTriangles = 0;
    float imbalance = 1.0f;
    float occupancy = 0.0f;
};

struct WorkloadAccumulator {
    int gridX = 6;
    int gridY = 4;
    int totalTriangles = 0;
    int visibleTriangles = 0;
    std::vector<int> bins;
};

struct AdaptiveHistory {
    bool initialized = false;
    float avgTriangles = 0.0f;
    float avgVisible = 0.0f;
    float avgImbalance = 1.0f;
    float avgOccupancy = 0.0f;
};

WorkloadAccumulator makeAccumulator(int gridX, int gridY) {
    WorkloadAccumulator acc;
    acc.gridX = gridX;
    acc.gridY = gridY;
    acc.bins.assign(gridX * gridY, 0);
    return acc;
}

void accumulateWorkload(const Renderer& renderer,
    const VertexSOA& vCache,
    const std::vector<triIndices>& triangles,
    WorkloadAccumulator& acc) {
    const int W = static_cast<int>(renderer.canvas.getWidth());
    const int H = static_cast<int>(renderer.canvas.getHeight());
    const float cellW = static_cast<float>(W) / static_cast<float>(acc.gridX);
    const float cellH = static_cast<float>(H) / static_cast<float>(acc.gridY);

    acc.totalTriangles += static_cast<int>(triangles.size());

    for (const auto& ind : triangles) {
        if (std::fabs(vCache.p.z[ind.v[0]]) > 1.0f || std::fabs(vCache.p.z[ind.v[1]]) > 1.0f || std::fabs(vCache.p.z[ind.v[2]]) > 1.0f) {
            continue;
        }

        const float cx = (vCache.p.x[ind.v[0]] + vCache.p.x[ind.v[1]] + vCache.p.x[ind.v[2]]) * (1.0f / 3.0f);
        const float cy = (vCache.p.y[ind.v[0]] + vCache.p.y[ind.v[1]] + vCache.p.y[ind.v[2]]) * (1.0f / 3.0f);
        const int gx = std::clamp(static_cast<int>(cx / cellW), 0, acc.gridX - 1);
        const int gy = std::clamp(static_cast<int>(cy / cellH), 0, acc.gridY - 1);

        acc.bins[gx + gy * acc.gridX] += 1;
        acc.visibleTriangles += 1;
    }
}

WorkloadStats finalizeWorkload(const WorkloadAccumulator& acc) {
    WorkloadStats stats;
    stats.totalTriangles = acc.totalTriangles;
    stats.visibleTriangles = acc.visibleTriangles;

    const int cellCount = acc.gridX * acc.gridY;
    if (cellCount == 0 || acc.visibleTriangles == 0) {
        stats.imbalance = 1.0f;
        stats.occupancy = 0.0f;
        return stats;
    }

    int maxCell = 0;
    int occupied = 0;
    for (int count : acc.bins) {
        if (count > 0) {
            occupied += 1;
            maxCell = std::max(maxCell, count);
        }
    }

    const float avg = static_cast<float>(acc.visibleTriangles) / static_cast<float>(cellCount);
    stats.imbalance = avg > 0.0f ? static_cast<float>(maxCell) / avg : 1.0f;
    stats.occupancy = static_cast<float>(occupied) / static_cast<float>(cellCount);
    return stats;
}

void updateHistory(AdaptiveHistory& history, const WorkloadStats& stats) {
    const float alpha = 0.2f;
    if (!history.initialized) {
        history.avgTriangles = static_cast<float>(stats.totalTriangles);
        history.avgVisible = static_cast<float>(stats.visibleTriangles);
        history.avgImbalance = stats.imbalance;
        history.avgOccupancy = stats.occupancy;
        history.initialized = true;
        return;
    }

    history.avgTriangles = history.avgTriangles * (1.0f - alpha) + static_cast<float>(stats.totalTriangles) * alpha;
    history.avgVisible = history.avgVisible * (1.0f - alpha) + static_cast<float>(stats.visibleTriangles) * alpha;
    history.avgImbalance = history.avgImbalance * (1.0f - alpha) + stats.imbalance * alpha;
    history.avgOccupancy = history.avgOccupancy * (1.0f - alpha) + stats.occupancy * alpha;
}

AdaptiveMTConfig chooseAdaptiveConfig(const Renderer& renderer, const AdaptiveHistory& history) {
    AdaptiveMTConfig config;
    const int W = static_cast<int>(renderer.canvas.getWidth());
    const int H = static_cast<int>(renderer.canvas.getHeight());

    const int hw = std::max(1u, std::thread::hardware_concurrency());
    const int maxThreads = std::min(static_cast<int>(hw), MT_MAX_THREADS);

    if (!history.initialized) {
        config.tileW = MT_TILE_W;
        config.tileH = MT_TILE_H;
        config.threadCount = std::min(THREAD_COUNT, maxThreads);
        return config;
    }

    float scale = 1.0f;
    if (history.avgVisible < 250.0f) {
        scale = 2.0f;
    }
    else if (history.avgVisible < 600.0f) {
        scale = 1.5f;
    }

    if (history.avgImbalance > 3.0f || history.avgOccupancy < 0.2f) {
        scale = 0.33f;
    }
    else if (history.avgImbalance > 2.0f || history.avgOccupancy < 0.35f) {
        scale = 0.5f;
    }

    const int minTileW = 128;
    const int minTileH = 128;
    config.tileW = std::clamp(static_cast<int>(std::round(MT_TILE_W * scale)), minTileW, W);
    config.tileH = std::clamp(static_cast<int>(std::round(MT_TILE_H * scale)), minTileH, H);

    int threads = maxThreads;
    if (history.avgVisible < 300.0f) {
        threads = 1;
    }
    else if (history.avgVisible < 1200.0f) {
        threads = std::max(1, maxThreads / 2);
    }

    const int tilesX = (W + config.tileW - 1) / config.tileW;
    const int tilesY = (H + config.tileH - 1) / config.tileH;
    const int tileCount = tilesX * tilesY;
    if (tileCount > 0) {
        threads = std::min(threads, tileCount);
    }
    config.threadCount = std::clamp(threads, 1, maxThreads);
    return config;
}

void renderMT(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L, ThreadPool& pool, const AdaptiveMTConfig& config, WorkloadAccumulator* acc) {

#if OPT_FRUSTUM_CULLING
    vec4 boundsCenter;
    float boundsRadius = 0.0f;
    mesh->getBoundsSphere(boundsCenter, boundsRadius);
    matrix view = camera * mesh->world;
    vec4 centerView = view * boundsCenter;
    float radiusView = boundsRadius * maxScaleFromMatrix(mesh->world);

    if (!renderer.sphereInFrustumView(centerView, radiusView)) {
        return;
    }

    matrix p = renderer.perspective * view;
#else
    matrix p = renderer.perspective * camera * mesh->world;
#endif

    const int W = (int)renderer.canvas.getWidth();
    const int H = (int)renderer.canvas.getHeight();

    auto vCache = std::make_shared<VertexSOA>();
    auto lp = std::make_shared<avx2::LightSIMD>(L, mesh->ka, mesh->kd);

    {
        Profiler::RegionTimer t("Vertex_Stage");
        mesh->preProcessVertexCache(p, (float)W, (float)H, *vCache);
    }

    if (acc) {
        accumulateWorkload(renderer, *vCache, mesh->triangles, *acc);
    }

    const int tilesX = (W + config.tileW - 1) / config.tileW;
    const int tilesY = (H + config.tileH - 1) / config.tileH;

    std::vector<ThreadPool::Job> batch;
    batch.reserve(tilesX * tilesY);

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            ScissorRect sc{
                tx * config.tileW,
                ty * config.tileH,
                std::min((tx + 1) * config.tileW, W),
                std::min((ty + 1) * config.tileH, H)
            };

            batch.emplace_back([&, sc, vCache, lp, mesh]() {
                for (auto& ind : mesh->triangles) {
                    if (std::fabs(vCache->p.z[ind.v[0]]) > 1.0f || std::fabs(vCache->p.z[ind.v[1]]) > 1.0f || std::fabs(vCache->p.z[ind.v[2]]) > 1.0f) {
                        continue;
                    }
                    triangle::drawMT(renderer, *vCache, ind, *lp, sc);
                }
            });
        }
    }

    pool.submitBatch(batch);
}

struct MeshRenderData {
    Mesh* mesh = nullptr;
    std::shared_ptr<VertexSOA> vCache;
    std::shared_ptr<avx2::LightSIMD> lp;
};

void renderSceneMT(Renderer& renderer, const std::vector<Mesh*>& scene, matrix& camera, Light& L, ThreadPool& pool, const AdaptiveMTConfig& config, WorkloadAccumulator* acc) {
    const int W = static_cast<int>(renderer.canvas.getWidth());
    const int H = static_cast<int>(renderer.canvas.getHeight());

    auto renderData = std::make_shared<std::vector<MeshRenderData>>();
    renderData->reserve(scene.size());

    for (auto* mesh : scene) {

#if OPT_FRUSTUM_CULLING
        vec4 boundsCenter;
        float boundsRadius = 0.0f;
        mesh->getBoundsSphere(boundsCenter, boundsRadius);
        matrix view = camera * mesh->world;
        vec4 centerView = view * boundsCenter;
        float radiusView = boundsRadius * maxScaleFromMatrix(mesh->world);

        if (!renderer.sphereInFrustumView(centerView, radiusView)) {
            continue;
        }
        matrix p = renderer.perspective * view;
#else
        matrix p = renderer.perspective * camera * mesh->world;
#endif

        MeshRenderData data;
        data.mesh = mesh;
        data.vCache = std::make_shared<VertexSOA>();
        data.lp = std::make_shared<avx2::LightSIMD>(L, mesh->ka, mesh->kd);

        mesh->preProcessVertexCache(p, static_cast<float>(W), static_cast<float>(H), *data.vCache);

        if (acc) {
            accumulateWorkload(renderer, *data.vCache, mesh->triangles, *acc);
        }

        renderData->push_back(std::move(data));
    }

    const int tilesX = (W + config.tileW - 1) / config.tileW;
    const int tilesY = (H + config.tileH - 1) / config.tileH;

    std::vector<ThreadPool::Job> batch;
    batch.reserve(tilesX * tilesY);

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            ScissorRect sc{
                tx * config.tileW,
                ty * config.tileH,
                std::min((tx + 1) * config.tileW, W),
                std::min((ty + 1) * config.tileH, H)
            };

            batch.emplace_back([&, sc, renderData]() {
                for (const auto& data : *renderData) {
                    for (auto& ind : data.mesh->triangles) {
                        if (std::fabs(data.vCache->p.z[ind.v[0]]) > 1.0f || std::fabs(data.vCache->p.z[ind.v[1]]) > 1.0f || std::fabs(data.vCache->p.z[ind.v[2]]) > 1.0f) {
                            continue;
                        }
                        triangle::drawMT(renderer, *data.vCache, ind, *data.lp, sc);
                    }
                }
                });
        }
    }

    pool.submitBatch(batch);
}

#else

// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.
void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;
    float width = renderer.canvas.getWidth();
    float height = renderer.canvas.getHeight();
#if OPT_VERTEX_CACHE && OPT_AVX_SIMD
    VertexSOA vCache;
    avx2::LightSIMD lp(L, mesh->ka, mesh->kd);

    mesh->preProcessVertexCache(p, width, height, vCache);

    for (auto& ind : mesh->triangles) {

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(vCache.p.z[ind.v[0]]) > 1.0f || fabs(vCache.p.z[ind.v[1]]) > 1.0f || fabs(vCache.p.z[ind.v[2]]) > 1.0f) {
            continue;
        }

        triangle::draw(renderer, vCache, ind, lp);
    }
#elif OPT_VERTEX_CACHE && !OPT_AVX_SIMD
    std::vector<Vertex> vcache;
    mesh->preProcessVertexCache(p, width, height, vcache);
    // Iterate through all triangles in the mesh using cached vertices
    for (triIndices& ind : mesh->triangles) {
        Vertex t[3];
        t[0] = vcache[ind.v[0]];
        t[1] = vcache[ind.v[1]];
        t[2] = vcache[ind.v[2]];

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        triangle tri(t[0], t[1], t[2]);
        tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
#else
    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh->triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh->vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
#endif
}

#endif

// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest() {
    Renderer renderer;
    // create light source {direction, diffuse intensity, ambient intensity}
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    #if OPT_LIGHT_PRE_NORMALIZE
    L.omega_i.normalise();
    #endif

#if OPT_MULTITHREAD
    std::unique_ptr<ThreadPool> pool;
    int activeThreads = 0;
    AdaptiveHistory history;
#endif

    // camera is just a matrix
    matrix camera = matrix::makeIdentity(); // Initialize the camera with identity matrix

    bool running = true; // Main loop control variable

    std::vector<Mesh*> scene; // Vector to store scene objects

    // Create a sphere and a rectangle mesh
    Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
    //Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

    // add meshes to scene
    scene.push_back(&mesh);
   // scene.push_back(&mesh2); 

    float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
    mesh.world = matrix::makeTranslation(x, y, z);
    //mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput(); // Handle user input
        renderer.clear(); // Clear the canvas for the next frame

        // Apply transformations to the meshes
     //   mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);
        mesh.world = matrix::makeTranslation(x, y, z);

        // Handle user inputs for transformations
        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;
        if (renderer.canvas.keyPressed('A')) x += -0.1f;
        if (renderer.canvas.keyPressed('D')) x += 0.1f;
        if (renderer.canvas.keyPressed('W')) y += 0.1f;
        if (renderer.canvas.keyPressed('S')) y += -0.1f;
        if (renderer.canvas.keyPressed('Q')) z += 0.1f;
        if (renderer.canvas.keyPressed('E')) z += -0.1f;

        AdaptiveMTConfig config = chooseAdaptiveConfig(renderer, history);
        if (!pool || activeThreads != config.threadCount) {
            pool = std::make_unique<ThreadPool>(config.threadCount);
            activeThreads = config.threadCount;
        }

        WorkloadAccumulator acc = makeAccumulator(6, 4);

        // Render each object in the scene
#if OPT_MULTITHREAD
        for (auto& m : scene) {
            renderMT(renderer, m, camera, L, *pool, config, &acc);
        }
        pool->waitIdle();
        updateHistory(history, finalizeWorkload(acc));
#else
        for (auto& m : scene) {
            render(renderer, m, camera, L);
        }
#endif

        renderer.present(); // Display the rendered frame
    }
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
    unsigned int r = rng.getRandomInt(0, 3);

    switch (r) {
    case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
    default: return matrix::makeIdentity();
    }
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
    Profiler profiler;
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
#if OPT_LIGHT_PRE_NORMALIZE
    L.omega_i.normalise();
#endif

#if OPT_MULTITHREAD
    std::unique_ptr<ThreadPool> pool;
    int activeThreads = 0;
    AdaptiveHistory history;
#endif

    bool running = true;

    std::vector<Mesh*> scene;

    // Create a scene of 40 cubes with random rotations
    for (unsigned int i = 0; i < 20; i++) {
        Mesh* m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
        m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (profiler.needToLoop()) {
        auto frameTimer = profiler.scope();
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                //end = std::chrono::high_resolution_clock::now();
                //std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                //start = std::chrono::high_resolution_clock::now();
            }
        }

#if OPT_MULTITHREAD
        AdaptiveMTConfig config = chooseAdaptiveConfig(renderer, history);
        if (!pool || activeThreads != config.threadCount) {
            pool = std::make_unique<ThreadPool>(config.threadCount);
            activeThreads = config.threadCount;
        }

        WorkloadAccumulator acc = makeAccumulator(6, 4);

#if OPT_RENDER_SCENE
        renderSceneMT(renderer, scene, camera, L, *pool, config, &acc);
#else
        for (auto& m : scene) {
            renderMT(renderer, m, camera, L, *pool, config, &acc);
        }
#endif
        pool->waitIdle();
        updateHistory(history, finalizeWorkload(acc));
#else
        for (auto& m : scene) {
            render(renderer, m, camera, L);
        }
#endif
        renderer.present();
    }

    for (auto& m : scene)
        delete m;

	profiler.printReport("Scene 1");
#if OPT_MULTITHREAD
    if (pool) {
        pool->dumpStats();
    }
#endif
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
    Profiler profiler;
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
#if OPT_LIGHT_PRE_NORMALIZE
    L.omega_i.normalise();
#endif

#if OPT_MULTITHREAD
    std::unique_ptr<ThreadPool> pool;
    int activeThreads = 0;
    AdaptiveHistory history;
#endif

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    const bool reportCycleTiming = true;
    auto cycleStart = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> cycleEnd;

    bool running = true;
    while (profiler.needToLoop()) {
        auto frameTimer = profiler.scope();
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                //end = std::chrono::high_resolution_clock::now();
                //std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                //start = std::chrono::high_resolution_clock::now();

                if (reportCycleTiming) {
                    cycleEnd = std::chrono::high_resolution_clock::now();
                    Profiler::recordRegion("Scene2 Cycle", std::chrono::duration<double, std::milli>(cycleEnd - cycleStart).count());
                    cycleStart = cycleEnd;
                }
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

#if OPT_MULTITHREAD
        AdaptiveMTConfig config = chooseAdaptiveConfig(renderer, history);
        if (!pool || activeThreads != config.threadCount) {
            pool = std::make_unique<ThreadPool>(config.threadCount);
            activeThreads = config.threadCount;
        }

        WorkloadAccumulator acc = makeAccumulator(6, 4);

#if OPT_RENDER_SCENE
        renderSceneMT(renderer, scene, camera, L, *pool, config, &acc);
#else
        for (auto& m : scene) {
            renderMT(renderer, m, camera, L, *pool, config, &acc);
        }
#endif
        pool->waitIdle();
        updateHistory(history, finalizeWorkload(acc));
#else
        for (auto& m : scene) {
            render(renderer, m, camera, L);
        }
#endif
        renderer.present();
    }

    for (auto& m : scene)
        delete m;

	profiler.printReport("Scene 2");
#if OPT_MULTITHREAD
    if (pool) {
	    pool->dumpStats();
    }
#endif
}

void scene3() {
    Profiler profiler;
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
#if OPT_LIGHT_PRE_NORMALIZE
    L.omega_i.normalise();
#endif

#if OPT_MULTITHREAD
    std::unique_ptr<ThreadPool> pool;
    int activeThreads = 0;
    AdaptiveHistory history;
#endif

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    struct RotatingMesh {
        Mesh* mesh;
        rRot rot;
    };
    std::vector<RotatingMesh> rotatingMeshes;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    auto addCube = [&](float x, float y, float z, float size) {
        Mesh* m = new Mesh();
        *m = Mesh::makeCube(size);
        scene.push_back(m);
        m->world = matrix::makeTranslation(x, y, z);
        rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
        rotatingMeshes.push_back({ m, r });
        };

    // Dense cluster on the left to create heavy tiles (load imbalance in parallel rendering)
    for (unsigned int y = 0; y < 40; y++) {
        for (unsigned int x = 0; x < 65; x++) {
            addCube(-120.0f + (static_cast<float>(x) * 2.0f), 45.0f - (static_cast<float>(y) * 2.0f), -35.f, 1.f);
        }
    }

    // Sparse cluster on the right to keep some tiles light
    for (unsigned int y = 0; y < 10; y++) {
        for (unsigned int x = 0; x < 20; x++) {
            addCube(30.0f + (static_cast<float>(x) * 3.5f), 20.0f - (static_cast<float>(y) * 3.5f), -60.f, 1.5f);
        }
    }

    // Offscreen cluster for frustum culling tests
    for (unsigned int y = 0; y < 20; y++) {
        for (unsigned int x = 0; x < 20; x++) {
            addCube(210.0f + (static_cast<float>(x) * 3.f), 80.0f - (static_cast<float>(y) * 3.f), -45.f, 1.f);
        }
    }

    // Large occluder to increase overdraw and highlight early-z behavior
    Mesh* occluder = new Mesh();
    *occluder = Mesh::makeRectangle(-80.f, -45.f, 20.f, 45.f);
    occluder->world = matrix::makeTranslation(-20.f, 0.f, -22.f);
    scene.push_back(occluder);

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    int cycle = 0;

    const bool reportCycleTiming = true;
    auto cycleStart = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> cycleEnd;

    while (profiler.needToLoop()) {
        auto frameTimer = profiler.scope();
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the scene
        for (auto& r : rotatingMeshes)
            r.mesh->world = r.mesh->world * matrix::makeRotateXYZ(r.rot.x, r.rot.y, r.rot.z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                if (reportCycleTiming) {
                    cycleEnd = std::chrono::high_resolution_clock::now();
                    Profiler::recordRegion("Scene3 Cycle", std::chrono::duration<double, std::milli>(cycleEnd - cycleStart).count());
                    cycleStart = cycleEnd;
                }
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

#if OPT_MULTITHREAD
        AdaptiveMTConfig config = chooseAdaptiveConfig(renderer, history);
        if (!pool || activeThreads != config.threadCount) {
            pool = std::make_unique<ThreadPool>(config.threadCount);
            activeThreads = config.threadCount;
        }

        WorkloadAccumulator acc = makeAccumulator(6, 4);

        #if OPT_RENDER_SCENE
        renderSceneMT(renderer, scene, camera, L, *pool, config, &acc);
        #else
        for (auto& m : scene) {
            renderMT(renderer, m, camera, L, *pool, config, &acc);
        }
        #endif
        pool->waitIdle();
        updateHistory(history, finalizeWorkload(acc));
#else
        for (auto& m : scene) {
            render(renderer, m, camera, L);
        }
#endif
        renderer.present();
    }

    for (auto& m : scene)
        delete m;

    profiler.printReport("Scene 3");
#if OPT_MULTITHREAD
    if (pool) {
        pool->dumpStats();
    }
#endif
}


// Entry point of the application
// No input variables
int main() {
    scene1();
    scene2();
    scene3();

    //if (SCENE_SELECT == 1) {
    //    scene1();
    //}
    //else if (SCENE_SELECT == 2) {
    //    scene2();
    //}
    //else if (SCENE_SELECT == 3) {
    //    scene3();
    //}
    //else {
    //    sceneTest();
    //}

    return 0;
}
