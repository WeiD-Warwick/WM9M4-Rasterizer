#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <chrono>

#include <cmath>
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
void renderMT(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L, ThreadPool& pool) {
    matrix p = renderer.perspective * camera * mesh->world;
    const int W = (int)renderer.canvas.getWidth();
    const int H = (int)renderer.canvas.getHeight();

    auto vCache = std::make_shared<VertexSOA>();
    auto lp = std::make_shared<avx2::LightSIMD>(L, mesh->ka, mesh->kd);

    {
        Profiler::RegionTimer t("Vertex_Stage");
        mesh->preProcessVertexCache(p, (float)W, (float)H, *vCache);
    }

    const int tilesX = (W + MT_TILE_W - 1) / MT_TILE_W;
    const int tilesY = (H + MT_TILE_H - 1) / MT_TILE_H;

    std::vector<ThreadPool::Job> batch;
    batch.reserve(tilesX * tilesY);

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            ScissorRect sc{
                tx * MT_TILE_W,
                ty * MT_TILE_H,
                std::min((tx + 1) * MT_TILE_W, W),
                std::min((ty + 1) * MT_TILE_H, H)
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
    ThreadPool pool(THREAD_COUNT);
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

        // Render each object in the scene
#if OPT_MULTITHREAD
        for (auto& m : scene) {
            renderMT(renderer, m, camera, L, pool);
        }
        pool.waitIdle();
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
    ThreadPool pool(THREAD_COUNT);
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
        for (auto& m : scene) {
            renderMT(renderer, m, camera, L, pool);
        }
        pool.waitIdle();
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
    pool.dumpStats();
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
    ThreadPool pool(THREAD_COUNT);
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
        for (auto& m : scene) {
            renderMT(renderer, m, camera, L, pool);
        }
        pool.waitIdle();
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
	pool.dumpStats();
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
    ThreadPool pool(THREAD_COUNT);
#endif

    struct rRot { float x; float y; float z; };
    struct RotMesh { Mesh* mesh; rRot rot; };

    std::vector<Mesh*> scene;
    std::vector<RotMesh> rotatingMeshes;
    std::vector<Mesh*> orbitSpheres;
    std::vector<Mesh*> layeredSlats;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    auto addRect = [&](float minX, float minY, float maxX, float maxY, float z) {
        Mesh* rect = new Mesh();
        *rect = Mesh::makeRectangle(minX, minY, maxX, maxY);
        rect->world = matrix::makeTranslation(0.0f, 0.0f, z);
        scene.push_back(rect);
        return rect;
        };

    auto addCube = [&](float size, float x, float y, float z) {
        Mesh* cube = new Mesh();
        *cube = Mesh::makeCube(size);
        cube->world = matrix::makeTranslation(x, y, z) * makeRandomRotation();
        scene.push_back(cube);
        rotatingMeshes.push_back({ cube, { rng.getRandomFloat(-.04f, .04f), rng.getRandomFloat(-.04f, .04f), rng.getRandomFloat(-.04f, .04f) } });
        return cube;
        };

    auto addSphere = [&](float radius, int lat, int lon, float x, float y, float z) {
        Mesh* s = new Mesh();
        *s = Mesh::makeSphere(radius, lat, lon);
        s->world = matrix::makeTranslation(x, y, z);
        scene.push_back(s);
        return s;
        };

    // Foreground occlusion frame (promotes layered/early-z benefits)
    const float frameZ = -6.0f;
    addRect(-8.0f, 2.5f, 8.0f, 6.5f, frameZ);
    addRect(-8.0f, -6.5f, 8.0f, -2.5f, frameZ);
    addRect(-8.0f, -2.5f, -3.5f, 2.5f, frameZ);
    addRect(3.5f, -2.5f, 8.0f, 2.5f, frameZ);

    // Layered slats behind the frame to generate depth layers for hierarchical rasterization
    for (int layer = 0; layer < 4; layer++) {
        float z = -10.0f - static_cast<float>(layer) * 6.0f;
        for (int s = 0; s < 6; s++) {
            float x0 = -7.5f + static_cast<float>(s) * 2.6f;
            float x1 = x0 + 1.6f;
            Mesh* slat = addRect(x0, -6.0f, x1, 6.0f, z);
            layeredSlats.push_back(slat);
        }
    }

    // Clustered cube fields (encourages triangle binning by spatial locality)
    const vec4 clusterCenters[] = {
        vec4(-5.0f, 3.5f, -24.0f, 1.0f),
        vec4(5.0f, 3.5f, -24.0f, 1.0f),
        vec4(-5.0f, -3.5f, -24.0f, 1.0f),
        vec4(5.0f, -3.5f, -24.0f, 1.0f),
        vec4(0.0f, 0.0f, -32.0f, 1.0f)
    };

    for (const auto& c : clusterCenters) {
        for (int y = 0; y < 6; y++) {
            for (int x = 0; x < 6; x++) {
                float jitterX = rng.getRandomFloat(-0.15f, 0.15f);
                float jitterY = rng.getRandomFloat(-0.15f, 0.15f);
                float jitterZ = rng.getRandomFloat(-1.2f, 1.2f);
                float px = c[0] + (static_cast<float>(x) * 0.7f) + jitterX;
                float py = c[1] + (static_cast<float>(y) * 0.7f) + jitterY;
                float pz = c[2] + jitterZ;
                addCube(0.32f, px, py, pz);
            }
        }
    }

    // High-poly spheres to keep per-cluster triangle counts high but localized
    for (int i = 0; i < 5; i++) {
        float angle = static_cast<float>(i) * (2.0f * static_cast<float>(M_PI) / 5.0f);
        float px = std::cos(angle) * 4.2f;
        float py = std::sin(angle) * 4.2f;
        Mesh* sphere = addSphere(1.1f, 32, 64, px, py, -38.0f);
        orbitSpheres.push_back(sphere);
    }

    // Fill-rate slabs (overdraw)
    for (int i = 0; i < 3; i++) {
        Mesh* slab = new Mesh();
        *slab = Mesh::makeRectangle(-10.0f, -10.0f, 10.0f, 10.0f);
        slab->world = matrix::makeTranslation(0.0f, 0.0f, -10.0f - static_cast<float>(i) * 2.0f);
        scene.push_back(slab);
    }

    // Far micro-geometry layer (small triangles confined to tiles)
    for (int y = 0; y < 12; y++) {
        for (int x = 0; x < 12; x++) {
            Mesh* shard = new Mesh();
            *shard = Mesh::makeRectangle(-0.18f, -0.18f, 0.18f, 0.18f);
            float px = -7.0f + static_cast<float>(x) * 1.2f + rng.getRandomFloat(-0.2f, 0.2f);
            float py = -5.0f + static_cast<float>(y) * 1.0f + rng.getRandomFloat(-0.2f, 0.2f);
            float pz = -55.0f + rng.getRandomFloat(-2.0f, 2.0f);
            shard->world = matrix::makeTranslation(px, py, pz);
            scene.push_back(shard);
        }
    }

    float zoffset = 5.0f;
    float zstep = -0.22f;
    float t = 0.0f;

    while (profiler.needToLoop()) {
        auto frameTimer = profiler.scope();
        renderer.canvas.checkInput();
        renderer.clear();

        t += 0.02f;
        float camX = std::sin(t * 0.6f) * 2.2f;
        float camY = std::cos(t * 0.4f) * 1.6f;
        camera = matrix::makeTranslation(-camX, -camY, -zoffset);

        zoffset += zstep;
        if (zoffset < -60.0f || zoffset > 5.0f)
            zstep *= -1.f;

        // Rotate all rotating meshes
        for (auto& rm : rotatingMeshes) {
            rm.mesh->world = rm.mesh->world * matrix::makeRotateXYZ(rm.rot.x, rm.rot.y, rm.rot.z);
        }

        // Orbit spheres to keep depth changes active
        float orbitScale = 1.0f + (std::sin(t * 1.2f) * 0.15f);
        for (int i = 0; i < static_cast<int>(orbitSpheres.size()); i++) {
            float angle = t * 0.7f + (static_cast<float>(i) * (2.0f * static_cast<float>(M_PI) / 5.0f));
            float orbitX = std::cos(angle) * 4.2f;
            float orbitY = std::sin(angle) * 4.2f;
            orbitSpheres[i]->world = matrix::makeTranslation(orbitX, orbitY, -38.0f)
                * matrix::makeRotateXYZ(t * 0.25f, t * 0.35f, t * 0.2f)
                * matrix::makeScale(orbitScale);
        }

        // Slight swaying of slats for depth-layer motion
        for (int i = 0; i < static_cast<int>(layeredSlats.size()); i++) {
            float sway = std::sin(t * 0.4f + static_cast<float>(i)) * 0.08f;
            float z = -10.0f - (static_cast<float>(i / 6) * 6.0f);
            layeredSlats[i]->world = matrix::makeTranslation(sway, 0.0f, z);
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

#if OPT_MULTITHREAD
        for (auto& m : scene) {
            renderMT(renderer, m, camera, L, pool);
        }
        pool.waitIdle();
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
    pool.dumpStats();
#endif
}


// Entry point of the application
// No input variables
int main() {

    if (SCENE_SELECT == 1) {
        scene1();
    }
    else if (SCENE_SELECT == 2) {
        scene2();
    }
    else if (SCENE_SELECT == 3) {
        scene3();
    }
    else {
        sceneTest();
    }

    return 0;
}