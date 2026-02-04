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
    #if OPT_LIGHT_PRENORMALIZE
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
#if OPT_LIGHT_PRENORMALIZE
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
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
    Profiler profiler;
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
#if OPT_LIGHT_PRENORMALIZE
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

// Entry point of the application
// No input variables
int main() {

    if (SCENE_SELECT == 1) {
        scene1();
    }
    else if (SCENE_SELECT == 2) {
        scene2();
    }
    else {
        sceneTest();
    }

    return 0;
}