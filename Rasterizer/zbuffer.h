#pragma once

#include <concepts>

#if OPT_ZBUFFER
#include <algorithm>
#include <utility>
#include <vector>
#endif

// Zbuffer class for managing depth values during rendering.
// This class is template-constrained to only work with floating-point types (`float` or `double`).

template<std::floating_point T> // Restricts T to be a floating-point type
class Zbuffer {
    T* buffer;                  // Pointer to the buffer storing depth values - can also use unique_ptr []here
    unsigned int width, height; // Dimensions of the Z-buffer

#if OPT_ZBUFFER
    unsigned int tilesX, tilesY;
    std::vector<unsigned char> tileDirty;
    std::vector<unsigned int> dirtyTiles;
    static constexpr unsigned int tileSize = 16;
#endif

public:
    // Constructor to initialize a Z-buffer with the given width and height.
    // Allocates memory for the buffer.
    // Input Variables:
    // - w: Width of the Z-buffer.
    // - h: Height of the Z-buffer.
    Zbuffer(unsigned int w, unsigned int h) : buffer(nullptr) {
        create(w, h);
    }

    // Default constructor for creating an uninitialized Z-buffer.
    Zbuffer() {
    }

    // Creates or reinitialies the Z-buffer with the given width and height.
    // Allocates memory for the buffer.
    // Input Variables:
    // - w: Width of the Z-buffer.
    // - h: Height of the Z-buffer.
    void create(unsigned int w, unsigned int h) {
        width = w;
        height = h;
        if (buffer != nullptr) delete[] buffer; // remove previous version
        buffer = new T[width * height]; // Allocate memory for the buffer

#if OPT_ZBUFFER
        tilesX = (width + tileSize - 1) / tileSize;
        tilesY = (height + tileSize - 1) / tileSize;
        tileDirty.assign(tilesX * tilesY, 0);
        dirtyTiles.clear();
        clearAll();
#endif
    }

    // Accesses the depth value at the specified (x, y) coordinate.
    // Input Variables:
    // - x: X-coordinate of the pixel.
    // - y: Y-coordinate of the pixel.
    // Returns a reference to the depth value at (x, y).
    T& operator () (unsigned int x, unsigned int y) {
        return buffer[(y * width) + x]; // Convert 2D coordinates to 1D index
    }

    // Clears the Z-buffer by setting all depth values to 1.0f,
    // which represents the farthest possible depth.
    void clear() {
        // could also use fill_n
#if OPT_ZBUFFER
        for (unsigned int tileIndex : dirtyTiles) {
            clearTile(tileIndex);
            tileDirty[tileIndex] = 0;
        }
        dirtyTiles.clear();
#else
        for (unsigned int i = 0; i < width * height; i++) {
            buffer[i] = T(1.0); // Reset each depth value
        }
#endif
    }

    // remove copying
    Zbuffer(const Zbuffer&) = delete;
    Zbuffer& operator=(const Zbuffer&) = delete;

    // Destructor to clean up memory allocated for the Z-buffer.
    ~Zbuffer() {
        delete[] buffer; // Free the allocated memory
    }

    // move operators just in case
#if OPT_ZBUFFER
    Zbuffer(Zbuffer&& other) noexcept
        : buffer(other.buffer),
        width(other.width),
        height(other.height),
        tilesX(other.tilesX),
        tilesY(other.tilesY),
        tileDirty(std::move(other.tileDirty)),
        dirtyTiles(std::move(other.dirtyTiles)) {
#else
    Zbuffer(Zbuffer && other) noexcept : buffer(other.buffer), width(other.width), height(other.height) {
#endif
        other.buffer = nullptr;
    }

    Zbuffer& operator=(Zbuffer&& other) noexcept {
        if (this != &other) {
            delete[] buffer;
            buffer = other.buffer;
            width = other.width;
            height = other.height;
#if OPT_ZBUFFER
            tilesX = other.tilesX;
            tilesY = other.tilesY;
            tileDirty = std::move(other.tileDirty);
            dirtyTiles = std::move(other.dirtyTiles);
#endif
            other.buffer = nullptr;
        }
        return *this;
    }

#if OPT_ZBUFFER
    void markDirty(unsigned int x, unsigned int y) {
        const unsigned int tileX = x / tileSize;
        const unsigned int tileY = y / tileSize;
        const unsigned int tileIndex = tileY * tilesX + tileX;
        if (!tileDirty[tileIndex]) {
            tileDirty[tileIndex] = 1;
            dirtyTiles.push_back(tileIndex);
        }
    }

private:
    void clearAll() {
        for (unsigned int i = 0; i < width * height; i++) {
            buffer[i] = T(1.0);
        }
        std::fill(tileDirty.begin(), tileDirty.end(), 0);
        dirtyTiles.clear();
    }

    void clearTile(unsigned int tileIndex) {
        const unsigned int tileX = tileIndex % tilesX;
        const unsigned int tileY = tileIndex / tilesX;
        const unsigned int startX = tileX * tileSize;
        const unsigned int startY = tileY * tileSize;
        const unsigned int endX = std::min(startX + tileSize, width);
        const unsigned int endY = std::min(startY + tileSize, height);
        for (unsigned int y = startY; y < endY; ++y) {
            T* row = buffer + (y * width);
            for (unsigned int x = startX; x < endX; ++x) {
                row[x] = T(1.0);
            }
        }
    }
#endif
};
