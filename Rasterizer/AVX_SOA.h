#pragma once
#include <vector>
#include <immintrin.h>

struct VertexSOA {
    std::vector<float> px, py, pz, pw;
    std::vector<float> nx, ny, nz;
    std::vector<float> cr, cg, cb;

    int size() const { return px.size(); }

    void resize(int n) {
        px.resize(n);
        py.resize(n);
        pz.resize(n);
        pw.resize(n);
        nx.resize(n);
        ny.resize(n);
        nz.resize(n);
        cr.resize(n);
        cg.resize(n);
        cb.resize(n);
    }
};