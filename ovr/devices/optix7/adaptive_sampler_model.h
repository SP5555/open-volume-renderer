#pragma once

#include "optix7_common.h"
#include <cuda_misc.h>

#include <iostream>
#include <vector>

class MLModel {
private:
    ovr::vec2i currentSize = { -1, -1 };

public:
    MLModel() = default;

    void importance_map_resize(float*& d_map, ovr::vec2i newSize) {
        if (currentSize.x != newSize.x || currentSize.y != newSize.y) {
            if (d_map) {
                CUDA_CHECK(cudaFree(d_map));
            }
            size_t mapSize = newSize.x * newSize.y * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_map, mapSize));
            currentSize = newSize;
        }
    }

    void importance_map_update(void* d_inRGBA, float* d_outImportanceMap, ovr::vec2i low_res_size, ovr::vec2i size) {

        // std::vector<ovr::vec4f> h_lowres(low_res_size.x * low_res_size.y);
        // CUDA_CHECK(cudaMemcpy(h_lowres.data(), d_inputRGBA, sizeof(ovr::vec4f) * h_lowres.size(), cudaMemcpyDeviceToHost));

        // printf("Low-res size: %d x %d | High-res size: %d x %d\n",
        //         low_res_size.x, low_res_size.y,
        //         size.x, size.y);

        std::vector<float> h_map(size.x * size.y);

        // const int grid_size = 40;
        // for (int y = 0; y < size.y; y++) {
        //     for (int x = 0; x < size.x; x++) {
        //         int idx = y * size.x + x;
        //         // h_map[idx] = ((x / grid_size + y / grid_size) % 2 == 0) ? 1.0f : 0.0f;
        //         h_map[idx] = (x < size.x / 2) ? 1.0f : 4.0f; // dummy
        //     }
        // }

        // Circle parameters
        const float cx = size.x * 0.5f;
        const float cy = size.y * 0.5f;
        const float radius = std::min(size.x, size.y) * 0.2f;

        for (int y = 0; y < size.y; y++) {
            for (int x = 0; x < size.x; x++) {
                int idx = y * size.x + x;

                float dx = x - cx;
                float dy = y - cy;
                float dist2 = dx * dx + dy * dy;

                if (dist2 <= radius * radius) {
                    h_map[idx] = 4.0f;
                } else {
                    h_map[idx] = 1.0f;
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(d_outImportanceMap, h_map.data(), sizeof(float) * h_map.size(), cudaMemcpyHostToDevice));
    }
};

