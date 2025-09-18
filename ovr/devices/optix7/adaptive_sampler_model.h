#pragma once

#include "optix7_common.h"
#include <cuda_misc.h>

#include <iostream>
#include <vector>
#include <fstream>

class MLModel {
private:
    ovr::vec2i lowResPassSize = { -1, -1 };
    ovr::vec2i highResPassSize = { -1, -1 };
    
    static bool sizeChanged(const ovr::vec2i& a, const ovr::vec2i& b) {
        return (a.x != b.x) || (a.y != b.y);
    }

public:
    MLModel() = default;

    void resizeImportanceMap(float*& d_map,
                             const ovr::vec2i& newLowResSize,
                             const ovr::vec2i& newHighResSize) 
    {
        if (sizeChanged(lowResPassSize, newLowResSize)) {
            lowResPassSize = newLowResSize;
        }

        if (sizeChanged(highResPassSize, newHighResSize)) {
            highResPassSize = newHighResSize;

            if (d_map) CUDA_CHECK(cudaFree(d_map));
            CUDA_CHECK(cudaMalloc(&d_map, highResPassSize.x * highResPassSize.y * sizeof(float)));
        }
    }

    void updateImportanceMap(ovr::vec4f* d_inRGBA, float* d_outImportanceMap) {

        // safety check
        if (d_inRGBA == nullptr || d_outImportanceMap == nullptr) {
            throw std::runtime_error("updateImportanceMap called with null pointers.");
        }
        if (highResPassSize.x < 0 || highResPassSize.y < 0) {
            throw std::runtime_error("high res pass size is not valid.");
        }

        // std::vector<ovr::vec4f> h_lowres(low_res_size.x * low_res_size.y);
        // CUDA_CHECK(cudaMemcpy(h_lowres.data(), d_inputRGBA, sizeof(ovr::vec4f) * h_lowres.size(), cudaMemcpyDeviceToHost));

        // printf("Low-res size: %d x %d | High-res size: %d x %d\n",
        //         low_res_size.x, low_res_size.y,
        //         size.x, size.y);

        // std::vector<float> h_map(size.x * size.y);
        
        // const int grid_size = 40;
        // for (int y = 0; y < size.y; y++) {
        //     for (int x = 0; x < size.x; x++) {
        //         int idx = y * size.x + x;
        //         // h_map[idx] = ((x / grid_size + y / grid_size) % 2 == 0) ? 1.0f : 0.0f;
        //         h_map[idx] = (x < size.x / 2) ? 1.0f : 4.0f; // dummy
        //     }
        // }
        
        // Circle parameters
        // const float cx = size.x * 0.5f;
        // const float cy = size.y * 0.5f;
        // const float radius = std::min(size.x, size.y) * 0.2f;
        
        // for (int y = 0; y < size.y; y++) {
        //     for (int x = 0; x < size.x; x++) {
        //         int idx = y * size.x + x;
        
        //         float dx = x - cx;
        //         float dy = y - cy;
        //         float dist2 = dx * dx + dy * dy;
        
        //         if (dist2 <= radius * radius) {
        //             h_map[idx] = 4.0f;
        //         } else {
        //             h_map[idx] = 1.0f;
        //         }
        //     }
        // }
        
        std::vector<ovr::vec4f> h_lowres(lowResPassSize.x * lowResPassSize.y);
        {
            std::ifstream in("test_map_float4.bin", std::ios::binary);
            if (!in) {
                throw std::runtime_error("Could not open test_map_float4.bin");
            }
            in.read(reinterpret_cast<char*>(h_lowres.data()), h_lowres.size() * sizeof(ovr::vec4f));
        }

        // fill with base 1.0f
        std::vector<float> h_map(highResPassSize.x * highResPassSize.y, 1.0f);

        // Center offsets (where to place the low res inside full res)
        int offsetX = (highResPassSize.x - lowResPassSize.x) / 2;
        int offsetY = (highResPassSize.y - lowResPassSize.y) / 2;
        for (int y = 0; y < lowResPassSize.y; y++) {
            for (int x = 0; x < lowResPassSize.x; x++) {
                int low_idx = y * lowResPassSize.x + x;
                int high_x = offsetX + x;
                int high_y = offsetY + y;

                if (high_x < 0 || high_x >= highResPassSize.x ||
                    high_y < 0 || high_y >= highResPassSize.y)
                    continue;

                int high_idx = high_y * highResPassSize.x + high_x;

                float v = h_lowres[low_idx][3];  // read alpha channel only
                // float v = h_lowres[low_idx][0];  // or red

                // rescale [0,1] to [1,16]
                float importance = 1.0f + v * 15.0f;

                h_map[high_idx] = importance;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_outImportanceMap, h_map.data(), sizeof(float) * h_map.size(), cudaMemcpyHostToDevice));
    }
};

