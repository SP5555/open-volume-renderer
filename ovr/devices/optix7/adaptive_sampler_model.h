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

    int baseSPP = 16;
    bool isAdaptive = false;
    std::string importanceMapFile = "mechhand_80_400_-100_80_0_0_mask.bin";
    
    static bool sizeChanged(const ovr::vec2i& a, const ovr::vec2i& b) {
        return (a.x != b.x) || (a.y != b.y);
    }

public:
    MLModel() = default;

    void setSampleBudget(int spp) {
        baseSPP = std::max(1, spp);
    }

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

        // printf("Low Res: (%d, %d) | High Res: (%d, %d)\n",
        //       lowResPassSize.x, lowResPassSize.y,
        //       highResPassSize.x, highResPassSize.y);

        // safety check
        if (d_inRGBA == nullptr || d_outImportanceMap == nullptr) {
            throw std::runtime_error("updateImportanceMap called with null pointers.");
        }
        if (highResPassSize.x < 0 || highResPassSize.y < 0) {
            throw std::runtime_error("high res pass size is not valid.");
        }

        float totalBudget =
            static_cast<float>(highResPassSize.long_product())
            * static_cast<float>(baseSPP);

        // std::vector<float> h_lowres(lowResPassSize.x * lowResPassSize.y);
        // CUDA_CHECK(cudaMemcpy(h_lowres.data(), d_inRGBA, sizeof(float) * h_lowres.size(), cudaMemcpyDeviceToHost));

        // fill with base 1.0f
        std::vector<float> h_map(highResPassSize.x * highResPassSize.y, 0.0f);
        
        // ===== 1 spp map =====
        // for (int y = 0; y < highResPassSize.y; y++) {
        //     for (int x = 0; x < highResPassSize.x; x++) {
        //         int idx = y * highResPassSize.x + x;
        //         h_map[idx] = 1.0f;
        //     }
        // }
        
        // ===== Circle map =====
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
        
        // ===== read bin map =====
        std::vector<float> h_lowres(lowResPassSize.x * lowResPassSize.y);
        double sumWeights = 0.0;
        if (isAdaptive) {
            std::ifstream in(importanceMapFile, std::ios::binary);

            if (!in) {
                throw std::runtime_error("Could not open bin file");
            }
            in.read(reinterpret_cast<char*>(h_lowres.data()), h_lowres.size() * sizeof(float));
            in.close();

            for (float v : h_lowres) {
                sumWeights += static_cast<double>(v);
            }
            
            if (sumWeights <= 0.0) {
                throw std::runtime_error("Sum of weights in lowres map is non-positive.");
            }
        }

        long long totalCheck = 0;

        // Center offsets (where to place the low res inside full res)
        int offsetX = (highResPassSize.x - lowResPassSize.x) / 2;
        int offsetY = (highResPassSize.y - lowResPassSize.y) / 2;
        for (int y = 0; y < lowResPassSize.y; y++) {
            for (int x = 0; x < lowResPassSize.x; x++) {
                // int low_idx = y * lowResPassSize.x + x;
                int low_idx = (lowResPassSize.y - 1 - y) * lowResPassSize.x + x;
                int high_x = x + offsetX;
                int high_y = y + offsetY;

                if (high_x < 0 || high_x >= highResPassSize.x ||
                    high_y < 0 || high_y >= highResPassSize.y)
                    continue;

                int high_idx = high_y * highResPassSize.x + high_x;

                float val;
                if (isAdaptive) { // adaptive
                    float v = h_lowres[low_idx];
                    val = (v / static_cast<float>(sumWeights)) * totalBudget;
                } else { // uniform
                    val = static_cast<float>(baseSPP);
                }

                if (val > 0.0f && val < 1.0f) {
                    val = 1.0f;
                }

                h_map[high_idx] = val;
                totalCheck += static_cast<long long>(val);
            }
        }

        printf("Total assigned samples: %lld / %.0f\n", totalCheck, totalBudget);
        printf("Unused samples: %.0f\n", totalBudget - static_cast<double>(totalCheck));

        CUDA_CHECK(cudaMemcpy(d_outImportanceMap, h_map.data(), sizeof(float) * h_map.size(), cudaMemcpyHostToDevice));
    }
};

