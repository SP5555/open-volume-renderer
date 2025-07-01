#include "device.h"
#include "device_impl.h"

#include <chrono>

namespace ovr::optix7 {

DeviceOptix7::~DeviceOptix7()
{
  pimpl.reset();
}

DeviceOptix7::DeviceOptix7() : MainRenderer(), pimpl(new Impl()) {}

void
DeviceOptix7::init(int argc, const char** argv)
{
  pimpl->init(argc, argv, this);
  pimpl->commit();
}

void
DeviceOptix7::swap()
{
  pimpl->swap();
}

void
DeviceOptix7::commit()
{
  pimpl->commit();
}

void
DeviceOptix7::render()
{
  auto s_LR = std::chrono::high_resolution_clock::now();
  pimpl->render_low_res();
  CUDA_CHECK(cudaDeviceSynchronize());
  auto e_LR = std::chrono::high_resolution_clock::now();
  auto d_LR = std::chrono::duration_cast<std::chrono::microseconds>(e_LR - s_LR);

  pimpl->importance_map_update();

  auto s_HR = std::chrono::high_resolution_clock::now();
  pimpl->render();
  CUDA_CHECK(cudaDeviceSynchronize());
  auto e_HR = std::chrono::high_resolution_clock::now();
  auto d_HR = std::chrono::duration_cast<std::chrono::microseconds>(e_HR - s_HR);

  printf("480p pass: %.3f ms | Full-res pass: %.3f ms\n", d_LR / 1000.0, d_HR / 1000.0);
  
  render_time += d_LR.count();
  render_time += d_HR.count();
}

void
DeviceOptix7::mapframe(FrameBufferData* fb)
{
  return pimpl->mapframe(fb);
}

} // namespace ovr::optix7
