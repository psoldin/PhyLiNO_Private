// Self-check for --gpuDevices parsing (InputOptions.cpp): the "device:count"
// expansion that lets a scan fill one GPU with workers before spilling onto
// the next (see InputOptions::gpu_device_for_worker()).

#include "InputOptions.h"

#include <gtest/gtest.h>

TEST(InputOptionsTests, ParseGpuDevicesExpandsAndFillsInOrder) {
  const auto devices = io::parse_gpu_devices("0:2,1:3");
  ASSERT_EQ(devices.size(), 5u);
  EXPECT_EQ(devices[0], 0);
  EXPECT_EQ(devices[1], 0);
  EXPECT_EQ(devices[2], 1);
  EXPECT_EQ(devices[3], 1);
  EXPECT_EQ(devices[4], 1);
}

TEST(InputOptionsTests, ParseGpuDevicesEmptyMeansNoOverride) {
  EXPECT_TRUE(io::parse_gpu_devices("").empty());
}

TEST(InputOptionsTests, ParseGpuDevicesRejectsEntryWithoutColon) {
  EXPECT_THROW(io::parse_gpu_devices("0"), std::invalid_argument);
}
