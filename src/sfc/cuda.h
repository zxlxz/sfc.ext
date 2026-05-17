#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/cuda/tex.h"
#include "sfc/cuda/exec.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/memory.h"

#ifndef __CUDA_ARCH__
#include "sfc/cuda/fft.h"
#include "sfc/cuda/module.h"
#include "sfc/cuda/texture.h"
#endif
