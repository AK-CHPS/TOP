#ifndef LBM_INIT_H
#define LBM_INIT_H

#include "lbm_config.h"

__global__ void init_state_kernel(float *mesh, int width, int height,
                                  lbm_config_t config);

#endif // LBM_INIT_H
