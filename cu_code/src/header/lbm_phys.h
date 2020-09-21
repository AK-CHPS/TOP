#ifndef LBM_PHYS_H
#define LBM_PHYS_H

#include "lbm_io.h"

__global__ void kernel_macroscopic_mesh(lbm_file_entry_t *mesh_out,
                                        float *mesh_in, int width, int height,
                                        lbm_config_t config);

__global__ void kernel_special_cells(float *mesh, int width, int height,
                                     lbm_config_t config);

__global__ void kernel_collision(float *mesh_out, float *mesh_in, int width,
                                 int height, lbm_config_t config);

__global__ void kernel_propagation(float *mesh_out, float *mesh_in, int width,
                                   int height, lbm_config_t config);

#endif // LBM_PHYS_H
