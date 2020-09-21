/********************  HEADERS  *********************/
#include "./header/lbm_init.h"
#include "./header/lbm_phys.h"
#include <assert.h>

__device__ static float
helper_compute_poiseuille(const int j, const int height,
                          const float inflow_max_velocity) {
  return 4.0f * inflow_max_velocity / (height * height) * (height * j - j * j);
}

__device__ static float get_vect_norme_2(const float v1[DIMENSIONS],
                                         const float v2[DIMENSIONS]) {
  float res = 0.0;

  for (int k = 0; k < DIMENSIONS; k++)
    res += v1[k] * v2[k];

  return res;
}

__device__ static float compute_equilibrium_profile(float velocity[DIMENSIONS],
                                                    float density,
                                                    int direction) {
  const float equil_weight[DIRECTIONS] = {
      4.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f, 1.0f / 9.0f,
      1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

  const float direction_matrix[DIRECTIONS][DIMENSIONS] = {
      {+0.0f, +0.0f}, {+1.0f, +0.0f}, {+0.0f, +1.0f},
      {-1.0f, +0.0f}, {+0.0f, -1.0f}, {+1.0f, +1.0f},
      {-1.0f, +1.0f}, {-1.0f, -1.0f}, {+1.0f, -1.0f}};

  // vars
  float p, p2, feq, v2;

  v2 = get_vect_norme_2(velocity, velocity);

  // calc e_i * v_i / c
  p = get_vect_norme_2(direction_matrix[direction], velocity);

  p2 = p * p;

  // terms without density and direction weight
  feq = 1.0f + (3.0f * p) + (4.5f * p2) - (1.5f * v2);

  // mult all by density and direction weight
  feq *= equil_weight[direction] * density;

  return feq;
}

__global__ void init_state_kernel(float *mesh, int width, int height,
                                  lbm_config_t config) {
  // get thread column
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  // get thread row
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  // get index of the thread
  const int i = row * width + column;
  // test if the thread is in the mesh
  if ((column >= width) || (row >= height))
    return;

  const float equil_weight[DIRECTIONS] = {
      4.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f, 1.0f / 9.0f,
      1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

  // global poisseuille
  float v[DIMENSIONS] = {
      helper_compute_poiseuille(row, height, config.inflow_max_velocity), 0.0f};

  for (int k = 0; k < DIRECTIONS; k++) {
    mesh[(i * DIRECTIONS) + k] = compute_equilibrium_profile(v, 1.0f, k);
  }

  // border
  if ((row == 0) || (row == height - 1)) {
    for (int k = 0; k < DIRECTIONS; k++) {
      mesh[(i * DIRECTIONS) + k] = equil_weight[k];
    }
  }

  // obstacle
  int obstacle_row = row - config.obstacle_y;
  int obstacle_column = column - config.obstacle_x;
  int obstacle_index = obstacle_row * (config.obstacle_width) + obstacle_column;

  if ((0 <= obstacle_row && obstacle_row < config.obstacle_height) &&
      (0 <= obstacle_column && obstacle_column < config.obstacle_width) &&
      config.obstacle_mesh[obstacle_index]) {
    for (int k = 0; k < DIRECTIONS; k++) {
      mesh[(i * DIRECTIONS) + k] = equil_weight[k];
    }
  }
}