/********************  HEADERS  *********************/
#include "./header/lbm_config.h"
#include "./header/lbm_phys.h"
#include <assert.h>
#include <stdlib.h>

__device__ static float get_cell_density(const float *cell) {
  // vars
  float res = 0.0;

  // loop on directions
  for (int k = 0; k < DIRECTIONS; k++)
    res += cell[k];

  return res;
}

__device__ static void get_cell_velocity(float v[DIMENSIONS], const float *cell,
                                         const float cell_density) {
  // vars
  float temp;
  float div = 1.0 / cell_density;

  const float direction_matrix[DIRECTIONS][DIMENSIONS] = {
      {+0.0f, +0.0f}, {+1.0f, +0.0f}, {+0.0f, +1.0f},
      {-1.0f, +0.0f}, {+0.0f, -1.0f}, {+1.0f, +1.0f},
      {-1.0f, +1.0f}, {-1.0f, -1.0f}, {+1.0f, -1.0f}};

  // loop on all dimensions
  for (int d = 0; d < DIMENSIONS; d++) {
    // reset value
    temp = 0.0;

    // sum all directions
    for (int k = 0; k < DIRECTIONS; k++) {
      temp += cell[k] * direction_matrix[k][d];
    }

    // normalize
    v[d] = temp * div;
  }
}

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

__device__ static void
compute_inflow_zou_he_poiseuille_distr(float *cell, const int j,
                                       const int height,
                                       const float inflow_max_velocity) {
  float v = helper_compute_poiseuille(j, height, inflow_max_velocity);

  // compute rho from u and inner flow on surface
  float density =
      (cell[0] + cell[2] + cell[4] + 2 * (cell[3] + cell[6] + cell[7])) *
      (1.0 - v);

  // now compute unknown microscopic values
  float a = 0.166667 * (density * v);
  cell[1] = cell[3];
  cell[5] = cell[7] - 0.5 * (cell[2] - cell[4]) + a;
  cell[8] = cell[6] + 0.5 * (cell[2] - cell[4]) + a;
}

__device__ static void compute_outflow_zou_he_const_density(float *cell) {
  // compute macroscopic v depeding on inner flow going onto the wall
  float v =
      (cell[0] + cell[2] + cell[4] + 2 * (cell[1] + cell[5] + cell[8])) - 1.0;

  // now can compute unknown microscopic values
  float a = 0.166667 * v;
  cell[3] = cell[1] - 0.66667 * v;
  cell[7] = cell[5] + 0.5 * (cell[2] - cell[4]) - a;
  cell[6] = cell[8] + 0.5 * (cell[4] - cell[2]) - a;
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

__global__ void kernel_macroscopic_mesh(lbm_file_entry_t *mesh_out,
                                        float *mesh_in, int width, int height,
                                        lbm_config_t config) {
  // get thread column
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  // get thread row
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  // get index of the thread
  const int i = row * width + column;
  // get the index where to write in the mesh_out array
  const int j = column * height + row;

  // test if the thread is in the mesh
  if ((column >= width) || (row >= height))
    return;

  // if is obstacle set to -1
  int obstacle_row = row - config.obstacle_y;
  int obstacle_column = column - config.obstacle_x;
  int obstacle_index = obstacle_row * (config.obstacle_width) + obstacle_column;

  if ((0 <= obstacle_row && obstacle_row < config.obstacle_height) &&
      (0 <= obstacle_column && obstacle_column < config.obstacle_width) &&
      config.obstacle_mesh[obstacle_index]) {
    mesh_out[j].density = -0.001;
    mesh_out[j].v = -0.001;
  } else {
    float density = get_cell_density(&(mesh_in[i * (DIRECTIONS)]));
    float v[DIMENSIONS] = {0.0f, 0.0f};
    get_cell_velocity(v, &(mesh_in[i * (DIRECTIONS)]), density);
    float norm = sqrt(get_vect_norme_2(v, v));

    mesh_out[j].density = density;
    mesh_out[j].v = norm;
  }
}

__global__ void kernel_special_cells(float *mesh, int width, int height,
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

  const int opposite_of[DIRECTIONS] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

  // compute_inflow_zou_he_poiseuille_distr
  if (column == 0 && row != 0 && row != height - 1) {
    compute_inflow_zou_he_poiseuille_distr(&(mesh[(i * DIRECTIONS)]), row,
                                           height, config.inflow_max_velocity);
  }

  // compute_outflow_zou_he_const_density
  if (column == (width - 1) && row != 0 && row != height - 1) {
    compute_outflow_zou_he_const_density(&(mesh[(i * DIRECTIONS)]));
  }

  // compute_bounce_back
  int obstacle_row = row - config.obstacle_y;
  int obstacle_column = column - config.obstacle_x;
  int obstacle_index = obstacle_row * (config.obstacle_width) + obstacle_column;

  if ((0 <= obstacle_row && obstacle_row < config.obstacle_height) &&
      (0 <= obstacle_column && obstacle_column < config.obstacle_width) &&
      config.obstacle_mesh[obstacle_index]) {
    for (int k = 0; k < DIRECTIONS; k++) {
      mesh[(i * DIRECTIONS) + k] = mesh[(i * DIRECTIONS) + opposite_of[k]];
    }
  }

  // walls
  if (row == 0 || row == height - 1) {
    for (int k = 0; k < DIRECTIONS; k++)
      mesh[(i * DIRECTIONS) + k] = mesh[(i * DIRECTIONS) + opposite_of[k]];
  }
}

__global__ void kernel_collision(float *mesh_out, float *mesh_in, int width,
                                 int height, lbm_config_t config) {
  // get thread column
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  // get thread row
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  // get index of the thread
  const int i = row * width + column;
  // test if the thread is in the mesh
  if (!(column > 0 && column < width - 1) || !(row > 0 && row < height - 1))
    return;

  float v[2];

  // compute macroscopic values
  float density = get_cell_density(&(mesh_in[(i * DIRECTIONS)]));
  get_cell_velocity(v, &(mesh_in[(i * DIRECTIONS)]), density);
  // loop on microscopic directions
  for (int k = 0; k < DIRECTIONS; k++) {
    // compute f at equilibr.
    float feq = compute_equilibrium_profile(v, density, k);
    // compute f out
    mesh_out[(i * DIRECTIONS) + k] =
        mesh_in[(i * DIRECTIONS) + k] -
        config.relax_parameter * (mesh_in[(i * DIRECTIONS) + k] - feq);
  }
}

__global__ void kernel_propagation(float *mesh_out, float *mesh_in, int width,
                                   int height, lbm_config_t config) {
  // get thread column
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  // get thread row
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  // get index of the thread
  const int i = row * width + column;
  // test if the thread is in the mesh
  if ((column >= width) || (row >= height))
    return;

  const int int_direction_matrix[DIRECTIONS][DIMENSIONS] = {
      {+0, +0}, {+1, +0}, {+0, +1}, {-1, +0}, {+0, -1},
      {+1, +1}, {-1, +1}, {-1, -1}, {+1, -1}};

  for (int k = 0; k < DIRECTIONS; k++) {
    int cc = column + int_direction_matrix[k][0];
    int rr = row + int_direction_matrix[k][1];

    if ((cc >= 0 && cc < width) && (rr >= 0 && rr < height)) {
      int j = rr * width + cc;
      mesh_out[(j * DIRECTIONS) + k] = mesh_in[(i * DIRECTIONS) + k];
    }
  }
}