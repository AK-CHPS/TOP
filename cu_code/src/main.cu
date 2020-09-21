#include "./header/lbm_config.h"
#include "./header/lbm_init.h"
#include "./header/lbm_io.h"
#include "./header/lbm_phys.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

#if 0
typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;

void ReadImage(const char *fileName, byte **pixels, int32 *width, int32 *height, int32 *bytesPerPixel)
{
	// Open the bitmap file
    FILE *imageFile = fopen(fileName, "rb");

    // seek to data
    fseek(imageFile, DATA_OFFSET_OFFSET, SEEK_SET);
    int32 dataOffset;
    
    // read the data offset
    fread(&dataOffset, 4, 1, imageFile);
    
    // seek and read width information
    fseek(imageFile, WIDTH_OFFSET, SEEK_SET);
    fread(width, 4, 1, imageFile);
    
    // seek and read height information
    fseek(imageFile, HEIGHT_OFFSET, SEEK_SET);
   	fread(height, 4, 1, imageFile);

   	// seek and read the number of byte per pixel
    int16 bitsPerPixel;
    fseek(imageFile, BITS_PER_PIXEL_OFFSET, SEEK_SET);
    fread(&bitsPerPixel, 2, 1, imageFile);
    *bytesPerPixel = ((int32)bitsPerPixel) / 8;
 
 	// get the size of file in byte
	int paddedRowSize = (((*width) * 3 + 3) & (~3)) * (*bytesPerPixel);/*(int)(4 * ceil((float)(*width) / 4.0f))*(*bytesPerPixel);*/
	int unpaddedRowSize = (*width)*(*bytesPerPixel);
	int totalSize = unpaddedRowSize*(*height);
	*pixels = (byte*)malloc(totalSize);

	// read the bitmap file 
	byte *currentRowPointer = *pixels + ((*height-1) * unpaddedRowSize);
	for (int i = 0; i < *height; i++)
	{
		fseek(imageFile, dataOffset+(i*paddedRowSize), SEEK_SET);
		fread(currentRowPointer, 1, unpaddedRowSize, imageFile);
		currentRowPointer -= unpaddedRowSize;
		for (int j = 0; j < paddedRowSize; j++){
			printf("%d ", (int)currentRowPointer[j]);
		}
		printf("\n");
	}
	
	// cloase the bitmap file
	fclose(imageFile);
}
#endif

int main(int argc, char const *argv[]) {
  if (argc < 1) {
    fprintf(stderr, "./main [config file]\n");
    exit(EXIT_FAILURE);
  }

  const char *config_filename = argv[1];

  load_config(config_filename);

  print_config();

  float *d_mesh, *d_tmp;
  lbm_file_entry_t *out, *d_out;
  cudaError_t cudaerr;

  // Host and Device memory allocation
  cudaMalloc(&d_tmp, MESH_WIDTH * MESH_HEIGHT * DIRECTIONS * sizeof(float));
  cudaMalloc(&d_mesh, MESH_WIDTH * MESH_HEIGHT * DIRECTIONS * sizeof(float));
  cudaMalloc(&d_out, MESH_WIDTH * MESH_HEIGHT * sizeof(lbm_file_entry_t));
  out = (lbm_file_entry_t *)malloc(MESH_WIDTH * MESH_HEIGHT *
                                   sizeof(lbm_file_entry_t));

  // Open output file
  FILE *output_file = open_output_file();

  // Write the header
  write_output_file_header(output_file);

  // Compute dimensions of the grid
  dim3 blockSize = dim3(8, 8, 1);
  int bx = (MESH_WIDTH + blockSize.x - 1) / blockSize.x;
  int by = (MESH_HEIGHT + blockSize.y - 1) / blockSize.y;
  dim3 gridSize = dim3(bx, by, 1);

  // initialisation kernel for mesh & tmp
  init_state_kernel<<<gridSize, blockSize>>>(d_mesh, MESH_WIDTH, MESH_HEIGHT,
                                             lbm_gbl_config);
  init_state_kernel<<<gridSize, blockSize>>>(d_tmp, MESH_WIDTH, MESH_HEIGHT,
                                             lbm_gbl_config);

  // output computation kernel tmp => out
  kernel_macroscopic_mesh<<<gridSize, blockSize>>>(d_out, d_tmp, MESH_WIDTH,
                                                   MESH_HEIGHT, lbm_gbl_config);

  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
           cudaGetErrorString(cudaerr));

  // loat out from the device to the host
  cudaMemcpy(out, d_out, MESH_WIDTH * MESH_HEIGHT * sizeof(lbm_file_entry_t),
             cudaMemcpyDeviceToHost);

  // save out in the output file
  save_output_file(output_file, out, MESH_WIDTH, MESH_HEIGHT);

  // for i from 0 to ITERATIONS_NB with step of FRAME_CPT do
  for (int i = 0; i < (ITERATIONS / WRITE_STEP_INTERVAL); i++) {
    fprintf(stderr, "%d/%d\n", i + 1, ITERATIONS / WRITE_STEP_INTERVAL);

    // for j from 0 to FRAME_CPT do
    for (int j = 0; j < WRITE_STEP_INTERVAL; j++) {
      // special cells kernel on mesh
      kernel_special_cells<<<gridSize, blockSize>>>(
          d_mesh, MESH_WIDTH, MESH_HEIGHT, lbm_gbl_config);

      // collision kernel on tmp and mesh
      kernel_collision<<<gridSize, blockSize>>>(d_tmp, d_mesh, MESH_WIDTH,
                                                MESH_HEIGHT, lbm_gbl_config);

      // propagation on mesh and tmp
      kernel_propagation<<<gridSize, blockSize>>>(d_mesh, d_tmp, MESH_WIDTH,
                                                  MESH_HEIGHT, lbm_gbl_config);
    }

    // output computation kernel tmp => out
    kernel_macroscopic_mesh<<<gridSize, blockSize>>>(
        d_out, d_tmp, MESH_WIDTH, MESH_HEIGHT, lbm_gbl_config);

    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n",
             cudaGetErrorString(cudaerr));

    // loat out from the device to the host
    cudaMemcpy(out, d_out, MESH_WIDTH * MESH_HEIGHT * sizeof(lbm_file_entry_t),
               cudaMemcpyDeviceToHost);

    // save out in the output file
    save_output_file(output_file, out, MESH_WIDTH, MESH_HEIGHT);
  }

  close_output_file(output_file);

  // free d_tmp
  cudaFree(d_tmp);
  // free d_out
  cudaFree(d_out);
  // free mesh
  cudaFree(d_mesh);
  // free config
  config_cleanup();
  // free out
  free(out);

  return 0;
}

#if 0
/*******************  FUNCTION  *********************/

/* write header in the output file */
void write_output_file_header(FILE* output_file)
{
	// setup header values
	lbm_file_header_t header;
	header.magick      = 0x12345;
	header.mesh_height = MESH_HEIGHT;
	header.mesh_width  = MESH_WIDTH;
	header.lines       = 1;

	// write file
	fwrite(&header, sizeof(header), 1, output_file);
}

/* open the output file */
FILE* open_output_file()
{
	// check if a filename is set
	if (RESULT_FILENAME == NULL)
		return NULL;

	// open result file
	FILE *output_file = fopen(RESULT_FILENAME, "w");

	// errors
	if (output_file == NULL){
		perror(RESULT_FILENAME);
		abort();
	}

	// write header
	write_output_file_header(output_file);

	return output_file;
}

/* close the output file */
void close_open_file(FILE* output_file)
{
	fclose(output_file);
}

/** Sauvegarde du maillage dans le fichier de sortie **/

__global__
void save_kernel(float *mesh, lbm_file_entry_t *outs, int width, int height)
{
	// get thread column 
  	const int column = blockIdx.x*blockDim.x + threadIdx.x;
	// get thread row
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	// get index of the thread
	const int i = row*width + column;
	// test if the thread is in the mesh
	if ((column >= width) || (row >= height)) return;

	const Vector direction_matrix[DIRECTIONS] = {
		{+0.0f,+0.0f},
		{+1.0f,+0.0f}, {+0.0f,+1.0f}, {-1.0f,+0.0f}, {+0.0f,-1.0f},
		{+1.0f,+1.0f}, {-1.0f,+1.0f}, {-1.0f,-1.0f}, {+1.0f,-1.0f}
	};

	float density = 0.0f;
	for (int k = 0; k < DIRECTIONS; k++)
		density += mesh[(i * DIRECTIONS) + k];

	float div = 1.0/density;
	Vector v = {0.0f, 0.0f};
	for (int d = 0; d < DIMENSIONS; d++){
		float tmp = 0.0f;
		for (int k = 0; k < DIRECTIONS; k++){
			tmp += mesh[(i * DIRECTIONS) + k] * direction_matrix[k][d];
		}
		v[d] = tmp * div;
	}

	float norm = sqrtf(v[0]*v[0] + v[1]*v[1]);

	outs[i].density = density;
	outs[i].v = norm;
}

/* save the mesh in the output file */
void save_frame(FILE* output_file, const Mesh* mesh)
{
	dim3 blockSize = dim3(8, 8, 1);

	int bx = (mesh->width + blockSize.x - 1) / blockSize.x;
	int by = (mesh->height + blockSize.y - 1) / blockSize.y;

	dim3 gridSize = dim3(bx, by, 1);

	save_kernel<<<gridSize, blockSize>>>(mesh->cells, mesh->outs, mesh->width, mesh->height);

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));

	for (int i = 0; i < mesh->width; i++){
		for (int j = 0; j < mesh->height; j++){
			fwrite(&mesh->outs[j * mesh->width + i], sizeof(lbm_file_entry_t), 1, output_file);
		}
	}
}



/** initialisation du maillage **/
__global__
void init_kernel(float* mesh, int width, int height, float inflow_max_velocity, int obstacle_r, int obstacle_column, int obstacle_row)
{
	// get thread column 
  	const int column = blockIdx.x*blockDim.x + threadIdx.x;
	// get thread row
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	// get index of the thread
	const int i = row*width + column;
	// test if the thread is in the mesh
	if ((column >= width) || (row >= height)) return;

	const Vector direction_matrix[DIRECTIONS] = {
		{+0.0f,+0.0f},
		{+1.0f,+0.0f}, {+0.0f,+1.0f}, {-1.0f,+0.0f}, {+0.0f,-1.0f},
		{+1.0f,+1.0f}, {-1.0f,+1.0f}, {-1.0f,-1.0f}, {+1.0f,-1.0f}
	};

	const float equil_weight[DIRECTIONS] = {
		4.0f/9.0f ,
		1.0f/9.0f , 1.0f/9.0f , 1.0f/9.0f , 1.0f/9.0f,
		1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
	};

	// setup_init_state_global_poiseuille_profile
	const Vector v = {4.0f * inflow_max_velocity / (height*height) * (height*row - row*row), 0.0f};

	for (int k = 0; k < DIRECTIONS; k++){
		float v2 = v[0] * v[0];
		float p = direction_matrix[k][0] * v[0];
		float p2 = p * p;
		mesh[(i * DIRECTIONS) + k] = (1.0f + (3.0f * p) + (4.5f * p2) - (1.5f * v2)) * equil_weight[k];
	}

	// setup_init_state_border
	if (row == 0){
		for (int k = 0; k < DIRECTIONS; k++){
			mesh[(i * DIRECTIONS) + k] = equil_weight[k];
		}
	}

	if (row == height-1){
		for (int k = 0; k < DIRECTIONS; k++){
			mesh[(i * DIRECTIONS) + k] = equil_weight[k];
		}
	}

	// setup_init_state_circle_obstacle
    if (((column-obstacle_column) * (column-obstacle_column)) + ((row-obstacle_row) * (row-obstacle_row)) <= (obstacle_r * obstacle_r)){
		for (int k = 0; k < DIRECTIONS ; k++){
			mesh[(i * DIRECTIONS) + k] = equil_weight[k];
		}
    }
}

void setup_init_state(Mesh* mesh)
{
	dim3 blockSize = dim3(8, 8, 1);

	int bx = (mesh->width + blockSize.x - 1) / blockSize.x;
	int by = (mesh->height + blockSize.y - 1) / blockSize.y;

	dim3 gridSize = dim3(bx, by, 1);

	init_kernel<<<gridSize, blockSize>>>(mesh->cells, mesh->width, mesh->height, INFLOW_MAX_VELOCITY, OBSTACLE_R, OBSTACLE_X, OBSTACLE_Y);

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
}


/** Traitement des cellules speciales **/

__global__
void special_cells_kernel(float *mesh, int width, int height, float inflow_max_velocity, int obstacle_r, int obstacle_column, int obstacle_row)
{
	// get thread column 
  	const int column = blockIdx.x*blockDim.x + threadIdx.x;
	// get thread row
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	// get index of the thread
	const int i = row*width + column;
	// test if the thread is in the mesh
	if ((column >= width) || (row >= height)) return;

	const int opposite_of[DIRECTIONS] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

	// compute_inflow_zou_he_poiseuille_distr
	if (column == 0 && row != 0 && row != height-1){
		float v = 4.0f * inflow_max_velocity / (height*height) * (height*row - row*row);
		float density = (mesh[i * DIRECTIONS + 0] + mesh[i * DIRECTIONS + 2] + mesh[i * DIRECTIONS + 4] + 2.0f * ( mesh[i * DIRECTIONS + 3] + mesh[i * DIRECTIONS + 6] + mesh[i * DIRECTIONS + 7] )) * (1.0f - v);
		float a = 0.166667f * (density * v);
		mesh[i * DIRECTIONS + 1] = mesh[i * DIRECTIONS + 3];
		mesh[i * DIRECTIONS + 5] = mesh[i * DIRECTIONS + 7] - 0.5f * (mesh[i * DIRECTIONS + 2] - mesh[i * DIRECTIONS + 4]) + a;
		mesh[i * DIRECTIONS + 8] = mesh[i * DIRECTIONS + 6] + 0.5f * (mesh[i * DIRECTIONS + 2] - mesh[i * DIRECTIONS + 4]) + a;
	}

	// compute_outflow_zou_he_const_density
	if (column == (width-1) && row != 0 && row != height-1){
		float v = (mesh[(i * DIRECTIONS) + 0] + mesh[(i * DIRECTIONS) + 2] + mesh[(i * DIRECTIONS) + 4] + 2 * (mesh[(i * DIRECTIONS) + 1] + mesh[(i * DIRECTIONS) + 5] + mesh[(i * DIRECTIONS) + 8])) - 1.0f;
		float a = 0.166667f * v;
		mesh[(i * DIRECTIONS) + 3] = mesh[(i * DIRECTIONS) + 1] - 0.66667f * v;
		mesh[(i * DIRECTIONS) + 7] = mesh[(i * DIRECTIONS) + 5] + 0.5f * (mesh[(i * DIRECTIONS) + 2] - mesh[(i * DIRECTIONS) + 4]) - a;
		mesh[(i * DIRECTIONS) + 6] = mesh[(i * DIRECTIONS) + 8] + 0.5f * (mesh[(i * DIRECTIONS) + 4] - mesh[(i * DIRECTIONS) + 2]) - a;
	}

	// compute_bounce_back
	if (((column-obstacle_column) * (column-obstacle_column)) + ((row-obstacle_row) * (row-obstacle_row)) <= (obstacle_r * obstacle_r)){
		for (int k = 0; k < DIRECTIONS ; k++)
			mesh[(i * DIRECTIONS) + k] = mesh[(i * DIRECTIONS) + opposite_of[k]];
	}

	// bottom wall
	if (row == 0){
		for (int k = 0; k < DIRECTIONS ; k++)
			mesh[(i * DIRECTIONS) + k] = mesh[(i * DIRECTIONS) + opposite_of[k]];
	}

	// upper wall
	if (row == height-1){
		for (int k = 0; k < DIRECTIONS ; k++)
			mesh[(i * DIRECTIONS) + k] = mesh[(i * DIRECTIONS) + opposite_of[k]];		
	}
}

void special_cells(Mesh* mesh)
{
	dim3 blockSize = dim3(8, 8, 1);

	int bx = (mesh->width + blockSize.x - 1) / blockSize.x;
	int by = (mesh->height + blockSize.y - 1) / blockSize.y;

	dim3 gridSize = dim3(bx, by, 1);

	special_cells_kernel<<<gridSize, blockSize>>>(mesh->cells, mesh->width, mesh->height, INFLOW_MAX_VELOCITY, OBSTACLE_R, OBSTACLE_X, OBSTACLE_Y);

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
}

/** Collision entre les cellules **/
__global__
void collision_kernel(float *cell_out, float *cell_in, int width, int height, float relax_paremeter)
{
	// get thread column 
  	const int column = blockIdx.x*blockDim.x + threadIdx.x;
	// get thread row
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	// get index of the thread
	const int i = row*width + column;
	// test if the thread is in the mesh
	if (!(column > 0 && column < width-1) || !(row > 0 && row < height-1)) return;

	const Vector direction_matrix[DIRECTIONS] = {
		{+0.0f,+0.0f},
		{+1.0f,+0.0f}, {+0.0f,+1.0f}, {-1.0f,+0.0f}, {+0.0f,-1.0f},
		{+1.0f,+1.0f}, {-1.0f,+1.0f}, {-1.0f,-1.0f}, {+1.0f,-1.0f}
	};

	const double equil_weight[DIRECTIONS] = {
		4.0/9.0f ,
		1.0f/9.0f , 1.0f/9.0f , 1.0f/9.0f , 1.0f/9.0f,
		1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
	};


	float density = 0.0f;
	float feq = 0.0f;
	Vector v = {0.0f, 0.0f};

	// get density of the input cell
	for (int k = 0; k < DIRECTIONS; k++){
		density += cell_in[(i * DIRECTIONS) + k];
	}

	// get velocity of the input cell
	float div = 1.0f/density;
	
	for (int d = 0; d < DIMENSIONS; d++){
		float tmp = 0.0;

		for (int k = 0; k < DIRECTIONS; k++){
			tmp += cell_in[(i * DIRECTIONS) + k] * direction_matrix[k][d];
		}

		v[d] = tmp * div;
	}

	for (int k = 0; k < DIRECTIONS; k++){
		// compute equilibrium profile
		float v2 = v[0] * v[0] + v[1] * v[1];
		float p = direction_matrix[k][0] * v[0] + direction_matrix[k][1] * v[1];
		float p2 = p * p;
		feq = 1.0f + (3.0f * p) + (4.5f * p2) - (1.5f * v2);
		feq *= equil_weight[k] * density;

		cell_out[(i * DIRECTIONS) + k] = cell_in[(i * DIRECTIONS) + k] - relax_paremeter * (cell_in[(i * DIRECTIONS) + k] - feq);
	}
}

void collision(Mesh* mesh_out, Mesh* mesh_in)
{
	dim3 blockSize = dim3(8, 8, 1);

	int bx = (mesh_in->width + blockSize.x - 1) / blockSize.x;
	int by = (mesh_in->height + blockSize.y - 1) / blockSize.y;

	dim3 gridSize = dim3(bx, by, 1);

	collision_kernel<<<gridSize, blockSize>>>(mesh_out->cells, mesh_in->cells, mesh_in->width, mesh_in->height, RELAX_PARAMETER);

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
}


/** Propagation des cellules **/
__global__
void propagation_kernel(float *cell_out, float *cell_in, int width, int height)
{
	// get thread column 
  	const int column = blockIdx.x*blockDim.x + threadIdx.x;
	// get thread row
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	// get index of the thread
	const int i = row * width + column;
	// test if the thread is in the mesh
	if ((column >= width) || (row >= height)) return;

	const int int_direction_matrix[DIRECTIONS][2] = {
		{+0,+0},
		{+1,+0}, {+0,+1}, {-1,+0}, {+0,-1},
		{+1,+1}, {-1,+1}, {-1,-1}, {+1,-1}
	};

	for (int k = 0; k < DIRECTIONS; k++){
		int cc = column + int_direction_matrix[k][0];
		int rr = row + int_direction_matrix[k][1];

		if ((cc >= 0 && cc < width) && (rr >= 0 && rr < height)){
			int j = rr * width + cc;
			cell_out[(j * DIRECTIONS) + k] = cell_in[(i * DIRECTIONS) + k];
		}
	}
}

void propagation(Mesh* mesh_out, Mesh* mesh_in)
{
	dim3 blockSize = dim3(8, 8, 1);

	int bx = (mesh_in->width + blockSize.x - 1) / blockSize.x;
	int by = (mesh_in->height + blockSize.y - 1) / blockSize.y;

	dim3 gridSize = dim3(bx, by, 1);

	propagation_kernel<<<gridSize, blockSize>>>(mesh_out->cells, mesh_in->cells, mesh_in->width, mesh_in->height);

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
}

int main(int argc, char const *argv[])
{
	Mesh mesh, tmp;
	const char* config_filename = NULL;

	//get config filename
	if (argc >= 2)
		config_filename = argv[1];
	else
 		config_filename = "config.txt";

 	load_config(config_filename);

 	Mesh_init(&tmp, MESH_WIDTH, MESH_HEIGHT);
 	Mesh_init(&mesh, MESH_WIDTH, MESH_HEIGHT);

 	FILE *output_file = open_output_file();

 	setup_init_state(&tmp);
 	setup_init_state(&mesh);

 	save_frame(output_file, &tmp);
 	
 	for (int i = 0; i < (ITERATIONS / WRITE_STEP_INTERVAL); i++){
 		fprintf(stderr, "%d/%d\n", i+1, ITERATIONS / WRITE_STEP_INTERVAL);

 		for (int j = 0; j < WRITE_STEP_INTERVAL; j++){
 			special_cells(&mesh);

	  		collision(&tmp, &mesh);

	 		propagation(&mesh, &tmp);
 		}

 		save_frame(output_file, &tmp);
 	}

 	/*
 	for (int i = 1; i < ITERATIONS; i++){
 		special_cells(&mesh);

 		collision(&tmp, &mesh);

 		propagation(&mesh, &tmp);

 		if (i % WRITE_STEP_INTERVAL == 0)
 			save_frame(&output_file, &tmp);
 	}
 	*/

 	Mesh_release(&tmp);
 	Mesh_release(&mesh);

  	//close_open_file(output_file);

	return 0;
}

#endif