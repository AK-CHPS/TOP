#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <string.h>

#define WIDTH 			202
#define HEIGHT 			162
#define DIRECTIONS 		9
#define ITERATIONS 		16000

#define get_mesh_cell(MESH, X, Y) (&MESH[(X * HEIGHT + Y) * DIRECTIONS])


const int direction_matrix[DIRECTIONS][2] = {
	{+0,+0},
	{+1,+0}, {+0,+1}, {-1,+0}, {+0,-1},
	{+1,+1}, {-1,+1}, {-1,-1}, {+1,-1}
};

const int right_direction_matrix[6] = {
	0, 1, 2, 4, 5, 8
};

const int left_direction_matrix[6] = {
    0, 2, 3, 4, 6, 7
};


void initialisation(double *mesh, double *temp, double *other)
{
	for(int i = 0; i < WIDTH; i++){
		for(int j = 0; j < HEIGHT; j++){
			for(int k = 0; k < DIRECTIONS; k++){
				get_mesh_cell(mesh ,i, j)[k] = rand()%100;
				get_mesh_cell(temp ,i, j)[k] = rand()%100;
				get_mesh_cell(other ,i, j)[k] = get_mesh_cell(temp ,i, j)[k];
			}
		}
	}
}

void compute_special_cells(int i, int j, double *mesh)
{
	if((i == 1 || i == WIDTH-2 || j == 1 || j == HEIGHT-2) && (i != 0 && i != WIDTH-1 && j != 0 && j != HEIGHT-1)){
		for(int k = 0; k < DIRECTIONS; k++)
			get_mesh_cell(mesh ,i, j)[k] = (i+1)/(j+1);	
	}
}

void special_cells(double* mesh)
{
	for(int i = 0; i < WIDTH; i++){
		for(int j = 0; j < HEIGHT; j++){
			if((i == 1 || i == WIDTH-2 || j == 1 || j == HEIGHT-2) && (i != 0 && i != WIDTH-1 && j != 0 && j != HEIGHT-1)){
				for(int k = 0; k < DIRECTIONS; k++)
					get_mesh_cell(mesh ,i, j)[k] = (i+1)/(j+1);	
			}	
		}
	}
}

void compute_collision(int i, int j, double *mesh_out, const double *mesh_in)
{
	// collision
	if((i >= 1 && i < WIDTH-1) && (j >= 1 && j < HEIGHT-1))
	{
		for(int k = 0; k < DIRECTIONS; k++){
			get_mesh_cell(mesh_out, i, j)[k] += (0.00001*2.3)/10.0 * get_mesh_cell(mesh_in, i, j)[k];		
		}
	}
}

void compute_cell_collision(double* cell_out, const double *cell_in)
{
	for(int k = 0; k < DIRECTIONS; k++){
		cell_out[k] += (0.00001*2.3)/10.0 * cell_in[k];		
	}
}


void collision(double *mesh_out, const double *mesh_in)
{
	//printf("\tcollision thread %d\n", omp_get_thread_num());

	for(int i = 1; i < WIDTH - 1; i++){
		for(int j = 1; j < HEIGHT - 1; j++){
			compute_cell_collision(get_mesh_cell(mesh_out, i, j), get_mesh_cell(mesh_in, i, j));
		}
	}
}

void compute_propagation(int i, int j, double *mesh_out, const double *mesh_in)
{
	for(int k = 0; k < DIRECTIONS; k++){
		int ii = i + direction_matrix[k][0];
		int jj = j + direction_matrix[k][1];	
			
		if(ii >= 0 && ii < WIDTH  && jj >= 0 && jj < HEIGHT)
			get_mesh_cell(mesh_out, ii, jj)[k] = get_mesh_cell(mesh_in, i, j)[k];
	}
}

void propagation(double *mesh_out, const double *mesh_in)
{
	//printf("\tpropagation thread %d\n", omp_get_thread_num());

	for(int i = 0; i < WIDTH; i++){
		for(int j = 0; j < HEIGHT; j++){
			compute_propagation(i, j, mesh_out, mesh_in);
		}
	}
}

void save_mesh(const double *mesh)
{
	FILE *f = fopen("test.raw", "wr");

	fwrite(mesh, WIDTH * HEIGHT * DIRECTIONS, sizeof(double), f);
	
	fclose(f);
}

int main(int argc, char const *argv[])
{

	srand(1);
	double *temp = calloc(WIDTH * HEIGHT * DIRECTIONS, sizeof(double));
	double *mesh = calloc(WIDTH * HEIGHT * DIRECTIONS, sizeof(double));
	double *other = calloc(WIDTH * HEIGHT * DIRECTIONS, sizeof(double));
	int *flag = calloc(WIDTH * HEIGHT, sizeof(int));
	omp_lock_t lock1, lock2;

	initialisation(mesh, temp, other);

	omp_init_lock(&lock1);
	omp_init_lock(&lock2);

	omp_set_lock(&lock2);

	#pragma omp parallel num_threads(2)
	{
		for(int it = 0; it < ITERATIONS; it++){

			if(it % 500 == 0 && omp_get_thread_num() == 0)
				printf("itération %d/%d\n", it, ITERATIONS);
			
			#pragma omp single
			for(int i = 0; i < 2; i++){
				for(int j = 0; j < HEIGHT; j++){
					compute_special_cells(i, j, mesh);
					compute_collision(i, j, temp, mesh);
				}
			}

			for(int i = 2; i < WIDTH; i++){
				#pragma omp single nowait
				{
					omp_set_lock(&lock1);
					for(int j = 0; j < HEIGHT; j++){
						compute_special_cells(i, j, mesh);
						compute_collision(i, j, temp, mesh);
					}
					omp_unset_lock(&lock2);
				}

				#pragma omp single nowait
				{
					omp_set_lock(&lock2);
					omp_unset_lock(&lock1);
					for(int j = 0; j < HEIGHT; j++){
						compute_propagation(i-2, j, mesh ,temp);
					}
				}
			}

			#pragma omp single
			for(int i = WIDTH-3; i < WIDTH; i++){
				for(int j = 0; j < HEIGHT; j++){
					compute_propagation(i, j, mesh ,temp);
				}
			}
		}
	}

	omp_destroy_lock(&lock1);
	omp_destroy_lock(&lock2);

	// code d'origine
	/*
	for(int it = 0; it < ITERATIONS; it++){

		if(it % 500 == 0)
			printf("itération %d/%d\n", it, ITERATIONS);
			
		special_cells(mesh);

		collision(temp, mesh);

		propagation(mesh, temp);
	}
	*/



	save_mesh(mesh);

	free(mesh);
	free(temp);
	free(other);
	free(flag);

	return 0;
}