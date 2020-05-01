#ifndef LBM_PHYS_H
#define LBM_PHYS_H

/********************  HEADERS  *********************/
#include "lbm_struct.h"
#include "lbm_comm.h"

/********************** CONSTS **********************/
const int opposite_of[DIRECTIONS];
const double equil_weight[DIRECTIONS];
const Vector direction_matrix[DIRECTIONS];

double get_vect_norme_2(const Vector vect1,const Vector vect2);
double get_cell_density(const lbm_mesh_cell_t cell);
void get_cell_velocity(Vector v,const lbm_mesh_cell_t cell, double cell_density);

/*******************  FUNCTION  *********************/
//collistion
double compute_equilibrium_profile(Vector velocity,double density,int direction);
void compute_cell_collision(lbm_mesh_cell_t cell_out, const lbm_mesh_cell_t cell_in);

/*******************  FUNCTION  *********************/
//limit conditions
void compute_bounce_back(lbm_mesh_cell_t cell);
void compute_inflow_zou_he_poiseuille_distr( const Mesh *mesh, lbm_mesh_cell_t cell,int id_y);
void compute_outflow_zou_he_const_density(lbm_mesh_cell_t mesh);

/*******************  FUNCTION  *********************/
//main functions
void special_cells(Mesh * mesh, const lbm_comm_t * mesh_comm);
void collision(Mesh * mesh_out,const Mesh * mesh_in);
void propagation(Mesh * mesh_out,const Mesh * mesh_in);

#endif
