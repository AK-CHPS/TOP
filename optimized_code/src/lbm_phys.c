/********************  HEADERS  *********************/
#include <assert.h>
#include <stdlib.h>
#include "./header/lbm_config.h"
#include "./header/lbm_struct.h"
#include "./header/lbm_phys.h"
#include "./header/lbm_comm.h"

/********************** CONSTS **********************/
/**
 * Definitions des 9 vecteurs de base utilisé pour discrétiser les directions sur chaque mailles.
**/
#if DIRECTIONS == 9 && DIMENSIONS == 2
const Vector direction_matrix[DIRECTIONS] = {
	{+0.0,+0.0},
	{+1.0,+0.0}, {+0.0,+1.0}, {-1.0,+0.0}, {+0.0,-1.0},
	{+1.0,+1.0}, {-1.0,+1.0}, {-1.0,-1.0}, {+1.0,-1.0}
};

const int int_direction_matrix[DIRECTIONS][2] = {
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
#else
#error Need to defined adapted direction matrix.
#endif

/********************** CONSTS **********************/
/**
 * Poids utilisé pour compenser les différentes de longueur des 9 vecteurs directions.
**/
#if DIRECTIONS == 9
const double equil_weight[DIRECTIONS] = {
	4.0/9.0 ,
	1.0/9.0 , 1.0/9.0 , 1.0/9.0 , 1.0/9.0,
	1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

//opposite directions, for bounce back implementation
const int opposite_of[DIRECTIONS] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
#else
#error Need to defined adapted equibirium distribution function
#endif

/*******************  FUNCTION  *********************/
/**
 * Renvoie le résultat du produit des deux vecteurs passés en paramêtre.
**/
double get_vect_norme_2(const Vector vect1,const Vector vect2)
{
	//vars
	int k;
	double res = 0.0;

	//loop on dimensions
	for ( k = 0 ; k < DIMENSIONS ; k++)
		res += vect1[k] * vect2[k];

	return res;
}

/*******************  FUNCTION  *********************/
/**
 * Calcule la densité macroscopiques de la cellule en sommant ses DIRECTIONS
 * densités microscopiques.
**/
double get_cell_density(const lbm_mesh_cell_t cell)
{
	//vars
	int k;
	double res = 0.0;

	//loop on directions
	for(k = 0 ; k < DIRECTIONS ; k++)
		res += cell[k];

	return res;
}

/*******************  FUNCTION  *********************/
/**
 * Calcule la vitesse macroscopiques de la cellule en sommant ses DIRECTIONS
 * densités microscopiques.
 * @param cell_density Densité macroscopique de la cellules.
**/
void get_cell_velocity(Vector v, const lbm_mesh_cell_t cell, const  double cell_density)
{
	//vars
	int k,d;
    double temp;
    double div = 1.0/cell_density;

	//loop on all dimensions
	for (d = 0 ; d < DIMENSIONS ; d++)
	{
		//reset value
		temp = 0.0;

		//sum all directions
		for ( k = 0 ; k < DIRECTIONS ; k++){
			temp += cell[k] * direction_matrix[k][d];
		}
		
		//normalize
		v[d] = temp * div;
	}
}

/*******************  FUNCTION  *********************/
/**
 * Calcule le profile de densité microscopique à l'équilibre pour une direction donnée.
 * C'est ici qu'intervient un dérivé de navier-stokes.
 * @param velocity Vitesse macroscopique du fluide sur la maille.
 * @param density Densité macroscopique du fluide sur la maille.
 * @param direction Direction pour laquelle calculer l'état d'équilibre.
**/
double compute_equilibrium_profile(Vector velocity, const double density, const int direction)
{
	//vars
	double p;
	double p2;
	double feq;
	double v2;

	v2 = get_vect_norme_2(velocity,velocity);

	//calc e_i * v_i / c
	p = get_vect_norme_2(direction_matrix[direction],velocity);

	p2 = p * p;

	//terms without density and direction weight
	feq = 1.0 + (3.0 * p) + (4.5 * p2) - (1.5 * v2);

	//mult all by density and direction weight
	feq *= equil_weight[direction] * density;

	return feq;
}


/*******************  FUNCTION  *********************/
/**
 * Calcule le vecteur de collision entre les fluides de chacune des directions.
**/
void compute_cell_collision(lbm_mesh_cell_t cell_out, const lbm_mesh_cell_t cell_in)
{
	//vars
	int k;
	double density, feq;
	Vector v;

	//compute macroscopic values
	density = get_cell_density(cell_in);
	get_cell_velocity(v,cell_in,density);

	//loop on microscopic directions
	for( k = 0 ; k < DIRECTIONS ; k++)
	{
		//compute f at equilibr.
		feq = compute_equilibrium_profile(v, density, k);
		//compute f out
		cell_out[k] = cell_in[k] - RELAX_PARAMETER * (cell_in[k] - feq);
	} 
}

/*******************  FUNCTION  *********************/
/**
 * Applique une reflexion sur les différentes directions pour simuler la présence d'un solide.
**/
void compute_bounce_back(lbm_mesh_cell_t cell)
{
	//vars
	int k;

	//compute bounce back
	for ( k = 0; k < DIRECTIONS ; k++)
		cell[k] = cell[opposite_of[k]];
}

/*******************  FUNCTION  *********************/
/**
 * Applique la méthode de Zou/He pour simler un fluidre entrant dans le domain de gauche vers la droite sur une
 * interface verticale. Le profile de vitesse du fluide entrant suit une distribution de poiseuille.
 * @param mesh Maillage considéré (surtout pour avoir la hauteur.)
 * @param cell Maille à mettre à jour.
 * @param id_y Position en y de la cellule pour savoir comment calculer la vitesse de poiseuille.
**/
void compute_inflow_zou_he_poiseuille_distr( const Mesh *mesh, lbm_mesh_cell_t cell,int id_y)
{
	//vars
	double v;
	double density;

	//errors
	#if DIRECTIONS != 9
	#error Implemented only for 9 directions
	#endif

	//set macroscopic fluide info
	//poiseuille distr on X and null on Y
	//we just want the norm, so v = v_x
	v = mesh->poiseuille[id_y];
	//v = helper_compute_poiseuille(id_y, mesh->height);

	//compute rho from u and inner flow on surface
	density = (cell[0] + cell[2] + cell[4] + 2 * ( cell[3] + cell[6] + cell[7] )) * (1.0 - v);

	//now compute unknown microscopic values
	double a = 0.166667 * (density * v);
	cell[1] = cell[3];
	cell[5] = cell[7] - 0.5 * (cell[2] - cell[4]) + a;
	cell[8] = cell[6] + 0.5 * (cell[2] - cell[4]) + a;

	//no need to copy already known one as the value will be "loss" in the wall at propagatation time
}

/*******************  FUNCTION  *********************/
/**
 * Applique la méthode de Zou/He pour simuler un fluide sortant du domaine de gauche vers la droite sur une
 * interface verticale. La condition appliquée pour construire l'équation est le maintient d'un gradiant de densité
 * nulle à l'interface.
 * @param mesh Maillage considéré (surtout pour avoir la hauteur.)
 * @param cell Maille à mettre à jour
 * @param id_y Position en y de la cellule pour savoir comment calculer la vitesse de poiseuille.
**/
void compute_outflow_zou_he_const_density(lbm_mesh_cell_t cell)
{
	//vars
	double v;

	//errors
	#if DIRECTIONS != 9
	#error Implemented only for 9 directions
	#endif

	//compute macroscopic v depeding on inner flow going onto the wall
	v = (cell[0] + cell[2] + cell[4] + 2 * (cell[1] + cell[5] + cell[8])) - 1.0;

	//now can compute unknown microscopic values
	double a = 0.166667 * v;
	cell[3] = cell[1] - 0.66667 * v;
	cell[7] = cell[5] + 0.5 * (cell[2] - cell[4]) - a;
	cell[6] = cell[8] + 0.5 * (cell[4] - cell[2]) - a;
}

/*******************  FUNCTION  *********************/
/**
 * Applique les actions spéciales liées aux conditions de bords ou aux réflexions sur l'obstacle.
**/
void special_cells(Mesh * mesh)
{
	for(int i = 0; i < mesh->left_in_cpt; i++){
		compute_inflow_zou_he_poiseuille_distr(mesh, mesh->left_in_cells[i], i);		
	}

	for(int i = 0; i < mesh->right_out_cpt; i++){
		compute_outflow_zou_he_const_density(mesh->right_out_cells[i]);
	}

	for(int i = 0; i < mesh->bounce_cpt; i++){
		compute_bounce_back(mesh->bounce_cells[i]);
	}
}

/*******************  FUNCTION  *********************/
/**
 * Calcule les collisions sur chacune des cellules.
 * @param mesh Maillage sur lequel appliquer le calcul.
**/
void collision(Mesh * mesh_out, const Mesh * mesh_in)
{
	//vars
	int i,j;

	//loop on all inner cells
	for( i = 1; i < mesh_in->width - 1; i++)
		for( j = 1; j < mesh_in->height - 1; j++)
			compute_cell_collision(Mesh_get_cell(mesh_out, i, j),Mesh_get_cell(mesh_in, i, j));
}

/*******************  FUNCTION  *********************/
/**
 * Propagation des densités vers les mailles voisines.
 * @param mesh_out Maillage de sortie.
 * @param mesh_in Maillage d'entrée (ne doivent pas être les mêmes).
**/
void propagation(Mesh * mesh_out, const Mesh * mesh_in)
{


	//vars
	int i,j,k;
	int ii,jj;

	//loop on all cells
	for ( i = 0 ; i < mesh_out->width; i++){
		for ( j = 0 ; j < mesh_out->height ; j++){
			//for all direction
			for ( k  = 0 ; k < DIRECTIONS ; k++){
				//compute destination point
				ii = (i + direction_matrix[k][0]);
				jj = (j + direction_matrix[k][1]);
				//propagate to neighboor nodes
				if ((ii >= 0 && ii < mesh_out->width) && (jj >= 0 && jj < mesh_out->height))
					Mesh_get_cell(mesh_out, ii, jj)[k] = Mesh_get_cell(mesh_in, i, j)[k];
			}
		}
	}
}


void collision_and_propagation(Mesh * mesh,  Mesh * temp)
{
	//vars
	int i, j, k, z;
	int ii,jj;
    int width = mesh->width;
    int height = mesh->height;
    double *cells_temp = temp->cells;
    double *cells_mesh = mesh->cells;

	/* warm up */
	for( i = 1; i < 3; i++)
		for( j = 1; j < height - 1; j++)
			compute_cell_collision(Mesh_get_cell(temp, i, j),Mesh_get_cell(mesh, i, j));

	//loop on all inner cells
	for (i = 1; i < width - 3; i ++){
		for ( j = 1 ; j < height-1 ; j ++){
			compute_cell_collision(Mesh_get_cell(temp, (i+2), j),Mesh_get_cell(mesh, (i+2), j));

			for ( k  = 0 ; k < DIRECTIONS ; k++){
				ii = (i + int_direction_matrix[k][0]);
				jj = (j + int_direction_matrix[k][1]);

				cells_mesh[(ii * height + jj) * DIRECTIONS + k] = cells_temp[(i * height + j) * DIRECTIONS + k];
			}
		}
	}

	/* cool down */
	for (i = width - 3; i < width - 1; i ++){
		for ( j = 1 ; j < height-1 ; j ++){
			for ( k  = 0 ; k < DIRECTIONS ; k++)
			{
				ii = (i + direction_matrix[k][0]);
				jj = (j + direction_matrix[k][1]);

				cells_mesh[(ii * height + jj) * DIRECTIONS + k] = cells_temp[(i * height + j) * DIRECTIONS + k];
			}
		}
	}

    for(j = 0; j < height-1; j++){
		// left mesh cells
		i = 0;
	    for(z = 0; z < 6; z++){
	        k = right_direction_matrix[z];

	        ii = (i + int_direction_matrix[k][0]);
			jj = (j + int_direction_matrix[k][1]);

		    cells_mesh[(ii * height + jj) * DIRECTIONS + k] = cells_temp[(i * height + j) * DIRECTIONS + k];
	    }
	} 

	for(j = 0; j < height-1; j++){
		// right mesh cells
		i = width-1;
	    for(z = 0; z < 6; z++){
		        k = left_direction_matrix[z];

		        ii = (i + int_direction_matrix[k][0]);
				jj = (j + int_direction_matrix[k][1]);
	
	            cells_mesh[(ii * height + jj) * DIRECTIONS + k] = cells_temp[(i * height + j) * DIRECTIONS + k];
	    }
	}
}
