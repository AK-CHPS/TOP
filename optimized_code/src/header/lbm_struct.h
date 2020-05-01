#ifndef LBM_STRUCT_H
#define LBM_STRUCT_H

/********************  HEADERS  *********************/
#include <stdint.h>
#include <stdio.h>
#include "lbm_config.h"

/********************** TYPEDEF *********************/
/**
 * Une cellule est un tableau de DIRECTIONS doubles pour stoquer les
 * probabilités microscopiques (f_i).
**/
typedef double *lbm_mesh_cell_t;
/** Représentation d'un vecteur pour la manipulation des vitesses macroscopiques. **/
typedef double Vector[DIMENSIONS];


/********************  STRUCT  **********************/
/** Une entrée du fichier, avec les deux grandeurs macroscopiques. **/
typedef struct lbm_file_entry_s
{
	double v;
	double density;
} lbm_file_entry_t;

/********************  STRUCT  **********************/
/**
 * Definit un maillage pour le domaine local. Ce maillage contient une bordure d'une cellule
 * contenant les mailles fantômes.
**/
typedef struct Mesh
{
	/** Cellules du maillages (MESH_WIDTH * MESH_HEIGHT). **/
	lbm_mesh_cell_t cells;
	/** Largeur du maillage local (mailles fantome comprises). **/
	int width;
	/** Largeur du maillage local (mailles fantome comprises). **/
	int height;
	/** tableau de velocitees et de desitees **/
	lbm_file_entry_t *values;
	/** Cellules d'entrées **/
	lbm_mesh_cell_t* left_in_cells;
	/** Cellules de sorties **/
	lbm_mesh_cell_t* right_out_cells;
	/** Balles rebondissantes **/
	lbm_mesh_cell_t* bounce_cells;
	/** Nombre de cellules d'entrées */
	int left_in_size;
	/** Nombre de cellules de sorties */
	int right_out_size;
	/** Nombre de balles reboncdissantes */
	int bounce_size;
	/** Compteur de cellules d'entrées */
	int left_in_cpt;
	/** Compteur de cellules de sorties */
	int right_out_cpt;
	/** Compteur de balles reboncdissantes */
	int bounce_cpt;
	/** tableau des valeurs de poiseuille pre calculees **/
	double *poiseuille;
} Mesh;

/********************  STRUCT  **********************/
/** Structure des en-têtes utilisée dans le fichier de sortie. **/
typedef struct lbm_file_header_s
{
	/** Taille totale du maillage simulé (hors mailles fantômes). **/
	uint32_t mesh_width;
	/** Taille totale du maillage simulé (hors mailles fantômes). **/
	uint32_t mesh_height;
	/** Number of vertical lines. **/
	uint32_t lines;
} lbm_file_header_t;

/********************  STRUCT  **********************/
/** Pour la lecture du fichier de sortie. **/
typedef struct lbm_data_file_s
{
	FILE * fp;
	lbm_file_header_t header;
	lbm_file_entry_t * entries;
} lbm_data_file_t;


/*******************  FUNCTION  *********************/
void Mesh_init( Mesh * mesh, int width,  int height );
void Mesh_release( Mesh * mesh );

/*******************  FUNCTION  *********************/
void fatal(const char * message);

/*******************  FUNCTION  *********************/
void add_left_in_cell(Mesh* mesh, lbm_mesh_cell_t node);

/*******************  FUNCTION  *********************/
void add_right_out_cell(Mesh* mesh, lbm_mesh_cell_t node);

/*******************  FUNCTION  *********************/
void add_bounce_cell(Mesh* mesh, lbm_mesh_cell_t node);

/*******************  FUNCTION  *********************/
/**
 * Fonction à utiliser pour récupérer une cellule du maillage en fonction de ses coordonnées.
**/
static inline lbm_mesh_cell_t Mesh_get_cell( const Mesh *mesh, int x, int y)
{
	return &mesh->cells[ (x * mesh->height + y) * DIRECTIONS];
}


/*******************  FUNCTION  *********************/
/**
 * Fonction à utiliser pour récupérer une colonne (suivant y, x fixé) du maillage en fonction de ses coordonnées.
**/
static inline lbm_mesh_cell_t Mesh_get_col( const Mesh * mesh, int x )
{
	return &mesh->cells[ x * mesh->height * DIRECTIONS + DIRECTIONS];
}

#endif //LBM_STRUCT_H
