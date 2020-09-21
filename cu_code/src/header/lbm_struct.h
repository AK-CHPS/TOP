#ifndef LBM_STRUCT_H
#define LBM_STRUCT_H

/********************  HEADERS  *********************/
#include "lbm_config.h"
#include <stdint.h>
#include <stdio.h>

/********************** TYPEDEF *********************/
/** Représentation d'un vecteur pour la manipulation des vitesses
 * macroscopiques. **/
typedef float Vector[DIMENSIONS];

/********************  STRUCT  **********************/
/** Une entrée du fichier, avec les deux grandeurs macroscopiques. **/
typedef struct lbm_file_entry_s {
  float v;
  float density;
} lbm_file_entry_t;

/** Mesh of 9 directions nodes **/
typedef struct Mesh {
  /** Cellules du maillages (MESH_WIDTH * MESH_HEIGHT). **/
  float *cells;
  /** Informations à ecrire dans le fichier de sortie (MESH_WIDTH *
   * MESH_HEIGHT). **/
  lbm_file_entry_t *outs;
  /** Largeur du maillage local (mailles fantome comprises). **/
  size_t width;
  /** Largeur du maillage local (mailles fantome comprises). **/
  size_t height;

} Mesh;

/** Structure des en-têtes utilisée dans le fichier de sortie. **/
typedef struct lbm_file_header_s {
  /** Pour validation du format du fichier. **/
  uint32_t magick;
  /** Taille totale du maillage simulé (hors mailles fantômes). **/
  uint32_t mesh_width;
  /** Taille totale du maillage simulé (hors mailles fantômes). **/
  uint32_t mesh_height;
  /** Number of vertical lines. **/
  uint32_t lines;
} lbm_file_header_t;

/** Pour la lecture du fichier de sortie. **/
typedef struct lbm_data_file_s {
  FILE *fp;
  lbm_file_header_t header;
  lbm_file_entry_t *entries;
} lbm_data_file_t;

/********************  FUNCTIONS  **********************/
/** Get a particle in the mesh **/
static inline float *Mesh_get_cell(const Mesh *mesh, int x, int y) {
  return &mesh->cells[(x * mesh->height + y) * DIRECTIONS];
}

/** Initialize the mesh **/
void Mesh_init(Mesh *mesh, size_t width, size_t height);

/** Release the mesh **/
void Mesh_release(Mesh *mesh);

#endif // LBM_STRUCT_H
