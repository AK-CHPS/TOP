/********************  HEADERS  *********************/
#include "./header/lbm_struct.h"
#include <stdlib.h>

/*******************  FUNCTION  *********************/
/**
 * Fonction d'initialisation du maillage local.
 * @param mesh Maillage à initialiser.
 * @param width Taille du maillage, mailles fantomes comprises.
 * @param height Taille du maillage, mailles fantomes comprises.
 **/
void Mesh_init(Mesh *mesh, size_t width, size_t height) {
  // setup params
  mesh->width = width;
  mesh->height = height;

  // alloc cells memory
  cudaMallocManaged(&mesh->cells, width * height * DIRECTIONS * sizeof(double));

  // alloc array to print in file
  cudaMallocManaged(&mesh->outs, width * height * sizeof(lbm_file_entry_t));

  // errors
  if (mesh->cells == NULL) {
    perror("malloc");
    abort();
  }
}

/*******************  FUNCTION  *********************/
/** Libère la mémoire d'un maillage. **/
void Mesh_release(Mesh *mesh) {
  // reset values
  mesh->width = 0;
  mesh->height = 0;

  // free cells memory
  cudaFree(mesh->cells);
  mesh->cells = NULL;

  // free outputs memory
  cudaFree(mesh->outs);
}

/*******************  FUNCTION  *********************/
void fatal(const char *message) {
  fprintf(stderr, "FATAL ERROR : %s\n", message);
  abort();
}