/********************  HEADERS  *********************/
#include <stdlib.h>
#include "./header/lbm_struct.h"

/*******************  FUNCTION  *********************/
/**
 * Fonction d'initialisation du maillage local.
 * @param mesh Maillage à initialiser.
 * @param width Taille du maillage, mailles fantomes comprises.
 * @param height Taille du maillage, mailles fantomes comprises.
 **/
void Mesh_init( Mesh * mesh, int width,  int height )
{
  //setup params
  mesh->width = width;
  mesh->height = height;

  //alloc cells memory
  mesh->cells = malloc( width * height  * DIRECTIONS * sizeof(double) );

  mesh->left_in_cells = malloc(sizeof(lbm_mesh_cell_t) * height);
  mesh->right_out_cells = malloc(sizeof(lbm_mesh_cell_t) * height);
  mesh->bounce_cells = malloc(sizeof(lbm_mesh_cell_t) * width * 2);

  mesh->values = malloc(sizeof(lbm_file_entry_t) * (width-2) * (height-2));

  mesh->left_in_cpt = 0;
  mesh->right_out_cpt = 0;
  mesh->bounce_cpt = 0;

  mesh->left_in_size = height;
  mesh->right_out_size = height;
  mesh->bounce_size = width * 2;

  mesh->poiseuille = malloc(sizeof(double) * height);

  //errors
  if( mesh->cells == NULL )
    {
      perror( "malloc" );
      abort();
    }
}


/*******************  FUNCTION  *********************/
/** Libère la mémoire d'un maillage. **/
void Mesh_release( Mesh *mesh )
{
  //reset values
  mesh->width = 0;
  mesh->height = 0;

  free(mesh->left_in_cells);
  free(mesh->right_out_cells);
  free(mesh->bounce_cells);
  free(mesh->poiseuille);
  free(mesh->values);

  mesh->left_in_cpt = 0;
  mesh->right_out_cpt = 0;
  mesh->bounce_cpt = 0;

  mesh->left_in_size = 0;
  mesh->right_out_size = 0;
  mesh->bounce_size = 0;

  //free memory
  free( mesh->cells );
  mesh->cells = NULL;
}

/*******************  FUNCTION  *********************/
void fatal(const char * message)
{
  fprintf(stderr,"FATAL ERROR : %s\n",message);
  abort();
}

/*******************  FUNCTION  *********************/
void add_left_in_cell(Mesh* mesh, lbm_mesh_cell_t node)
{
  if(mesh->left_in_cpt < mesh->left_in_size){
    mesh->left_in_cells[mesh->left_in_cpt++] = node;
  }else{
    mesh->left_in_size *= 2;
    mesh->left_in_cells = realloc(mesh->left_in_cells, mesh->left_in_size * sizeof(lbm_mesh_cell_t));
    mesh->left_in_cells[mesh->left_in_cpt++] = node;
  }
}

void add_right_out_cell(Mesh* mesh, lbm_mesh_cell_t node)
{
  if(mesh->right_out_cpt < mesh->right_out_size){
    mesh->right_out_cells[mesh->right_out_cpt++] = node;
  }else{
    mesh->right_out_size *= 2;
    mesh->right_out_cells = realloc(mesh->right_out_cells, mesh->right_out_size * sizeof(lbm_mesh_cell_t));
    mesh->right_out_cells[mesh->right_out_cpt++] = node;
  }
}

void add_bounce_cell(Mesh* mesh, lbm_mesh_cell_t node)
{
  if(mesh->bounce_cpt < mesh->bounce_size){
    mesh->bounce_cells[mesh->bounce_cpt++] = node;
  }else{
    mesh->bounce_size *= 2;
    mesh->bounce_cells = realloc(mesh->bounce_cells, mesh->bounce_size * sizeof(lbm_mesh_cell_t));
    mesh->bounce_cells[mesh->bounce_cpt++] = node;
  }
}