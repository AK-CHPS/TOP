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
  mesh->cells = malloc( width * height  * DIRECTIONS * sizeof( double ) );

  //tableau de cellules speciales
  mesh->spec_tab = malloc((width*2 + height*2) * 2 * sizeof(int));

  //nombre de cellule speciale
  mesh->spec_cpt = 0;

  //taille du tableau de cellule speciale
  mesh->spec_size = (width*2 + height*2);

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

  //free memory
  free( mesh->cells);
  free( mesh->spec_tab);

  mesh->cells = NULL;
}

/*******************  FUNCTION  *********************/
/**
 * Fonction d'initialisation du maillage local.
 * @param mesh Maillage à initialiser.
 * @param width Taille du maillage, mailles fantomes comprises.
 * @param height Taille du maillage, mailles fantomes comprises.
 **/
void lbm_mesh_type_t_init( lbm_mesh_type_t * meshtype, int width,  int height )
{
  //setup params
  meshtype->width = width;
  meshtype->height = height;

  //alloc cells memory
  meshtype->types = malloc( width * height * sizeof( lbm_cell_type_t ) );

  //errors
  if( meshtype->types == NULL )
    {
      perror( "malloc" );
      abort();
    }
}

/*******************  FUNCTION  *********************/
/** Libère la mémoire d'un maillage. **/
void lbm_mesh_type_t_release( lbm_mesh_type_t * mesh )
{
  //reset values
  mesh->width = 0;
  mesh->height = 0;

  //free memory
  free( mesh->types );
  mesh->types = NULL;
}

/*******************  FUNCTION  *********************/
void fatal(const char * message)
{
  fprintf(stderr,"FATAL ERROR : %s\n",message);
  abort();
}

/*******************  FUNCTION  *********************/
void add_spec_cell(Mesh *mesh, int i, int j)
{
  if(mesh->spec_cpt < mesh->spec_size){
    mesh->spec_tab[mesh->spec_cpt * 2] = i;
    mesh->spec_tab[mesh->spec_cpt * 2 + 1] = j;
    mesh->spec_cpt++;
  }else{
    mesh->spec_tab = realloc(mesh->spec_tab, mesh->spec_size*2*2*sizeof(int));
    mesh->spec_size *= 2;
    add_spec_cell(mesh, i, j);
  }
}