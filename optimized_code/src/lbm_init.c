/********************  HEADERS  *********************/
#include <mpi.h>
#include <assert.h>
#include "./header/lbm_phys.h"
#include "./header/lbm_init.h"

#include "./header/lbm_struct.h"

/*******************  FUNCTION  *********************/
/**
 * Initialisation de l'obstacle, on bascule les types des mailles associés à CELL_BOUNCE_BACK.
 * Ici l'obstacle est un cercle de centre (OBSTACLE_X,OBSTACLE_Y) et de rayon OBSTACLE_R.
 **/
void setup_init_state_circle_obstacle(Mesh * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm)
{
  //vars
  int i,j;

  for(j =  mesh_comm->y ; j <  mesh->height + mesh_comm->y ; j++){
    for(i =  mesh_comm->x; i < mesh->width + mesh_comm->x ; i++){
      if ( ((i-OBSTACLE_X) * (i-OBSTACLE_X)) + ((j-OBSTACLE_Y) * (j-OBSTACLE_Y)) <= (OBSTACLE_R * OBSTACLE_R)){
        *( lbm_cell_type_t_get_cell( mesh_type , i - mesh_comm->x, j - mesh_comm->y) ) = CELL_BOUNCE_BACK;
        add_spec_cell(mesh, i - mesh_comm->x, j - mesh_comm->y);
      }
    }
  }
}

/*******************  FUNCTION  *********************/
/**
 * Initialise le fluide complet avec un distribution de poiseuille correspondant un état d'écoulement
 * linéaire à l'équilibre.
 * @param mesh Le maillage à initialiser.
 * @param mesh_type La grille d'information notifiant le type des mailles.
 **/
void setup_init_state_global_poiseuille_profile(Mesh * mesh, lbm_mesh_type_t * mesh_type,const lbm_comm_t * mesh_comm)
{
  //vars
  int i,j,k;
  Vector v = {0.0,0.0};
  const double density = 1.0;

  for ( j = 0 ; j < mesh->height ; j++){
    for ( i = 0 ; i < mesh->width ; i++){
      *( lbm_cell_type_t_get_cell( mesh_type , i, j) ) = CELL_FUILD;
      for ( k = 0 ; k < DIRECTIONS ; k++){
          if (i != 0){
            Mesh_get_cell(mesh, i, j)[k] = equil_weight[k];
          }else{
            v[0] = helper_compute_poiseuille(j + mesh_comm->y,MESH_HEIGHT);
            Mesh_get_cell(mesh, i, j)[k] = compute_equilibrium_profile(v,density,k);  
          }
      } 
    }
  }
}

/*******************  FUNCTION  *********************/
/**
 * Initialisation des conditions aux bords.
 * @param mesh Le maillage à initialiser.
 * @param mesh_type La grille d'information notifiant le type des mailles.
 **/
void setup_init_state_border(Mesh * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm)
{
  //vars
  int i,j,k;
  Vector v = {0.0,0.0};
  const double density = 1.0;

  for(j = 0; j < mesh->height; j++){
    if(mesh_comm->left_id == -1){
      *( lbm_cell_type_t_get_cell( mesh_type , 0, j) ) = CELL_LEFT_IN;
      add_spec_cell(mesh, 0, j);
    }
    if(mesh_comm->right_id == -1){
      *( lbm_cell_type_t_get_cell( mesh_type , mesh->width - 1, j) ) = CELL_RIGHT_OUT;    
      add_spec_cell(mesh, mesh->width - 1, j);
    }
  }

  for (i = 0 ; i < mesh->width; i++){
    if(mesh_comm->top_id == -1){
      for(k = 0 ; k < DIRECTIONS ; k++)
        Mesh_get_cell(mesh, i, 0)[k] = compute_equilibrium_profile(v,density,k);
      *( lbm_cell_type_t_get_cell( mesh_type , i, 0) ) = CELL_BOUNCE_BACK;
      add_spec_cell(mesh, i, 0);
    }
    if(mesh_comm->bottom_id == -1){
      for(k = 0 ; k < DIRECTIONS ; k++)
        Mesh_get_cell(mesh, i, mesh->height - 1)[k] = compute_equilibrium_profile(v,density,k);
      *( lbm_cell_type_t_get_cell( mesh_type , i, mesh->height - 1) ) = CELL_BOUNCE_BACK;
      add_spec_cell(mesh, i, mesh->height - 1);
    }
  }
}

/*******************  FUNCTION  *********************/
/**
 * Mise en place des conditions initiales.
 * @param mesh Le maillage à initialiser.
 * @param mesh_type La grille d'information notifiant le type des mailles.
 * @param mesh_comm La structure de communication pour connaitre notre position absolue dans le maillage globale.
 **/
void setup_init_state(Mesh * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm)
{
  setup_init_state_global_poiseuille_profile(mesh,mesh_type,mesh_comm);
  setup_init_state_border(mesh,mesh_type,mesh_comm);
  setup_init_state_circle_obstacle(mesh,mesh_type, mesh_comm);
}
