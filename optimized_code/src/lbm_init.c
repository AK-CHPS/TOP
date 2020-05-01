/********************  HEADERS  *********************/
#include <mpi.h>
#include <assert.h>
#include "./header/lbm_phys.h"
#include "./header/lbm_init.h"

#include "./header/lbm_struct.h"

static double helper_compute_poiseuille(const int i, const int size)
{
  double y = (double)(i - 1);
  double L = (double)(size - 1);
  return 4.0 * INFLOW_MAX_VELOCITY / ( L * L ) * ( L * y - y * y );
}

static void compute_poiseuille(Mesh *mesh)
{
  for(int j = 0; j < mesh->height; j++){
    mesh->poiseuille[j] = helper_compute_poiseuille(j, MESH_HEIGHT);
  }
}


/*******************  FUNCTION  *********************/
/**
 * Initialisation de l'obstacle, on bascule les types des mailles associés à CELL_BOUNCE_BACK.
 * Ici l'obstacle est un cercle de centre (OBSTACLE_X,OBSTACLE_Y) et de rayon OBSTACLE_R.
 **/
static void setup_init_state_circle_obstacle(Mesh * mesh, const lbm_comm_t * mesh_comm)
{
  //vars
  int i,j;

  for(i =  mesh_comm->x; i < mesh->width + mesh_comm->x ; i++){
    for(j =  mesh_comm->y ; j <  mesh->height + mesh_comm->y ; j++){
      if ( ((i-OBSTACLE_X) * (i-OBSTACLE_X)) + ((j-OBSTACLE_Y) * (j-OBSTACLE_Y)) <= (OBSTACLE_R * OBSTACLE_R)){
        add_bounce_cell(mesh, Mesh_get_cell( mesh, i - mesh_comm->x, j - mesh_comm->y));
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
static void setup_init_state_global_poiseuille_profile(Mesh * mesh, const lbm_comm_t * mesh_comm)
{
  //vars
  int i,j,k;
  Vector v = {0.0,0.0};
  const double density = 1.0;

  for ( i = 0 ; i < mesh->width; i++){
    for ( j = 0 ; j < mesh->height; j++){
      v[0] = mesh->poiseuille[j];
      for ( k = 0 ; k < DIRECTIONS ; k++){
          Mesh_get_cell(mesh, i, j)[k] = compute_equilibrium_profile(v,density,k); 
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
static void setup_init_state_border(Mesh * mesh, const lbm_comm_t * mesh_comm)
{
  //vars
  int i,j,k;

  for(j = 0; j < mesh->height; j++){
    if(mesh_comm->left_id == -1){
      add_left_in_cell(mesh, Mesh_get_cell( mesh, 0, j));
    }
    if(mesh_comm->right_id == -1){
      add_right_out_cell(mesh, Mesh_get_cell( mesh, mesh->width - 1, j));
    }
  }

  for (i = 0 ; i < mesh->width; i++){
    if(mesh_comm->top_id == -1){
      for(k = 0 ; k < DIRECTIONS ; k++)
        Mesh_get_cell(mesh, i, 0)[k] = equil_weight[k];//compute_equilibrium_profile(v,density,k);
      add_bounce_cell(mesh, Mesh_get_cell( mesh, i, 0));
    }
    if(mesh_comm->bottom_id == -1){
      for(k = 0 ; k < DIRECTIONS ; k++)
        Mesh_get_cell(mesh, i, mesh->height - 1)[k] = equil_weight[k];//compute_equilibrium_profile(v,density,k);
      add_bounce_cell(mesh, Mesh_get_cell( mesh, i, mesh->height - 1));
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
void setup_init_state(Mesh * mesh, const lbm_comm_t * mesh_comm)
{
  compute_poiseuille(mesh);
  setup_init_state_global_poiseuille_profile(mesh, mesh_comm);
  setup_init_state_border(mesh, mesh_comm);
  setup_init_state_circle_obstacle(mesh, mesh_comm);
}
