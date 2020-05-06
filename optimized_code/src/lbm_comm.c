/********************  HEADERS  *********************/
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "./header/lbm_comm.h"

/*******************  FUNCTION  *********************/
int lbm_helper_pgcd(int a, int b)
{
  int c;
  while(b!=0)
    {
      c = a % b;
      a = b;
      b = c;
    }
  return a;
}

/*******************  FUNCTION  *********************/
/**
 * Affiche la configuation du lbm_comm pour un rank donné
 * @param mesh_comm Configuration à afficher
 **/
void  lbm_comm_print( lbm_comm_t *mesh_comm )
{
  int rank ;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  printf( " RANK %d ( LEFT %d RIGHT %d TOP %d BOTTOM %d) ( POSITION %d %d ) (WH %d %d ) \n",
    mesh_comm->id,
    mesh_comm->left_id,
    mesh_comm->right_id,
    mesh_comm->top_id,
    mesh_comm->bottom_id,
    mesh_comm->x,
    mesh_comm->y,
    mesh_comm->width,
    mesh_comm->height );
}

/*******************  FUNCTION  *********************/
int helper_get_rank_id(int nb_x,int nb_y,int rank_x,int rank_y)
{
  if (rank_x < 0 || rank_x >= nb_x)
    return -1;
  else if (rank_y < 0 || rank_y >= nb_y)
    return -1;
  else
    return (rank_x + rank_y * nb_x);
}

/*******************  FUNCTION  *********************/
/**
 * Initialise un lbm_comm :
 * - Voisins
 * - Taille du maillage local
 * - Position relative
 * @param mesh_comm MeshComm à initialiser
 * @param rank Rank demandant l'initalisation
 * @param comm_size Taille totale du communicateur
 * @param width largeur du maillage
 * @param height hauteur du maillage
 **/
void lbm_comm_init( lbm_comm_t * mesh_comm, int rank, int comm_size, int width, int height )
{
  //vars
  int nb_x;
  int nb_y;
  int rank_x;
  int rank_y;
  int step_x, step_y;

  // compute splitting
  nb_x = comm_size;
  nb_y = 1;

  // calc current rank postition (ID)
  rank_x = rank;
  rank_y = 0;

  // set up
  mesh_comm->nb_x = nb_x;
  mesh_comm->nb_y = nb_y;

  step_x = width / nb_x;
  step_y = height;

  if(step_x <= 3){
    fatal("To much processus for not enough work");
  }

  if(rank == comm_size -1){
      //setup size (+2 for ghost cells on border)
      mesh_comm->width = step_x + (width % nb_x) + 2;
      mesh_comm->height = step_y + 2;

      //setup position
      mesh_comm->x = rank_x * step_x;
      mesh_comm->y = rank_y * step_y;


  }else{
      //setup size (+2 for ghost cells on border)
      mesh_comm->width = step_x + 2;
      mesh_comm->height = step_y + 2;

      //setup position
      mesh_comm->x = rank_x * step_x;
      mesh_comm->y = rank_y * step_y;
  }

  // Compute neighbour nodes id
  mesh_comm->id  = helper_get_rank_id(nb_x, nb_y, rank_x, rank_y);
  mesh_comm->left_id  = helper_get_rank_id(nb_x, nb_y,rank_x - 1, rank_y);
  mesh_comm->right_id = helper_get_rank_id(nb_x, nb_y,rank_x + 1, rank_y);
  mesh_comm->top_id  = helper_get_rank_id(nb_x, nb_y,rank_x, rank_y+1);
  mesh_comm->bottom_id = helper_get_rank_id(nb_x, nb_y,rank_x, rank_y-1);

  mesh_comm->request_cpt = 0;


/**************************************************************************************
  //check
  if (width % comm_size != 0)
    fatal("Can't get a 2D cut for current problem size and number of processes.");

  //compute splitting
  nb_x = lbm_helper_pgcd(comm_size,width);
  nb_y = comm_size / nb_x;

  //calc current rank position (ID)
  rank_x = rank % nb_x;
  rank_y = rank / nb_x;

  //setup nb
  mesh_comm->nb_x = nb_x;
  mesh_comm->nb_y = nb_y;

  //setup size (+2 for ghost cells on border)
  mesh_comm->width = width / nb_x + 2;
  mesh_comm->height = height / nb_y + 2;

  //setup position
  mesh_comm->x = rank_x * width / nb_x;
  mesh_comm->y = rank_y * height / nb_y;
  
  // Compute neighbour nodes id
  mesh_comm->id  = helper_get_rank_id(nb_x, nb_y, rank_x, rank_y);
  mesh_comm->left_id  = helper_get_rank_id(nb_x, nb_y,rank_x - 1, rank_y);
  mesh_comm->right_id = helper_get_rank_id(nb_x, nb_y,rank_x + 1, rank_y);
  mesh_comm->top_id  = helper_get_rank_id(nb_x, nb_y,rank_x, rank_y+1);
  mesh_comm->bottom_id = helper_get_rank_id(nb_x, nb_y,rank_x, rank_y-1);

*******************************************************************************************/

  mesh_comm->request_cpt = 0;

  //if debug print comm
#ifndef NDEBUG
  lbm_comm_print( mesh_comm );
#endif
}


/*******************  FUNCTION  *********************/
/**
 * Libere un lbm_comm
 * @param mesh_comm MeshComm à liberer
 **/
void lbm_comm_release( lbm_comm_t * mesh_comm )
{
  mesh_comm->x = 0;
  mesh_comm->y = 0;
  mesh_comm->width = 0;
  mesh_comm->height = 0;
  mesh_comm->right_id = -1;
  mesh_comm->left_id = -1;
  mesh_comm->top_id = -1;
  mesh_comm->bottom_id = -1;
}

/*******************  FUNCTION  *********************/
/**
 * Debut de communications asynchrones
 * @param mesh_comm MeshComm à utiliser
 * @param mesh_to_process Mesh a utiliser lors de l'échange des mailles fantomes
 **/
void lbm_comm_sync_ghosts_horizontal( lbm_comm_t * mesh_comm, Mesh *mesh_to_process, lbm_comm_type_t comm_type, int target_rank, int x )
{
  //if target is -1, no comm
  if (target_rank == -1)
    return;

  switch (comm_type){
    case COMM_SEND:
      MPI_Isend( &Mesh_get_cell(mesh_to_process, x, 0)[0], (mesh_comm->height-2) * DIRECTIONS, MPI_DOUBLE, target_rank, 1, MPI_COMM_WORLD, &mesh_comm->requests[mesh_comm->request_cpt++]);
      break;

    case COMM_RECV:
      MPI_Recv( &Mesh_get_cell(mesh_to_process, x, 0)[0], (mesh_comm->height-2) * DIRECTIONS, MPI_DOUBLE, target_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      break;
    default:
      fatal("Unknown type of communication.");
    }
}

/*******************  FUNCTION  *********************/
void lbm_comm_ghost_exchange(lbm_comm_t * mesh, Mesh *mesh_to_process)
{
  //Left to right phase : on reçoit à droite et on envoie depuis la gauche
  lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_SEND,mesh->right_id,mesh->width - 2);
  
  // Right to left phase : on reçoit à gauche et on envoie depuis la droite
  lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_SEND,mesh->left_id,1);
}

/*******************  FUNCTION  *********************/
void lbm_comm_sync_ghosts_wait( lbm_comm_t * mesh, Mesh *mesh_to_process)
{
  MPI_Waitall(mesh->request_cpt, mesh->requests, MPI_STATUS_IGNORE);

  lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_RECV,mesh->left_id,0);
    
  lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_RECV,mesh->right_id,mesh->width - 1);

  mesh->request_cpt = 0;
}