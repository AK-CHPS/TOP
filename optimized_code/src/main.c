/********************  HEADERS  *********************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include "./header/lbm_config.h"
#include "./header/lbm_struct.h"
#include "./header/lbm_phys.h"
#include "./header/lbm_init.h"
#include "./header/lbm_comm.h"

/*******************  FUNCTION  *********************/
/**
 * Ecrit l'en-tête du fichier de sortie. Cet en-tête sert essentiellement à fournir les informations
 * de taille du maillage pour les chargements.
 * @param fp Descripteur de fichier à utiliser pour l'écriture.
 **/
void write_file_header(MPI_File fp,lbm_comm_t * mesh_comm)
{
  //setup header values
  lbm_file_header_t header;
  header.mesh_height = MESH_HEIGHT;
  header.mesh_width  = MESH_WIDTH;
  header.lines       = mesh_comm->nb_y;
  
  MPI_File_write(fp, &header, sizeof(lbm_file_header_t), MPI_BYTE, MPI_STATUS_IGNORE);
}

/*******************  FUNCTION  *********************/
void open_output_file(lbm_comm_t * mesh_comm, int rank, MPI_File *fp)
{
  //check if empty filename => so noout
  if (RESULT_FILENAME == NULL)
    return;

  //open result file
  MPI_File_open(MPI_COMM_WORLD, RESULT_FILENAME, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, fp);

  //errors
  if (fp == NULL)
    {
      perror(RESULT_FILENAME);
      abort();
    }

  //write header
  if(rank == 0){
    write_file_header((*fp),mesh_comm);
  }
}

void close_file(MPI_File* fp)
{
  MPI_File_close(fp);
}

/*******************  FUNCTION  *********************/
/**
 * Sauvegarde le résultat d'une étape de calcul. Cette fonction peu être appelée plusieurs fois
 * lors d'une sauvegarde en MPI sur plusieurs processus pour sauvegarder les un après-les autres
 * chacun des domaines.
 * Ne sont écrit que les vitesses et densités macroscopiques sous forme de flotant simple.
 * @param fp Descripteur de fichier à utiliser pour l'écriture.
 * @param mesh Domaine à sauvegarder.
 **/

void save_frame(MPI_File * fp,const Mesh * mesh, const int rank, const int size)
{
  int i, j, offset; 
  static int it;
  Vector v;

  offset = sizeof(lbm_file_header_t) + (it++ * size + rank) * ((mesh->width-2) * (mesh->height-2) * sizeof(lbm_file_entry_t));

  for(i = 1 ; i < mesh->width - 1 ; i++){
    for(j = 1; j < mesh->height - 1 ; j++){
      mesh->values[(i-1) * (mesh->height-2) + (j-1)].density = get_cell_density(Mesh_get_cell(mesh, i, j));
      get_cell_velocity(v,Mesh_get_cell(mesh, i, j), mesh->values[(i-1) * (mesh->height-2) + (j-1)].density);
      mesh->values[(i-1) * (mesh->height-2) + (j-1)].v = __builtin_sqrt(get_vect_norme_2(v,v));
    }
  }

  MPI_File_write_at((*fp), offset, mesh->values, (mesh->height-2) * (mesh->width-2) * sizeof(lbm_file_entry_t), MPI_BYTE, MPI_STATUS_IGNORE);
}

/*******************  FUNCTION  *********************/
int main(int argc, char * argv[])
{
  //vars
  Mesh mesh;
  Mesh temp;
  lbm_comm_t mesh_comm;
  int i, rank, comm_size;
   MPI_File fp;
  const char * config_filename = NULL;

  //init MPI and get current rank and commuincator size.
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

  //get config filename
  if (argc >= 2)
    config_filename = argv[1];
  else
    config_filename = "config.txt";

  //load config file and display it on master node	
  load_config(config_filename);
  if (rank == RANK_MASTER)
    print_config();
	
  MPI_Barrier(MPI_COMM_WORLD); // k : wait for config 

  //init structures, allocate memory...
  lbm_comm_init( &mesh_comm, rank, comm_size, MESH_WIDTH, MESH_HEIGHT);
  Mesh_init( &mesh, lbm_comm_width( &mesh_comm ), lbm_comm_height( &mesh_comm ) );
  Mesh_init( &temp, lbm_comm_width( &mesh_comm ), lbm_comm_height( &mesh_comm ) );

  //master open the output file
  open_output_file(&mesh_comm, rank, &fp);

  //setup initial conditions on mesh
  setup_init_state( &mesh, &mesh_comm);
  setup_init_state( &temp, &mesh_comm);

  //ils ont fait quoi avec les fichiers?
  
  //write initial condition in output file
  if (lbm_gbl_config.output_filename != NULL)
    save_frame(&fp, &temp, rank, comm_size);

  //time steps
  for ( i = 1 ; i < ITERATIONS ; i++ )
    {
      //print progress
      if( (rank == RANK_MASTER) && (i%500 == 0) )
	      printf("Progress [%5d / %5d]\n",i,ITERATIONS-1);

      lbm_comm_ghost_exchange( &mesh_comm, &temp);

      //compute special actions (border, obstacle...)
      special_cells( &mesh);

      lbm_comm_sync_ghosts_wait(&mesh_comm, &temp);

      // compute collision and propagation
      collision_and_propagation(&mesh, &temp);

      //save step
      if(i % WRITE_STEP_INTERVAL == 0 && lbm_gbl_config.output_filename != NULL )
        save_frame(&fp, &temp, rank, comm_size);
    }


  if(fp != NULL){
    close_file(&fp);
  }

  //Free memory
  lbm_comm_release( &mesh_comm );
  Mesh_release( &mesh );
  Mesh_release( &temp );

  //close MPI
  MPI_Finalize();

  return EXIT_SUCCESS;
}
