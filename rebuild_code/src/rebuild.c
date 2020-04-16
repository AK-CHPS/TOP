#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <string.h>


//////////////////////////////////////////////////////////////////////////////////////
//										CONFIG										//
//////////////////////////////////////////////////////////////////////////////////////

// Strucutre de donnée représentant la config utilisée //
typedef struct lbm_config_s
{
	//discretisation
	int iterations;
	int width;
	int height;
	//obstacle
	double obstacle_r;
	double obstacle_x;
	double obstacle_y;
	//flow parameters
	double inflow_max_velocity;
	double reynolds;
	//derived flow parameters
	double kinetic_viscosity;
	double relax_parameter;
	//results
	const char * output_filename;
	int write_interval;
} lbm_config_t;

// Variable globale permettant d'acceder a la config //
extern lbm_config_t lbm_gbl_config;

//number of space dimentions to consider
#define DIMENSIONS 2
#define DIRECTIONS 9
//mesh discretisation
#define MESH_WIDTH (lbm_gbl_config.width)
#define MESH_HEIGHT (lbm_gbl_config.height)
//obstable parameter
#define OBSTACLE_R (lbm_gbl_config.obstacle_r)
#define OBSTACLE_X (lbm_gbl_config.obstacle_x)
#define OBSTACLE_Y (lbm_gbl_config.obstacle_y)
//time discretisation
#define ITERATIONS (lbm_gbl_config.iterations)
//initial conditions
//velocity of fluide on left input interface
#define INFLOW_MAX_VELOCITY (lbm_gbl_config.inflow_max_velocity)
//fluid parameters
#define REYNOLDS (lbm_gbl_config.reynolds)
#define KINETIC_VISCOSITY (lbm_gbl_config.kinetic_viscosity)
#define RELAX_PARAMETER (lbm_gbl_config.relax_parameter)
//result filename
#define RESULT_FILENAME (lbm_gbl_config.output_filename)
//#define RESULT_MAGICK 0x12345
#define WRITE_BUFFER_ENTRIES 4096
#define WRITE_STEP_INTERVAL (lbm_gbl_config.write_interval)

// declaration de la variable globale //
lbm_config_t lbm_gbl_config;

// initialise la config avec des valeurs par default //
void setup_default_values(void)
{
	//directisation.
	lbm_gbl_config.iterations = 10000;
	lbm_gbl_config.width = 800;
	lbm_gbl_config.height = 100;
	//obstacle
	lbm_gbl_config.obstacle_x = 0.0;
	lbm_gbl_config.obstacle_y = 0.0;
	lbm_gbl_config.obstacle_r = 0.0;
	//flow
	lbm_gbl_config.inflow_max_velocity = 0.1;
	lbm_gbl_config.reynolds = 100;
	//result output file
	lbm_gbl_config.output_filename = NULL;
	lbm_gbl_config.write_interval = 50;
}

// Calculs des parametres derivés	//
void update_derived_parameter(void)
{
	lbm_gbl_config.kinetic_viscosity = (lbm_gbl_config.inflow_max_velocity * 2.0 * lbm_gbl_config.obstacle_r / lbm_gbl_config.reynolds);
	lbm_gbl_config.relax_parameter = 1.0 / (3.0 * lbm_gbl_config.kinetic_viscosity + 0.5);
}

// charge le fichier de config et initialise la variable gloable //
void load_config(const char * filename)
{
	FILE * fp;
	char buffer[1024], buffer2[1024];
	int intValue, line = 0;
	double doubleValue;

	// ouverture du fichier de config
	fp = fopen(filename,"r");
	if (fp == NULL)
	{
		perror(filename);
		abort();
	}

	// initilaise la varible de config avec les valuers par defaut
	setup_default_values();

	//loop on lines
	while (fgets(buffer,1024,fp) != NULL)
	{
		line++;
		if (buffer[0] == '#')
		{
			// Commentaires
		} else if (sscanf(buffer,"iterations = %d\n",&intValue) == 1) {
			 lbm_gbl_config.iterations = intValue;
		} else if (sscanf(buffer,"width = %d\n",&intValue) == 1) {
			 lbm_gbl_config.width = intValue;
			 if (lbm_gbl_config.obstacle_x == 0.0)
				lbm_gbl_config.obstacle_x = (lbm_gbl_config.width / 5.0 + 1.0);
		} else if (sscanf(buffer,"height = %d\n",&intValue) == 1) {
			lbm_gbl_config.height = intValue;
			if (lbm_gbl_config.obstacle_r == 0.0)
				lbm_gbl_config.obstacle_r = (lbm_gbl_config.height / 10.0 + 1.0);
			if (lbm_gbl_config.obstacle_y == 0.0)
				lbm_gbl_config.obstacle_y = (lbm_gbl_config.height / 2.0 + 3.0);
		} else if (sscanf(buffer,"obstacle_r = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.obstacle_r = doubleValue;
		} else if (sscanf(buffer,"obstacle_x = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.obstacle_x = doubleValue;
		} else if (sscanf(buffer,"obstacle_y = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.obstacle_y = doubleValue;
		} else if (sscanf(buffer,"inflow_max_velocity = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.inflow_max_velocity = doubleValue;
		} else if (sscanf(buffer,"reynolds = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.reynolds = doubleValue;
		} else if (sscanf(buffer,"kinetic_viscosity = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.kinetic_viscosity = doubleValue;
		} else if (sscanf(buffer,"relax_parameter = %lf\n",&doubleValue) == 1) {
			 lbm_gbl_config.relax_parameter = doubleValue;
		} else if (sscanf(buffer,"write_interval = %d\n",&intValue) == 1) {
			 lbm_gbl_config.write_interval = intValue;
		} else if (sscanf(buffer,"output_filename = %s\n",buffer2) == 1) {
			 lbm_gbl_config.output_filename = strdup(buffer2);
		} else {
			fprintf(stderr,"Invalid config option line %d : %s\n",line,buffer);
			abort();
		}
	}

	// verification //
	if (!feof(fp))
	{
		perror(filename);
		abort();
	}else{
		fclose(fp);
	}

	update_derived_parameter();
}

// affichage de la configuraton actuelle //
void print_config(void)
{
	printf("=================== CONFIG ===================\n");
	//discretisation
	printf("%-20s = %d\n","iterations",lbm_gbl_config.iterations);
	printf("%-20s = %d\n","width",lbm_gbl_config.width);
	printf("%-20s = %d\n","height",lbm_gbl_config.height);
	//obstacle
	printf("%-20s = %lf\n","obstacle_r",lbm_gbl_config.obstacle_r);
	printf("%-20s = %lf\n","obstacle_x",lbm_gbl_config.obstacle_x);
	printf("%-20s = %lf\n","obstacle_y",lbm_gbl_config.obstacle_y);
	//flow parameters
	printf("%-20s = %lf\n","reynolds",lbm_gbl_config.reynolds);
	printf("%-20s = %lf\n","reynolds",lbm_gbl_config.reynolds);
	printf("%-20s = %lf\n","inflow_max_velocity",lbm_gbl_config.inflow_max_velocity);
	//results
	printf("%-20s = %s\n","output_filename",lbm_gbl_config.output_filename);
	printf("%-20s = %d\n","write_interval",lbm_gbl_config.write_interval);
	printf("------------ Derived parameters --------------\n");
	printf("%-20s = %lf\n","kinetic_viscosity",lbm_gbl_config.kinetic_viscosity);
	printf("%-20s = %lf\n","relax_parameter",lbm_gbl_config.relax_parameter);
	printf("==============================================\n");
	
}

//////////////////////////////////////////////////////////////////////////////////////
//									STRUCTURES										//
//////////////////////////////////////////////////////////////////////////////////////

// Tableau de direction representant une cellule
typedef double *lbm_mesh_cell_t;
// Représentation d'un vecteur pour la manipulation des vitesses macroscopiques. //
typedef double Vector[DIMENSIONS];

// Definit un maillage pour le domaine local. Ce maillage contient une bordure d'une cellule contenant les mailles fantômes //
typedef struct Mesh
{
	// Cellules du maillages (MESH_WIDTH * MESH_HEIGHT). //
	lbm_mesh_cell_t cells;
	// Largeur du maillage local (mailles fantome comprises). //
	int width;
	// Largeur du maillage local (mailles fantome comprises). //
	int height;
} Mesh;

// Definition des différents type de cellule pour savoir quel traitement y appliquer lors du calcul //
typedef enum lbm_cell_type_e
{
	// Cellule de fluide standard, uniquement application des collisions. //
	CELL_FUILD,
	// Cellules de l'obstacle ou des bordure supérieures et inférieurs. Application de réflexion. //
	CELL_BOUNCE_BACK,
	// Cellule de la paroie d'entrée. Application de Zou/He avec V fixé. //
	CELL_LEFT_IN,
	// Cellule de la paroie de sortie. Application de Zou/He avec gradiant de densité constant. //
	CELL_RIGHT_OUT
} lbm_cell_type_t;

// Tableau maitnenant les informations de type pour les cellules //
typedef struct lbm_mesh_type_s
{
	// Type des cellules du maillages (MESH_WIDTH * MESH_HEIGHT). //
	lbm_cell_type_t * types;
	// Largeur du maillage local (mailles fantome comprises). //
	int width;
	// Largeur du maillage local (mailles fantome comprises). //
	int height;
} lbm_mesh_type_t;

// Structure des en-têtes utilisée dans le fichier de sortie. //
typedef struct lbm_file_header_s
{
	// Pour validation du format du fichier. //
	//uint32_t magick;
	// Taille totale du maillage simulé (hors mailles fantômes). //
	uint32_t mesh_width;
	// Taille totale du maillage simulé (hors mailles fantômes). //
	uint32_t mesh_height;
	// Number of vertical lines. //
	uint32_t lines;
} lbm_file_header_t;

// Une entrée du fichier, avec les deux grandeurs macroscopiques. //
typedef struct lbm_file_entry_s
{
	float v;
	float density;
} lbm_file_entry_t;

// Pour la lecture du fichier de sortie. //
typedef struct lbm_data_file_s
{
	FILE * fp;
	lbm_file_header_t header;
	lbm_file_entry_t * entries;
} lbm_data_file_t;

// Fonction à utiliser pour récupérer une cellule du maillage en fonction de ses coordonnées //
static inline lbm_mesh_cell_t Mesh_get_cell( const Mesh *mesh, int x, int y)
{
	return &mesh->cells[ (x * mesh->height + y) * DIRECTIONS ];
}

// Fonction à utiliser pour récupérer une colonne (suivant y, x fixé) du maillage en fonction de ses coordonnées //
static inline lbm_mesh_cell_t Mesh_get_col( const Mesh * mesh, int x )
{
	//+DIRECTIONS to skip the first (ghost) line
	return &mesh->cells[ x * mesh->height * DIRECTIONS + DIRECTIONS];
}

// Fonction à utiliser pour récupérer un pointeur sur le type d'une cellule du maillage en fonction de ses coordonnées. //
static inline lbm_cell_type_t * lbm_cell_type_t_get_cell( const lbm_mesh_type_t * meshtype, int x, int y)
{
	return &meshtype->types[ x * meshtype->height + y];
}

// Initialisation et allocation du maillage //
void Mesh_init( Mesh * mesh, int width,  int height )
{
  //setup params
  mesh->width = width;
  mesh->height = height;

  //alloc cells memory
  mesh->cells = malloc( width * height  * DIRECTIONS * sizeof( double ) );
  //mesh->cells = NULL;

  //errors
  if( mesh->cells == NULL )
    {
      perror( "malloc" );
      abort();
    }
}

// liberation d'un maillage //
void Mesh_release( Mesh *mesh )
{
  //reset values
  mesh->width = 0;
  mesh->height = 0;

  //free memory
  free( mesh->cells );
  mesh->cells = NULL;
}

// initialisation du type des cellules
void lbm_mesh_type_t_init( lbm_mesh_type_t * meshtype, int width,  int height )
{
  //setup params
  meshtype->width = width;
  meshtype->height = height;

  //alloc cells memory

  	//////////////////////////////////////////////////////////////	
	//		Attention	ici il y avait un +2 que j'ai viré		//
	//////////////////////////////////////////////////////////////

  meshtype->types = malloc( width * height * sizeof( lbm_cell_type_t ) );

  //errors
  if( meshtype->types == NULL )
    {
      perror( "malloc" );
      abort();
    }
}

// liberation du des infos sur le type des cellules
void lbm_mesh_type_t_release( lbm_mesh_type_t * mesh )
{
  //reset values
  mesh->width = 0;
  mesh->height = 0;

  //free memory
  free( mesh->types );
  mesh->types = NULL;
}

//////////////////////////////////////////////////////////////////////////////////////
//								COMMUNICATIONS										//
//////////////////////////////////////////////////////////////////////////////////////

// Definition de l'ID du processus maître. //
#define RANK_MASTER 0

// Definition des différents type de cellule pour savoir quel traitement y appliquer lors du calcul //
typedef enum lbm_corner_pos_e
{
	CORNER_TOP_LEFT = 0,
	CORNER_TOP_RIGHT = 1,
	CORNER_BOTTOM_LEFT = 2,
	CORNER_BOTTOM_RIGHT = 3,
} lbm_corner_pos_t;

typedef enum lbm_comm_type_e
{
	COMM_SEND,
	COMM_RECV
} lbm_comm_type_t;

// Structure utilisée pour stoquer les informations relatives aux communications //
typedef struct lbm_comm_t_s
{
	// Position de la maille locale dans le maillage global (origine). //
	int x;
	int y;
	// Taille de la maille locale. //
	int width;
	int height;
	int nb_x;
	int nb_y;
	// Id du voisin de droite, -1 si aucun. //
	int right_id;
	// Id du voisin de gauche, -1 si aucun. //
	int left_id;
	int top_id;
	int bottom_id;
	int corner_id[4];

	//////////////////////////////	
	//			A voir			//
	//////////////////////////////

	// Requète asynchrone en cours. //
	MPI_Request requests[32];
	lbm_mesh_cell_t buffer;
} lbm_comm_t;

static inline int lbm_comm_width( lbm_comm_t *mc )
{
	return mc->width;
}

static inline int lbm_comm_height( lbm_comm_t *mc )
{
	return mc->height;
}

// affichage des informations de communications //
void  lbm_comm_print( lbm_comm_t *mesh_comm )
{
  int rank ;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  printf( " RANK %d ( LEFT %d RIGHT %d TOP %d BOTTOM %d CORNER %d, %d, %d, %d ) ( POSITION %d %d ) (WH %d %d ) \n",
	  rank,
	  mesh_comm->left_id,
	  mesh_comm->right_id,
	  mesh_comm->top_id,
	  mesh_comm->bottom_id,
	  mesh_comm->corner_id[0],
	  mesh_comm->corner_id[1],
	  mesh_comm->corner_id[2],
	  mesh_comm->corner_id[3],
	  mesh_comm->x,
	  mesh_comm->y,
	  mesh_comm->width,
	  mesh_comm->height );
}

// permet d'obtenir l'id du voisin si il existe //
int helper_get_rank_id(int nb_x,int nb_y,int rank_x,int rank_y)
{
  if (rank_x < 0 || rank_x >= nb_x)
    return -1;
  else if (rank_y < 0 || rank_y >= nb_y)
    return -1;
  else
    return (rank_x + rank_y * nb_x);
}

// permet de trouver le plus grand diviseur commeun entre deux nombres
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

void lbm_comm_init( lbm_comm_t * mesh_comm, int rank, int comm_size, int width, int height )
{
  	int nb_x, nb_y, rank_x, rank_y;

 	// repartition du maillage entre les processus //
  	// divisions en hauteur //
  	//nb_y = lbm_helper_pgcd(comm_size,width);
  	//nb_x = comm_size / nb_y;
  	// division en largeur //
  	nb_x = lbm_helper_pgcd(comm_size,width);
  	nb_y = comm_size / nb_x;

	//check
	assert(nb_x * nb_y == comm_size);
	if (height % nb_y != 0){
	  	perror("Can't get a 2D cut for current problem size and number of processes.");
		abort();
	}

	// calcul du rang en fonction de la repartition dans le maillage //
	rank_x = rank % nb_x;
	rank_y = rank / nb_x;

	// initialisation du nombre de decoupage en x et en y //
	mesh_comm->nb_x = nb_x;
	mesh_comm->nb_y = nb_y;

	// initialisation de la taille des maillages locaux (mailles fantomes comprises)
	mesh_comm->width = width / nb_x + 2;
	mesh_comm->height = height / nb_y + 2;

	// initialisation de la position de chaque ID dans le maillage global
	mesh_comm->x = rank_x * width / nb_x;
	mesh_comm->y = rank_y * height / nb_y;
		
	// Calcul des identifiants des voisins 
	mesh_comm->left_id  = helper_get_rank_id(nb_x,nb_y,rank_x - 1,rank_y);
	mesh_comm->right_id = helper_get_rank_id(nb_x,nb_y,rank_x + 1,rank_y);
	mesh_comm->top_id = helper_get_rank_id(nb_x,nb_y,rank_x,rank_y - 1);
	mesh_comm->bottom_id = helper_get_rank_id(nb_x,nb_y,rank_x,rank_y + 1);
	mesh_comm->corner_id[CORNER_TOP_LEFT] = helper_get_rank_id(nb_x,nb_y,rank_x - 1,rank_y - 1);
	mesh_comm->corner_id[CORNER_TOP_RIGHT] = helper_get_rank_id(nb_x,nb_y,rank_x + 1,rank_y - 1);
	mesh_comm->corner_id[CORNER_BOTTOM_LEFT] = helper_get_rank_id(nb_x,nb_y,rank_x - 1,rank_y + 1);
	mesh_comm->corner_id[CORNER_BOTTOM_RIGHT] = helper_get_rank_id(nb_x,nb_y,rank_x + 1,rank_y + 1);

	//////////////////////////////	
	//			A voir			//
	//////////////////////////////

	//if more than 1 on y, need transmission buffer
	if (nb_y > 1)
	  {
	    mesh_comm->buffer = malloc(sizeof(double) * DIRECTIONS * width / nb_x);
	  } else {
	  mesh_comm->buffer = NULL;
	}

	// affichage des informations de debugging 
	#ifndef NDEBUG
	  lbm_comm_print( mesh_comm );
	#endif
}

// liberation des informations utiles aux communications //
void lbm_comm_release( lbm_comm_t * mesh_comm )
{
	mesh_comm->x = 0;
	mesh_comm->y = 0;
	mesh_comm->width = 0;
	mesh_comm->height = 0;
	mesh_comm->right_id = -1;
	mesh_comm->left_id = -1;
	if (mesh_comm->buffer != NULL)
		free(mesh_comm->buffer);
	mesh_comm->buffer = NULL;
}

// echange des mailles fantômes horizontales
void lbm_comm_sync_ghosts_horizontal( lbm_comm_t * mesh, Mesh *mesh_to_process, lbm_comm_type_t comm_type, int target_rank, int x )
{
  //vars
  MPI_Status status;

  //if target is -1, no comm
  if (target_rank == -1)
    return;

  int y, k;

  switch (comm_type)
    {
    case COMM_SEND:
      for( y = 1 ; y < mesh_to_process->height - 2; y++ )
		for ( k = 0 ; k < DIRECTIONS ; k++)
	 		 MPI_Send( &Mesh_get_cell(mesh_to_process, x, y)[k], 1, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD);
      break;
    case COMM_RECV:
      for( y = 1 ; y < mesh_to_process->height - 2; y++ )
		for ( k = 0 ; k < DIRECTIONS ; k++)
	  		MPI_Recv( &Mesh_get_cell(mesh_to_process, x, y)[k], 1, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD,&status);
      break;
    default:
      	perror("Unknown type of communication.");
    	abort();
    }
}

// echange des mailles fantômes diagonal
void lbm_comm_sync_ghosts_diagonal( lbm_comm_t * mesh, Mesh *mesh_to_process, lbm_comm_type_t comm_type, int target_rank, int x ,int y)
{
  //vars
  MPI_Status status;

  //if target is -1, no comm
  if (target_rank == -1)
    return;

  switch (comm_type)
    {
    case COMM_SEND:
      MPI_Send( Mesh_get_cell( mesh_to_process, x, y ), 1, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD);
      break;
    case COMM_RECV:
      MPI_Recv( Mesh_get_cell( mesh_to_process, x, y ), 1, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, &status);
      break;
    default:
      	perror("Unknown type of communication.");
    	abort();
    }
}

// echange des mailles fantômes vertical
void lbm_comm_sync_ghosts_vertical( lbm_comm_t * mesh, Mesh *mesh_to_process, lbm_comm_type_t comm_type, int target_rank, int y )
{
  //vars
  MPI_Status status;
  int x, k;

  //if target is -1, no comm
  if (target_rank == -1)
    return;

  switch (comm_type)
    {
    case COMM_SEND:
      for ( x = 1 ; x < mesh_to_process->width - 2 ; x++)
	for ( k = 0 ; k < DIRECTIONS ; k++)
	  MPI_Send( &Mesh_get_cell(mesh_to_process, x, y)[k], 1, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD);
      break;
    case COMM_RECV:
      for ( x = 1 ; x < mesh_to_process->width - 2 ; x++)
	for ( k = 0 ; k < DIRECTIONS ; k++)
	  MPI_Recv( &Mesh_get_cell(mesh_to_process, x, y)[k], 1, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD,&status);
      break;
    default:
      	perror("Unknown type of communication.");
    	abort();
    }
}

// echange de toutes les mailles fantômes
void lbm_comm_ghost_exchange(lbm_comm_t * mesh, Mesh *mesh_to_process )
{
  //vars
  int rank;

  //get rank
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // droite vers gauche
  lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_SEND,mesh->right_id,mesh->width - 2);
  lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_RECV,mesh->left_id, 0);

  //prevend comm mixing to avoid bugs
  MPI_Barrier(MPI_COMM_WORLD);
	
  // gauche vers droite
  lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_SEND,mesh->left_id,1);
  lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_RECV,mesh->right_id,mesh->width - 1);

  //prevend comm mixing to avoid bugs
  MPI_Barrier(MPI_COMM_WORLD);
	
  // bas vers haut
  lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_SEND,mesh->bottom_id,mesh->height - 2);
  lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_RECV,mesh->top_id,0);

  //prevend comm mixing to avoid bugs
  MPI_Barrier(MPI_COMM_WORLD);

  // haut vers bas
  lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_SEND,mesh->top_id,1);
  lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_RECV,mesh->bottom_id,mesh->height - 1);

  //prevend comm mixing to avoid bugs
  MPI_Barrier(MPI_COMM_WORLD);

  // haut gauche vers bas droit 
  lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_SEND,mesh->corner_id[CORNER_TOP_LEFT],1,1);
  lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_RECV,mesh->corner_id[CORNER_BOTTOM_RIGHT],mesh->width - 1,mesh->height - 1);

  //prevend comm mixing to avoid bugs
  MPI_Barrier(MPI_COMM_WORLD);

  // bas gauche vers haut droit
  lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_SEND,mesh->corner_id[CORNER_BOTTOM_LEFT],1,mesh->height - 2);
  lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_RECV,mesh->corner_id[CORNER_TOP_RIGHT],mesh->width - 1,0);

  //prevend comm mixing to avoid bugs
  MPI_Barrier(MPI_COMM_WORLD);

  // haut droit vers bas gauche
  lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_SEND,mesh->corner_id[CORNER_TOP_RIGHT],mesh->width - 2,1);
  lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_RECV,mesh->corner_id[CORNER_BOTTOM_LEFT],0,mesh->height - 1);

  //prevend comm mixing to avoid bugs
  MPI_Barrier(MPI_COMM_WORLD);

  // bas gauche vers haut droit
  lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_SEND,mesh->corner_id[CORNER_BOTTOM_LEFT],1,mesh->height - 2);
  lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_RECV,mesh->corner_id[CORNER_TOP_RIGHT],mesh->width - 1,0);
}

//////////////////////////////////////////////////////////////////////////////////////
//									PHYSIQUE										//
//////////////////////////////////////////////////////////////////////////////////////

#if DIRECTIONS == 9 && DIMENSIONS == 2
const Vector direction_matrix[DIRECTIONS] = {
	{+0.0,+0.0},
	{+1.0,+0.0}, {+0.0,+1.0}, {-1.0,+0.0}, {+0.0,-1.0},
	{+1.0,+1.0}, {-1.0,+1.0}, {-1.0,-1.0}, {+1.0,-1.0}
};
#else
#error Need to defined adapted direction matrix.
#endif

#if DIRECTIONS == 9
const double equil_weight[DIRECTIONS] = {
	4.0/9.0 ,
	1.0/9.0 , 1.0/9.0 , 1.0/9.0 , 1.0/9.0,
	1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
const int opposite_of[DIRECTIONS] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
#else
#error Need to defined adapted equibirium distribution function
#endif

// calcul de la norme 2 du vecteur de dimensions
double get_vect_norme_2(const Vector vect1,const Vector vect2)
{
	int k;
	double res = 0.0;

	for ( k = 0 ; k < DIMENSIONS ; k++)
		res += vect1[k] * vect2[k];

	return res;
}

// calcul la somme des directions
double get_cell_density(const lbm_mesh_cell_t cell)
{
	int k;
	double res = 0.0;

	if(cell == NULL){
		perror("CELL IS NULL");
		abort();
	}

	for( k = 0 ; k < DIRECTIONS ; k++)
		res += cell[k];

	return res;
}

// calcul la velocité du cellule
void get_cell_velocity(Vector v,const lbm_mesh_cell_t cell,double cell_density)
{
	int k,d;

	if(v == NULL){
		perror("VECTOR IS NULL");
		abort();
	}else if(cell == NULL){
		perror("CELL IS NULL");
		abort();
	}

	for ( d = 0 ; d < DIMENSIONS ; d++)
	{
		v[d] = 0.0;

		for ( k = 0 ; k < DIRECTIONS ; k++)
			v[d] += cell[k] * direction_matrix[k][d];

		v[d] = v[d] / cell_density;
	}
}

// calcul du profil d'equilibre //
double compute_equilibrium_profile(Vector velocity,double density,int direction)
{
	double v2, p, p2, feq;

	//velocity norme 2 (v * v)
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

// calcul des colisions entre les cellules //
void compute_cell_collision(lbm_mesh_cell_t cell_out,const lbm_mesh_cell_t cell_in)
{
	int k;
	double density, feq;
	Vector v;

	density = get_cell_density(cell_in);
	get_cell_velocity(v,cell_in,density);

	for( k = 0 ; k < DIRECTIONS ; k++)
	{
		feq = compute_equilibrium_profile(v,density,k);
		cell_out[k] = cell_in[k] - RELAX_PARAMETER * (cell_in[k] - feq);
	}
}

// caclul le rebond en arriere //
void compute_bounce_back(lbm_mesh_cell_t cell)
{
	int k;

	//compute bounce back
	for ( k = 0 ; k < DIRECTIONS ; k++)
		cell[k] = cell[opposite_of[k]];
}

// calcul poisefeuille de bourg palette 
double helper_compute_poiseuille(int i,int size)
{
	double y = (double)(i - 1);
	double L = (double)(size - 1);
	return 4.0 * INFLOW_MAX_VELOCITY / ( L * L ) * ( L * y - y * y );
}

// calcul des flux d'entrées
void compute_inflow_zou_he_poiseuille_distr( const Mesh *mesh, lbm_mesh_cell_t cell,int id_y)
{
	//vars
	double v;
	double density;

	//errors
	#if DIRECTIONS != 9
	#error Implemented only for 9 directions
	#endif

	v = helper_compute_poiseuille(id_y,mesh->height);

	//compute rho from u and inner flow on surface
	density = (cell[0] + cell[2] + cell[4] + 2 * ( cell[3] + cell[6] + cell[7] )) / (1.0 - v) ;

	//now compute unknown microscopic values
	cell[1] = cell[3];// + (2.0/3.0) * density * v_y <--- no velocity on Y so v_y = 0
	cell[5] = cell[7] - 0.5 * (cell[2] - cell[4]) + 0.166667 * (density * v);
	                       //+ (1.0/2.0) * density * v_y    <--- no velocity on Y so v_y = 0
	cell[8] = cell[6] + 0.5 * (cell[2] - cell[4]) + 0.166667 * (density * v);
	                       //- (1.0/2.0) * density * v_y    <--- no velocity on Y so v_y = 0
	//no need to copy already known one as the value will be "loss" in the wall at propagatation time
}

// calcul du flux de sortie
void compute_outflow_zou_he_const_density(lbm_mesh_cell_t cell)
{
	const double density = 1.0;
	double v;

	//errors
	#if DIRECTIONS != 9
	#error Implemented only for 9 directions
	#endif

	//compute macroscopic v depeding on inner flow going onto the wall
	v = -1.0 + (1.0 / density) * (cell[0] + cell[2] + cell[4] + 2 * (cell[1] + cell[5] + cell[8]));

	//now can compute unknown microscopic values
	cell[3] = cell[1] - 0.66667 * density * v;
	cell[7] = cell[5] + 0.5 * (cell[2] - cell[4])
	                       //- (1.0/2.0) * (density * v_y)    <--- no velocity on Y so v_y = 0
	                         - 0.166667 * (density * v);
	cell[6] = cell[8] + 0.5 * (cell[4] - cell[2])
	                       //+ (1.0/2.0) * (density * v_y)    <--- no velocity on Y so v_y = 0
	                         - 0.166667 * (density * v);
}

// calcul le comportement des cellules special
void special_cells(Mesh * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm)
{
	//vars
	int i,j;

	//loop on all inner cells
	for( i = 1 ; i < mesh->width - 1; i++ )
	{
		for( j = 1 ; j < mesh->height - 1; j++)
		{
			switch (*( lbm_cell_type_t_get_cell( mesh_type , i, j) ))
			{
				case CELL_FUILD:
					break;
				case CELL_BOUNCE_BACK:
					compute_bounce_back(Mesh_get_cell(mesh, i, j));
					break;
				case CELL_LEFT_IN:
					compute_inflow_zou_he_poiseuille_distr(mesh, Mesh_get_cell(mesh, i, j) ,j + mesh_comm->y);
					break;
				case CELL_RIGHT_OUT:
					compute_outflow_zou_he_const_density(Mesh_get_cell(mesh, i, j));
					break;
			}
		}
	}
}

// calcul le comportement des cellules lors d'un collision
void collision(Mesh * mesh_out,const Mesh * mesh_in)
{
	//vars
	int i,j;

	//errors
	if(mesh_in->width != mesh_out->width || mesh_in->height != mesh_out->height){
		perror("ERREUR DIMENSION DES MAILLAGES");
		abort();
	}

	// inversion du sens de parcours des boucles 
	for( i = 1 ; i < mesh_in->width - 1 ; i++ )
		for( j = 1 ; j < mesh_in->height - 1 ; j++)
			compute_cell_collision(Mesh_get_cell(mesh_out, i, j),Mesh_get_cell(mesh_in, i, j));
}

// propagation
void propagation(Mesh * mesh_out,const Mesh * mesh_in)
{
	int i, j, k, ii, jj;

	//loop on all cells
	for ( i = 0 ; i < mesh_out->width; i++)
	{
		for ( j = 0 ; j < mesh_out->height ; j++)
		{
			//for all direction
			for ( k  = 0 ; k < DIRECTIONS ; k++)
			{
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
//////////////////////////////////////////////////////////////////////////////////////
//								INITIALISATIONS										//
//////////////////////////////////////////////////////////////////////////////////////

// initialisation de la velocité
void init_cond_velocity_0_density_1(Mesh * mesh)
{
  	int i,j,k;

  	if(mesh == NULL){
  		perror("MESH IS NULL");
  		abort();
  	}

  	//loop on all cells
  	for ( i = 0 ; i <  mesh->width ; i++)
    	for ( j = 0 ; j <  mesh->height ; j++)
      		for ( k = 0 ; k < DIRECTIONS ; k++){
				Mesh_get_cell(mesh, i, j)[k] = equil_weight[k];
      		}
}

// initialisation de l'obstacle circulaire
void setup_init_state_circle_obstacle(Mesh * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm)
{
  	int i,j;

  	//loop on nodes
  	for ( i =  mesh_comm->x; i < mesh->width + mesh_comm->x ; i++)
      	for ( j =  mesh_comm->y ; j <  mesh->height + mesh_comm->y ; j++)
	  		if ( ( (i-OBSTACLE_X) * (i-OBSTACLE_X) ) + ( (j-OBSTACLE_Y) * (j-OBSTACLE_Y) ) <= OBSTACLE_R * OBSTACLE_R )
	      		*( lbm_cell_type_t_get_cell( mesh_type , i - mesh_comm->x, j - mesh_comm->y) ) = CELL_BOUNCE_BACK;
}

// initialisation de cellules
void setup_init_state_global_poiseuille_profile(Mesh * mesh, lbm_mesh_type_t * mesh_type,const lbm_comm_t * mesh_comm)
{
	//vars
	int i,j,k;
	Vector v = {0.0,0.0};
	const double density = 1.0;

	for ( i = 0 ; i < mesh->width ; i++){
      	for ( j = 0 ; j < mesh->height ; j++){
	  		*( lbm_cell_type_t_get_cell( mesh_type , i, j) ) = CELL_FUILD;
	  		for ( k = 0 ; k < DIRECTIONS ; k++){

		      	if(1/*mesh_comm->x <= 1*/){
					v[0] = helper_compute_poiseuille(j + mesh_comm->y,MESH_HEIGHT);
					v[1] = helper_compute_poiseuille(i + mesh_comm->x,MESH_WIDTH);
		      		Mesh_get_cell(mesh, i, j)[k] = compute_equilibrium_profile(v,density,k);
		      	}else{
		      		Mesh_get_cell(mesh, i, j)[k] = equil_weight[k];
		      	}
	    	}
		}
    }
}

// initialisation du bord du domaine
void setup_init_state_border(Mesh * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm)
{
  	int i,j,k;
  	Vector v = {0.0,0.0};
  	const double density = 1.0;

  	// initialisation des cellules de gauche comme entree
  	if( mesh_comm->left_id == -1 )
      	for ( j = 1 ; j < mesh->height - 1 ; j++)
			*( lbm_cell_type_t_get_cell( mesh_type , 0, j) ) = CELL_LEFT_IN;

	// initialisation des cellules de droites comme sortie
  	if( mesh_comm->right_id == -1 )  
      	for ( j = 1 ; j < mesh->height - 1 ; j++)
			*( lbm_cell_type_t_get_cell( mesh_type , mesh->width - 1, j) ) = CELL_RIGHT_OUT;

  	// initialisation des cellules du haut comme solide
  	if (mesh_comm->top_id == -1)
    	for ( i = 0 ; i < mesh->width ; i++)
      		for ( k = 0 ; k < DIRECTIONS ; k++){
				//compute equilibr.
				Mesh_get_cell(mesh, i, 0)[k] = compute_equilibrium_profile(v,density,k);
				//mark as bounce back
				*( lbm_cell_type_t_get_cell( mesh_type , i, 0) ) = CELL_BOUNCE_BACK;
			}

  	// initialisation des cellules du bas comme solide 
  	if (mesh_comm->bottom_id == -1)
    	for ( i = 0 ; i < mesh->width ; i++)
      		for ( k = 0 ; k < DIRECTIONS ; k++){
	  			//compute equilibr.
	  			Mesh_get_cell(mesh, i, mesh->height - 1)[k] = compute_equilibrium_profile(v,density,k);
	  			//mark as bounce back
	  			*( lbm_cell_type_t_get_cell( mesh_type , i, mesh->height - 1) ) = CELL_BOUNCE_BACK;
	  		}
}


// initialisation du maillage
void setup_init_state(Mesh * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm)
{
	//init_cond_velocity_0_density_1(mesh);
  	setup_init_state_global_poiseuille_profile(mesh,mesh_type,mesh_comm);
  	setup_init_state_border(mesh,mesh_type,mesh_comm);
  	setup_init_state_circle_obstacle(mesh,mesh_type, mesh_comm);
}


//////////////////////////////////////////////////////////////////////////////////////
//										MAIN										//
//////////////////////////////////////////////////////////////////////////////////////

void save_frame(FILE * fp,const Mesh * mesh)
{
  	//write buffer to write float instead of double
  	lbm_file_entry_t buffer[WRITE_BUFFER_ENTRIES];
  	int i,j,cnt;
  	double density, norm;
  	Vector v;

  	//loop on all values
  	cnt = 0;
  	for (i = 1 ; i < mesh->width - 1 ; i++){
      	for (j = 1 ; j < mesh->height - 1 ; j++){
		  //compute macrospic values
		  density = get_cell_density(Mesh_get_cell(mesh, i, j));
		  get_cell_velocity(v,Mesh_get_cell(mesh, i, j),density);
		  norm = sqrt(get_vect_norme_2(v,v));

		  //fill buffer
		  buffer[cnt].density = density;
		  buffer[cnt].v = norm;
		  cnt++;
				
	  	//flush buffer if full
	  	if (cnt == WRITE_BUFFER_ENTRIES){
	      	fwrite(buffer,sizeof(lbm_file_entry_t),cnt,fp);
	      	cnt = 0;}
		}
    }

  	//final flush
  	if (cnt != 0)
    	fwrite(buffer,sizeof(lbm_file_entry_t),cnt,fp);
}

// ecrit le maillage de tous les processus dans le fichier de sortie
void save_frame_all_domain( FILE * fp, Mesh *source_mesh, Mesh *temp, int rank, int comm_size)
{
  	int i = 0;
  	MPI_Status status;

  	if(rank > 0){
  		MPI_Send( source_mesh->cells, source_mesh->width * source_mesh->height * DIRECTIONS, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
  	}else{
  		save_frame(fp,source_mesh);
  		for( i = 1 ; i < comm_size ; i++ ){
      		MPI_Recv( temp->cells, source_mesh->width  * source_mesh->height * DIRECTIONS, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status );
      		save_frame(fp,temp);
    	}
  	}  
}

// ecrit le header dans le fichier de sortie //
void write_file_header(FILE * fp,lbm_comm_t * mesh_comm)
{
  //setup header values
  lbm_file_header_t header;
  //header.magick    = RESULT_MAGICK;
  header.mesh_height = MESH_HEIGHT;
  header.mesh_width  = MESH_WIDTH;
  header.lines       = mesh_comm->nb_y;

  //write file
  fwrite(&header,sizeof(header),1,fp);
}

// ouver le fichier de sortie 
FILE * open_output_file(lbm_comm_t * mesh_comm)
{
	FILE * fp;

  
	if (RESULT_FILENAME == NULL)
    	return NULL;

  	//open result file
  	fp = fopen(RESULT_FILENAME,"w");

  	//errors
  	if (fp == NULL)
    {
      	perror(RESULT_FILENAME);
      	abort();
    }

  	//write header
  	write_file_header(fp,mesh_comm);

  	return fp;
}

// ferme le fichier de sortie 
void close_file(FILE* fp)
{
  	fclose(fp);
}

int main(int argc, char *argv[])
{
	// declatartion des varaibles
  	Mesh mesh, temp, temp_render;
  	lbm_mesh_type_t mesh_type;
  	lbm_comm_t mesh_comm;
  	int i, rank, comm_size;
  	FILE * fp = NULL;
  	const char * config_filename = NULL;

  	// Initialisation de MPI
  	MPI_Init( &argc, &argv );
  	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  	MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

  	// Recuperation du nom du fichier de config
  	config_filename = (argc >= 2) ? argv[1] : "config.txt";

  	// chargement de la config et affichage de la config par le processus maitre
  	load_config(config_filename);
  	if (rank == RANK_MASTER)
    	print_config();

    // attente de 0
    MPI_Barrier(MPI_COMM_WORLD);

    // initialisation des informations de communications
    lbm_comm_init( &mesh_comm, rank, comm_size, MESH_WIDTH, MESH_HEIGHT);
    // initialisation et allocation des maillages
    Mesh_init( &mesh, lbm_comm_width( &mesh_comm ), lbm_comm_height( &mesh_comm ) );
  	Mesh_init( &temp, lbm_comm_width( &mesh_comm ), lbm_comm_height( &mesh_comm ) );
  	Mesh_init( &temp_render, lbm_comm_width( &mesh_comm ), lbm_comm_height( &mesh_comm ) );
  	// initialisation du type des cellules (tres important)
  	lbm_mesh_type_t_init( &mesh_type, lbm_comm_width( &mesh_comm ), lbm_comm_height( &mesh_comm ));

  	// le maitre ouvre le fichier
  	if(rank == RANK_MASTER)
    	fp = open_output_file(&mesh_comm);

  	// initialise les maillages
  	setup_init_state( &mesh, &mesh_type, &mesh_comm);
  	setup_init_state( &temp, &mesh_type, &mesh_comm);

  	// ecrit les conditions initiales dans le fichier
  	if (lbm_gbl_config.output_filename != NULL)
    	save_frame_all_domain(fp, &mesh, &temp_render, rank, comm_size);

    MPI_Barrier(MPI_COMM_WORLD);

  	for ( i = 1 ; i < ITERATIONS ; i++ ){
  		// affiche la progression
  		if( (rank == RANK_MASTER) && (i%500 == 0) )
			printf("Progress [%5d / %5d]\n",i,ITERATIONS-1);
  		
		// calcul le comportement des cellules speciales
      	special_cells( &mesh, &mesh_type, &mesh_comm);
      
      	// calclul le comportement des cellules en collision
      	collision( &temp, &mesh);

      	// echange des mailles fantômes
      	lbm_comm_ghost_exchange( &mesh_comm, &temp );
      	
      	// propagation 
      	propagation( &mesh, &temp);
      
      	//save step
      	if ( i % WRITE_STEP_INTERVAL == 0 && lbm_gbl_config.output_filename != NULL )
			save_frame_all_domain(fp, &mesh, &temp_render , rank, comm_size);
  	}



    // le maitre ferme le fichier 
  	if( rank == RANK_MASTER && fp != NULL)
      	close_file(fp);

  	// liberation des information utilises aux communications
  	lbm_comm_release( &mesh_comm );
  	// liberation des maillages 
  	Mesh_release( &mesh );
  	Mesh_release( &temp );
  	Mesh_release( &temp_render );
  	// libération des informations sur le type des cellules
  	lbm_mesh_type_t_release( &mesh_type );

  	// Finalisation de MPI
  	MPI_Finalize();
	return 0;
}