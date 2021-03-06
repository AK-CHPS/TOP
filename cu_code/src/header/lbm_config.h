#ifndef LBM_CONFIG_H
#define LBM_CONFIG_H

/********************  DEFINITIONS  *********************/

//      #define concat(a,b,c,d,e) a##b##c##d##e
// number of space dimentions to consider
#define DIMENSIONS 2
#define DIRECTIONS 9
// mesh discretisation
#define MESH_WIDTH (lbm_gbl_config.width)
#define MESH_HEIGHT (lbm_gbl_config.height)
// obstable parameter
//#define OBSTACLE_R (lbm_gbl_config.obstacle_r)
#define OBSTACLE_MESH (lbm_gbl_config.obstable_mesh)
#define OBSTACLE_WIDTH (lbm_gbl_config.obstable_width)
#define OBSTACLE_HEIGHT (lbm_gbl_config.obstable_height)
#define OBSTACLE_X (lbm_gbl_config.obstacle_x)
#define OBSTACLE_Y (lbm_gbl_config.obstacle_y)
// time discretisation
#define ITERATIONS (lbm_gbl_config.iterations)
// initial conditions
// velocity of fluide on left input interface
#define INFLOW_MAX_VELOCITY (lbm_gbl_config.inflow_max_velocity)
// fluid parameters
#define REYNOLDS (lbm_gbl_config.reynolds)
#define KINETIC_VISCOSITY (lbm_gbl_config.kinetic_viscosity)
#define RELAX_PARAMETER (lbm_gbl_config.relax_parameter)
//       #define __FLUSH_INOUT__ concat(s,l,e,e,p)(1) //AHAH
// result filename
#define RESULT_FILENAME (lbm_gbl_config.output_filename)
#define RESULT_MAGICK 0x12345
#define WRITE_BUFFER_ENTRIES 4096
#define WRITE_STEP_INTERVAL (lbm_gbl_config.write_interval)
//       #define FLUSH_INOUT() __FLUSH_INOUT__      //nice one

/********************  STRUCT  *********************/
/**
 * Structure de configuration du problème à résoudre.
 **/
typedef struct lbm_config_s {
  // discretisation
  int iterations;
  int width;
  int height;
  // obstacle
  unsigned char *obstacle_mesh;
  int obstacle_width;
  int obstacle_height;
  int obstacle_x;
  int obstacle_y;
  // flow parameters
  float inflow_max_velocity;
  float reynolds;
  // derived flow parameters
  float kinetic_viscosity;
  float relax_parameter;
  // results
  const char *output_filename;
  int write_interval;
} lbm_config_t;

/*****************  Functions  *******************/
void load_config(const char *filename);
void update_derived_parameter(void);
void config_cleanup(void);
void print_config(void);
void setup_default_values(void);

/****************  EXTERNAL VARS ****************/
/**
 * Configuration accessible sous le forme d'une variable globale.
 * A utiliser comme une constante à part pour le chargement initiale.
 **/
extern lbm_config_t lbm_gbl_config;

#endif // LBM_CONFIG_H
