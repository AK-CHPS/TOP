/********************  HEADERS  *********************/
#include "./header/lbm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/****************  GLOBAL VARS ****************/

lbm_config_t lbm_gbl_config;

/*****************  FUNCTION  *******************/
/**
 * Application des valeurs par defaut au cas ou l'utilisateur en définirait pas
 *tout dans le fichier de configuration.
 **/
void setup_default_values(void) {
  // directisation.
  lbm_gbl_config.iterations = 10000;
  lbm_gbl_config.width = 800;
  lbm_gbl_config.height = 100;
  // obstacle
  lbm_gbl_config.obstacle_x = 0.0f;
  lbm_gbl_config.obstacle_y = 0.0f;
  lbm_gbl_config.obstacle_r = 0.0f;
  // flow
  lbm_gbl_config.inflow_max_velocity = 0.1f;
  lbm_gbl_config.reynolds = 100;
  // result output file
  lbm_gbl_config.output_filename = NULL;
  lbm_gbl_config.write_interval = 50;
}

/*****************  FUNCTION  *******************/
/**
 * Calcul des paramètres dérivés.
 **/
void update_derived_parameter(void) {
  // derived parameter
  lbm_gbl_config.kinetic_viscosity =
      (lbm_gbl_config.inflow_max_velocity * 2.0f * lbm_gbl_config.obstacle_r /
       lbm_gbl_config.reynolds);
  lbm_gbl_config.relax_parameter =
      1.0f / (3.0f * lbm_gbl_config.kinetic_viscosity + 1.0f / 2.0f);
}

/*****************  FUNCTION  *******************/
/**
 * Open and initialize the mesh for the obstacle
 * src: https://stackoverflow.com/questions/9296059/read-pixel-value-in-bmp-file
 **/
void ReadBMP(const char *filename, unsigned char **obstacle_mesh, int *width,
             int *height) {
  FILE *f = fopen(filename, "rb");

  if (f == NULL)
    perror("Image file can't be open");

  unsigned char info[54];
  fread(info, sizeof(unsigned char), 54, f);

  (*width) = *(int *)&info[18];
  (*height) = *(int *)&info[22];
  unsigned data_offset = *(unsigned *)&info[10];

  int row_padded = ((*width) * 3 + 3) & (~3);
  unsigned char *data =
      (unsigned char *)malloc(sizeof(unsigned char) * row_padded);
  unsigned char *local_mesh =
      (unsigned char *)malloc((*width) * (*height) * sizeof(unsigned char));
  cudaMalloc(obstacle_mesh, (*width) * (*height) * sizeof(unsigned char));
  unsigned char tmp;

  fseek(f, data_offset, 0);

  for (int i = 0; i < (*height); i++) {
    fread(data, sizeof(unsigned char), row_padded, f);

    for (int j = 0; j < (*width); j++) {
      int y = j * 3;
      tmp = data[y];
      data[y] = data[y + 2];
      data[y + 2] = tmp;

      int R = (int)data[y], G = (int)data[y + 1], B = (int)data[y + 2];

      float luminance = (0.2126f * R + 0.7152f * G + 0.0722f * B) / 255.0;

      local_mesh[i * (*width) + j] = luminance < 0.5;
    }
  }

  cudaMemcpy(*obstacle_mesh, local_mesh,
             (*width) * (*height) * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  fclose(f);
  free(data);
  free(local_mesh);
}

/*****************  FUNCTION  *******************/
/**
 * Chargement de la config depuis le fichier.
 **/
void load_config(const char *config_filename) {
  // vars
  FILE *fp;
  char buffer[1024];
  char buffer2[1024];
  char buffer3[1024];
  int intValue;
  float doubleValue;
  int line = 0;

  // open the config file
  fp = fopen(config_filename, "r");
  if (fp == NULL) {
    perror(config_filename);
    abort();
  }

  // load default values
  setup_default_values();

  // loop on lines
  while (fgets(buffer, 1024, fp) != NULL) {
    line++;
    if (buffer[0] == '#') {
      // comment, nothing to do
    } else if (sscanf(buffer, "iterations = %d\n", &intValue) == 1) {
      lbm_gbl_config.iterations = intValue;
    } else if (sscanf(buffer, "width = %d\n", &intValue) == 1) {
      lbm_gbl_config.width = intValue;
      if (lbm_gbl_config.obstacle_x == 0.0)
        lbm_gbl_config.obstacle_x = (lbm_gbl_config.width / 5.0 + 1.0);
    } else if (sscanf(buffer, "height = %d\n", &intValue) == 1) {
      lbm_gbl_config.height = intValue;
      if (lbm_gbl_config.obstacle_r == 0.0)
        lbm_gbl_config.obstacle_r = (lbm_gbl_config.height / 10.0 + 1.0);
      if (lbm_gbl_config.obstacle_y == 0.0)
        lbm_gbl_config.obstacle_y = (lbm_gbl_config.height / 2.0 + 3.0);
    } else if (sscanf(buffer, "obstacle_filename = %s\n", buffer3) == 1) {
      ReadBMP(buffer3, &lbm_gbl_config.obstacle_mesh,
              &lbm_gbl_config.obstacle_width, &lbm_gbl_config.obstacle_height);
    } else if (sscanf(buffer, "obstacle_x = %d\n", &intValue) == 1) {
      lbm_gbl_config.obstacle_x = intValue;
    } else if (sscanf(buffer, "obstacle_y = %d\n", &intValue) == 1) {
      lbm_gbl_config.obstacle_y = intValue;
    } else if (sscanf(buffer, "inflow_max_velocity = %f\n", &doubleValue) ==
               1) {
      lbm_gbl_config.inflow_max_velocity = doubleValue;
    } else if (sscanf(buffer, "reynolds = %f\n", &doubleValue) == 1) {
      lbm_gbl_config.reynolds = doubleValue;
    } else if (sscanf(buffer, "kinetic_viscosity = %f\n", &doubleValue) == 1) {
      lbm_gbl_config.kinetic_viscosity = doubleValue;
    } else if (sscanf(buffer, "relax_parameter = %f\n", &doubleValue) == 1) {
      lbm_gbl_config.relax_parameter = doubleValue;
    } else if (sscanf(buffer, "write_interval = %d\n", &intValue) == 1) {
      lbm_gbl_config.write_interval = intValue;
    } else if (sscanf(buffer, "output_filename = %s\n", buffer2) == 1) {
      lbm_gbl_config.output_filename = strdup(buffer2);
    } else {
      fprintf(stderr, "Invalid config option line %d : %s\n", line, buffer);
      abort();
    }
  }

  // check error
  if (!feof(fp)) {
    perror(config_filename);
    abort();
  }

  if ((lbm_gbl_config.obstacle_x + lbm_gbl_config.obstacle_width >=
       lbm_gbl_config.width) ||
      (lbm_gbl_config.obstacle_y + lbm_gbl_config.obstacle_height >=
       lbm_gbl_config.height)) {
    fprintf(stderr, "Obstacle is out of the mesh (%d >= %d) and (%d >= %d)\n",
            lbm_gbl_config.obstacle_x + lbm_gbl_config.obstacle_width,
            lbm_gbl_config.width,
            lbm_gbl_config.obstacle_y + lbm_gbl_config.obstacle_height,
            lbm_gbl_config.height);
    abort();
  }

  update_derived_parameter();
}

/*****************  FUNCTION  *******************/
/**
 * Nettoyage de la mémoire dynamique de la config.
 **/
void config_cleanup(void) {
  free((void *)lbm_gbl_config.output_filename);
  cudaFree(lbm_gbl_config.obstacle_mesh);
}

/*****************  FUNCTION  *******************/
/**
 * Affichage de la config.
 **/
void print_config(void) {
  printf("=================== CONFIG ===================\n");
  // discretisation
  printf("%-20s = %d\n", "iterations", lbm_gbl_config.iterations);
  printf("%-20s = %d\n", "width", lbm_gbl_config.width);
  printf("%-20s = %d\n", "height", lbm_gbl_config.height);
  // obstacle
  // printf("%-20s = %f\n", "obstacle_r", lbm_gbl_config.obstacle_r);
  printf("%-20s = %d\n", "obstacle_width", lbm_gbl_config.obstacle_width);
  printf("%-20s = %d\n", "obstacle_height", lbm_gbl_config.obstacle_height);
  printf("%-20s = %d\n", "obstacle_x", lbm_gbl_config.obstacle_x);
  printf("%-20s = %d\n", "obstacle_y", lbm_gbl_config.obstacle_y);
  // flow parameters
  printf("%-20s = %f\n", "reynolds", lbm_gbl_config.reynolds);
  printf("%-20s = %f\n", "reynolds", lbm_gbl_config.reynolds);
  printf("%-20s = %f\n", "inflow_max_velocity",
         lbm_gbl_config.inflow_max_velocity);
  // results
  printf("%-20s = %s\n", "output_filename", lbm_gbl_config.output_filename);
  printf("%-20s = %d\n", "write_interval", lbm_gbl_config.write_interval);
  printf("------------ Derived parameters --------------\n");
  printf("%-20s = %f\n", "kinetic_viscosity", lbm_gbl_config.kinetic_viscosity);
  printf("%-20s = %f\n", "relax_parameter", lbm_gbl_config.relax_parameter);
  printf("==============================================\n");
}
