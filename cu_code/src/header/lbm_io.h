#ifndef LBM_IO_H
#define LBM_IO_H
#include <stdint.h>
#include <stdio.h>

/** Une entrée du fichier, avec les deux grandeurs macroscopiques. **/
typedef struct lbm_file_entry_s {
  float v;
  float density;
} lbm_file_entry_t;

/** Structure des en-têtes utilisée dans le fichier de sortie. **/
typedef struct lbm_file_header_s {
  /** Pour validation du format du fichier. **/
  uint32_t magick;
  /** Taille totale du maillage simulé (hors mailles fantômes). **/
  uint32_t mesh_width;
  /** Taille totale du maillage simulé (hors mailles fantômes). **/
  uint32_t mesh_height;
  /** Number of vertical lines. **/
  uint32_t lines;
} lbm_file_header_t;

/* write header in the output file */
void write_output_file_header(FILE *output_file);

/* open the output file */
FILE *open_output_file();

/* close the output file */
void close_output_file(FILE *output_file);

/* write macroscopique mesh in the output file */
void save_output_file(FILE *output_file, lbm_file_entry_t *mesh, int width,
                      int height);

#endif // LBM_IO_H