#include "./header/lbm_config.h"
#include "./header/lbm_io.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* write header in the output file */
void write_output_file_header(FILE *output_file) {
  // setup header values
  lbm_file_header_t header;
  header.magick = 0x12345;
  header.mesh_height = MESH_HEIGHT;
  header.mesh_width = MESH_WIDTH;
  header.lines = 1;

  // write file
  fwrite(&header, sizeof(header), 1, output_file);
}

/* open the output file */
FILE *open_output_file() {
  // check if a filename is set
  if (RESULT_FILENAME == NULL)
    return NULL;

  // open result file
  FILE *output_file = fopen(RESULT_FILENAME, "w");

  // errors
  if (output_file == NULL) {
    perror(RESULT_FILENAME);
    abort();
  }

  // write header
  write_output_file_header(output_file);

  return output_file;
}

/* close the output file */
void close_output_file(FILE *output_file) { fclose(output_file); }

/* write macroscopique mesh in the output file */
void save_output_file(FILE *output_file, lbm_file_entry_t *mesh, int width,
                      int height) {
  fwrite(mesh, sizeof(lbm_file_entry_t), width * height, output_file);
}