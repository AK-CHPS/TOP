#ifndef LBM_INIT_H
#define LBM_INIT_H

/********************  HEADERS  *********************/
#include "lbm_struct.h"
#include "lbm_comm.h"

void setup_init_state(Mesh *mesh,
		      const lbm_comm_t * mesh_comm);

#endif //LBM_INIT_H
