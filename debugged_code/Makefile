#Commands
CC=gcc
MPICC=mpicc
RM=rm -f
MAKEDEPEND=makedepend

#flags
CFLAGS=-Wall -g
LDFLAGS=-lm
OFLAGS=-O3

#Files
SRC_DIR=./src
HEADERS_DIR=./src/header
OBJ_DIR=./build
LBM_SOURCES=$(SRC_DIR)/main.c $(SRC_DIR)/lbm_phys.c $(SRC_DIR)/lbm_init.c $(SRC_DIR)/lbm_struct.c $(SRC_DIR)/lbm_comm.c $(SRC_DIR)/lbm_config.c
LBM_HEADERS=$(wildcard $(HEADERS_DIR)/*.h)
LBM_OBJECTS=$(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(LBM_SOURCES))

TARGET=lbm display

all: $(TARGET)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c build
	$(MPICC) $(CFLAGS) $(OFLAGS) -c -o $@ $< 

lbm: $(LBM_OBJECTS)
	$(MPICC) $(CFLAGS) $(OFLAGS) -o $@ $^ $(LDFLAGS)

display: src/display.c
	$(CC) $(CFLAGS) $(OFLAGS) -o $@ src/display.c

build:
	mkdir

clean:
	$(RM) $(LBM_OBJECTS)
	$(RM) $(TARGET)

depend:
	$(MAKEDEPEND) -Y. $(LBM_SOURCES) src/display.c

.PHONY: clean all depend

# DO NOT DELETE

main.o: src/header/lbm_config.h src/header/lbm_struct.h src/header/lbm_phys.h src/header/lbm_comm.h src/header/lbm_init.h
lbm_phys.o: src/header/lbm_config.h src/header/lbm_struct.h src/header/lbm_phys.h src/header/lbm_comm.h
lbm_init.o: src/header/lbm_phys.h src/header/lbm_struct.h src/header/lbm_config.h src/header/lbm_comm.h src/header/lbm_init.h
lbm_struct.o: src/header/lbm_struct.h src/header/lbm_config.h
lbm_comm.o: src/header/lbm_comm.h src/header/lbm_struct.h src/header/lbm_config.h
lbm_config.o: src/header/lbm_config.h
display.o: src/header/lbm_struct.h src/header/lbm_config.h
