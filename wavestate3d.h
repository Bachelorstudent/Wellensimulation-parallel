
#ifndef WAVESTATE3D_H
#define WAVESTATE3D_H

typedef struct _wavestate3dAllign wavestate3dAllign;
typedef wavestate3dAllign *pwavestate3dAllign;
typedef const wavestate3dAllign *pcwavestate3dAllign;

typedef struct _mpiDataExchangeStruct mpiDataExchangeStruct;
typedef mpiDataExchangeStruct *pmpiDataExchangeStruct;

#include "settings.h"

struct _wavestate3dAllign {
  int nx;			/* Number of point masses in x direction */
  int ny;			/* Number of point masses in y direction */
  int nz;			/* Number of point masses in z direction */
  char edgeArray[6];            /* edge position */

  real h;			/* Step width */
  real crho;			/* Elasticity / density constant */

  real *x;			/* Displacements */
  real *v;			/* Velocities */
};

struct _mpiDataExchangeStruct {
//  int toCuboid;
  
  real *xv;
//  real *v;
};

/* Create a mpi buffer */ //direction: y = 1, z = 2
pmpiDataExchangeStruct
new_mpiDataExchangeStruct(int direction, int nx, int ny, int nz, int steps, int cuboidNumber);

/* Create a wavestate3d object */
pwavestate3dAllign
new_wavestate3d_allign(int nx, int ny, int nz, real h, real crho, int steps);

pwavestate3dAllign***
//new_wavestate3d_cuboid(int nx, int ny, int nz, real h, real crho, int steps, int cuboidNumberX, int cuboidNumberY, int cuboidNumberZ);
new_wavestate3d_cuboid(int nx, int ny, int nz, real h, real crho, int steps, int cuboidNumberX, int cuboidNumberY, int cuboidNumberZ, int first_mpi, int last_mpi);

/* Delete a wavestate3d object */
void
del_wavestate3d_allign(pwavestate3dAllign wv);

void
del_wavestate3d_cuboid(pwavestate3dAllign*** wvCuboid, int cuboidNumberX, int cuboidNumberY, int cuboidNumberZ, int first_mpi, int last_mpi);

/* Set displacements and velocities to zero or one */
void
zero_wavestate3d_allign(pwavestate3dAllign wv, int steps);

void
zero_wavestate3d_cuboid(pwavestate3dAllign*** wvCuboid, int steps, int cuboidNumberX, int cuboidNumberY, int cuboidNumberZ, int first_mpi, int last_mpi);

/* Set interesting boundary values */
void
boundary_wavestate3d_allign(pwavestate3dAllign wv, real t, int steps);

void
boundary_wavestate3d_cuboid(pwavestate3dAllign*** wvCuboid, real t, int steps);

/* Print a cuboid */
void
print_wavestate3d_allign(pwavestate3dAllign wv, int steps);

/* Perform one step of the leapfrog method */
void
leapfrog_wavestate3d_cuboid(pwavestate3dAllign wvVec, real delta, int steps);

/* Data exchange between two cuboids */
void
x_direction_data_exchange_wavestate3d_sse2(pwavestate3dAllign wvVec1, pwavestate3dAllign wvVec2, int steps);

void
y_direction_data_exchange_wavestate3d_sse2(pwavestate3dAllign wvVec1, pwavestate3dAllign wvVec2, int steps);

void
z_direction_data_exchange_wavestate3d_sse2(pwavestate3dAllign wvVec1, pwavestate3dAllign wvVec2, int steps);

#endif
