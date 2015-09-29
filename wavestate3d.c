#include <stdio.h>
#include <math.h>
#include <stdlib.h>
//#include <omp.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <mm_malloc.h>
#include "wavestate3d.h"

/* ------------------------------------------------------------
   Create a mpi buffer
   ------------------------------------------------------------ */

pmpiDataExchangeStruct
new_mpiDataExchangeStruct(int direction, int nx, int ny, int nz, int steps, int cuboidNumber)
{
  pmpiDataExchangeStruct mpiBuffer;
  
  mpiBuffer = (pmpiDataExchangeStruct) malloc(sizeof(mpiDataExchangeStruct));
  if(cuboidNumber==0)cuboidNumber++;
  if(direction==1){
    mpiBuffer->xv = (real*) _mm_malloc(cuboidNumber * 2 * (nx+steps*2) * steps * nz * sizeof(real), 64);
  }else if(direction==2){
    mpiBuffer->xv = (real*) _mm_malloc(cuboidNumber * 2 * (nx+steps*2) * (ny+steps*2) * steps * sizeof(real), 64);
  }

  return mpiBuffer;
}

/* ------------------------------------------------------------
   Create a wavestate3d object
   ------------------------------------------------------------ */

pwavestate3dAllign
new_wavestate3d_allign(int nx, int ny, int nz, real h, real crho, int steps)
{
  pwavestate3dAllign wvVec;
  int dimensionLength = (nz + steps * 2) * (ny + steps * 2) * (nx + steps * 2);

  wvVec = (pwavestate3dAllign) malloc(sizeof(wavestate3dAllign));

  wvVec->x = (real*) _mm_malloc(dimensionLength * sizeof(real), 32);
  wvVec->v = (real*) _mm_malloc(dimensionLength * sizeof(real), 32);

  wvVec->nx = nx;
  wvVec->ny = ny;
  wvVec->nz = nz;
  wvVec->h = h;
  wvVec->crho = crho;
  
  return wvVec;
}


pwavestate3dAllign***
new_wavestate3d_cuboid(int nx, int ny, int nz, real h, real crho, int steps, int cuboidNumberX, int cuboidNumberY, int cuboidNumberZ, int first_mpi, int last_mpi)
{

  pwavestate3dAllign*** wvVecCuboid;
  int dimensionLength = (nz + steps * 2) * (ny + steps * 2) * (nx + steps * 2);
  int cnt1, cnt2, cnt3;

  wvVecCuboid = (pwavestate3dAllign ***) malloc((size_t) sizeof(wavestate3dAllign**) * cuboidNumberZ);
  for(cnt1=0; cnt1<cuboidNumberZ; cnt1++){
    wvVecCuboid[cnt1] = (pwavestate3dAllign **) malloc((size_t) sizeof(wavestate3dAllign*) * cuboidNumberY);
    for(cnt2=0; cnt2<cuboidNumberY; cnt2++){
      wvVecCuboid[cnt1][cnt2] = (pwavestate3dAllign *) malloc((size_t) sizeof(wavestate3dAllign) * cuboidNumberX);

      for(cnt3=0; cnt3<cuboidNumberX; cnt3++){
        wvVecCuboid[cnt1][cnt2][cnt3] = (pwavestate3dAllign) malloc(sizeof(wavestate3dAllign));
        if(first_mpi<=cnt1*cuboidNumberX*cuboidNumberY+cnt2*cuboidNumberX+cnt3 && last_mpi>cnt1*cuboidNumberX*cuboidNumberY+cnt2*cuboidNumberX+cnt3){

          wvVecCuboid[cnt1][cnt2][cnt3]->x = (real*) _mm_malloc(dimensionLength * sizeof(real), 32);
          wvVecCuboid[cnt1][cnt2][cnt3]->v = (real*) _mm_malloc(dimensionLength * sizeof(real), 32);

          wvVecCuboid[cnt1][cnt2][cnt3]->nx = nx;
          wvVecCuboid[cnt1][cnt2][cnt3]->ny = ny;
          wvVecCuboid[cnt1][cnt2][cnt3]->nz = nz;
          wvVecCuboid[cnt1][cnt2][cnt3]->h = h;
          wvVecCuboid[cnt1][cnt2][cnt3]->crho = crho;

          wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[0] = 0;
          wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[1] = 0;
          wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[2] = 0;
          wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[3] = 0;
          wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[4] = 0;
          wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[5] = 0;

          if(cnt3 == 0)
            wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[0] = 1;
          if(cnt3 == cuboidNumberX-1)
            wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[1] = 1;
          if(cnt2 == 0)
            wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[2] = 1;
          if(cnt2 == cuboidNumberY-1)
            wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[3] = 1;
          if(cnt1 == 0)
            wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[4] = 1;
          if(cnt1 == cuboidNumberZ-1)
            wvVecCuboid[cnt1][cnt2][cnt3]->edgeArray[5] = 1;
        }


      }
    }
  }

  return wvVecCuboid;
}

/* ------------------------------------------------------------
   Delete a wavestate3d object
   ------------------------------------------------------------ */

void
del_wavestate3d_allign(pwavestate3dAllign wv)
{
  free(wv->v);
  free(wv->x);

  wv->v = 0;			/* Safety measure */
  wv->x = 0;

  free(wv);
}

void
del_wavestate3d_cuboid(pwavestate3dAllign*** wvCuboid, int cuboidNumberX, int cuboidNumberY, int cuboidNumberZ, int first_mpi, int last_mpi)
{
  int cnt1, cnt2, cnt3;

  for(cnt1=0; cnt1<cuboidNumberZ; cnt1++){
    for(cnt2=0; cnt2<cuboidNumberY; cnt2++){
      for(cnt3=0; cnt3<cuboidNumberX; cnt3++){

        if(first_mpi<=cnt1*cuboidNumberX*cuboidNumberY+cnt2*cuboidNumberX+cnt3 && last_mpi>cnt1*cuboidNumberX*cuboidNumberY+cnt2*cuboidNumberX+cnt3){
          free(wvCuboid[cnt1][cnt2][cnt3]->x);
          free(wvCuboid[cnt1][cnt2][cnt3]->v);
        }

	free(wvCuboid[cnt1][cnt2][cnt3]);
      }
      free(wvCuboid[cnt1][cnt2]);
    }
    free(wvCuboid[cnt1]);
  }

  free(wvCuboid);
}

/* ------------------------------------------------------------
   Set displacements and velocities to zero
   ------------------------------------------------------------ */

void
zero_wavestate3d_allign(pwavestate3dAllign wv, int steps)
{
  real *x = wv->x;
  real *v = wv->v;
  int nx = wv->nx + steps*2;
  int ny = wv->ny + steps*2;
  int nz = wv->nz + steps*2;
  int i,j,k;

  for(i=0; i<nz; i++){
    for(j=0; j<ny; j++){
      for(k=0; k<nx; k++){
	x[i*ny*nx + j*nx + k] = 0.0;
      }
    }
  }

  for(i=0; i<nz; i++){
    for(j=0; j<ny; j++){
      for(k=0; k<nx; k++){
	v[i*ny*nx + j*nx + k] = 0.0;
      }
    }
  } 
}


void
zero_wavestate3d_cuboid(pwavestate3dAllign*** wvCuboid, int steps, int cuboidNumberX, int cuboidNumberY, int cuboidNumberZ, int first_mpi, int last_mpi)
{
  real *x;
  real *v;
  int nx = wvCuboid[first_mpi/(cuboidNumberX*cuboidNumberY)][(first_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][first_mpi%cuboidNumberX]->nx + steps*2;
  int ny = wvCuboid[first_mpi/(cuboidNumberX*cuboidNumberY)][(first_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][first_mpi%cuboidNumberX]->ny + steps*2;
  int nz = wvCuboid[first_mpi/(cuboidNumberX*cuboidNumberY)][(first_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][first_mpi%cuboidNumberX]->nz + steps*2;
  int cnt1, cnt2, cnt3, i,j,k;

  for(cnt1=0; cnt1<cuboidNumberZ; cnt1++){
    for(cnt2=0; cnt2<cuboidNumberY; cnt2++){
      for(cnt3=0; cnt3<cuboidNumberX; cnt3++){

        if(cnt1>=first_mpi/(cuboidNumberX*cuboidNumberY) && cnt1<last_mpi/(cuboidNumberX*cuboidNumberY) && cnt2>=(first_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX && cnt2<(last_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX && cnt3>=first_mpi%cuboidNumberX && cnt3<last_mpi%cuboidNumberX){
          x = wvCuboid[cnt1][cnt2][cnt3]->x;
          v = wvCuboid[cnt1][cnt2][cnt3]->v;
          for(i=0; i<nz; i++){
            for(j=0; j<ny; j++){
              for(k=0; k<nx; k++){
	        x[i*ny*nx + j*nx + k] = 0.0;
              }
            }
          }

          for(i=0; i<nz; i++){
            for(j=0; j<ny; j++){
              for(k=0; k<nx; k++){
	        v[i*ny*nx + j*nx + k] = 0.0;
              }
            }
          }
        }

      }
    }
  }

}


/* ------------------------------------------------------------
   Set interesting boundary values
   ------------------------------------------------------------ */

void
boundary_wavestate3d_allign(pwavestate3dAllign wv, real t)
{
  real *x = wv->x;

  if(0.0 < t && t < 0.25)
    x[0] = sin(M_PI * t / 0.125);
}

void
boundary_wavestate3d_cuboid(pwavestate3dAllign*** wvCuboid, real t)
{
  real *x = wvCuboid[0][0][0]->x;

  if(0.0 < t && t < 0.25)
    x[0] = sin(M_PI * t / 0.125);
}


/* ------------------------------------------------------------
   Prints a cuboid
   ------------------------------------------------------------ */

void
print_wavestate3d_allign(pwavestate3dAllign wv, int steps)
{
  real *x = wv->x;
  real *v = wv->v;
  int nx = wv->nx + steps*2;
  int ny = wv->ny + steps*2;
  int nz = wv->nz + steps*2;
  int i,j,k;

  printf("x\n");
  for(i=0; i<nz; i++){
    for(j=0; j<ny; j++){
      for(k=0; k<nx; k++){
        printf("%.2f ",x[i*ny*nx + j*nx + k]);
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("v\n");
  for(i=0; i<nz; i++){
    for(j=0; j<ny; j++){
      for(k=0; k<nx; k++){
        printf("%.2f ",v[i*ny*nx + j*nx + k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

/* ------------------------------------------------------------
   Perform one step of the leapfrog method
   ------------------------------------------------------------ */

void
leapfrog_wavestate3d_cuboid(pwavestate3dAllign wvVec, real delta, int steps)
{
  real *x = wvVec->x;
  real *v = wvVec->v;
  real h = wvVec->h;
  real crho = wvVec->crho;
  int i,j,k;
  const real constVal = (crho*delta)/(h*h);
  const real const6 = 6.0;

  int arrayCntX = wvVec->nx + steps * 2;
  int arrayCntXY = wvVec->ny + steps * 2;
  arrayCntXY *= arrayCntX;

  int lastImportantField[3];
  int stepsLocal[3];

  //x
  if(wvVec->edgeArray[0])
    stepsLocal[2]=steps-1;
  else
    stepsLocal[2]=0;

  if(wvVec->edgeArray[1])
    lastImportantField[2]=wvVec->nx+steps;
  else
    lastImportantField[2]=wvVec->nx+steps*2-1;
  //y
  if(wvVec->edgeArray[2])
    stepsLocal[1]=steps-1;
  else
    stepsLocal[1]=0;

  if(wvVec->edgeArray[3])
    lastImportantField[1]=wvVec->ny+steps;
  else
    lastImportantField[1]=wvVec->ny+steps*2-1;
  //z
  if(wvVec->edgeArray[4])
    stepsLocal[0]=steps-1;
  else
    stepsLocal[0]=0;

  if(wvVec->edgeArray[5])
    lastImportantField[0]=wvVec->nz+steps;
  else
    lastImportantField[0]=wvVec->nz+steps*2-1;




#ifdef __SSE2__
__m128d rCenter, rLeft, rRight, rConstVal1, rConstVal2, rResult, rConst6;
#endif

  rConstVal1 = _mm_load_pd1(&constVal);
  rConstVal2 = _mm_load_pd1(&delta);
  rConst6 = _mm_load_pd1(&const6);

  //Update velocities
  for(;stepsLocal[0]<steps && stepsLocal[1]<steps && stepsLocal[2]<steps && lastImportantField[0]>=wvVec->nz + steps && lastImportantField[1]>=wvVec->ny + steps && lastImportantField[2]>=wvVec->nx + steps;){
    for(i=stepsLocal[0]+1; i<lastImportantField[0]; i++){
      for(j=stepsLocal[1]+1; j<lastImportantField[1]; j++){

        k=stepsLocal[2]+1;

        if((i*arrayCntXY + j*arrayCntX + k) % 2){
          v[i*arrayCntXY + j*arrayCntX + k] += (x[i*arrayCntXY + j*arrayCntX + k + 1] + x[i*arrayCntXY + j*arrayCntX + k - 1] + x[i*arrayCntXY + (j+1)*arrayCntX + k] + x[i*arrayCntXY + (j-1)*arrayCntX + k] + x[(i+1)*arrayCntXY + j*arrayCntX + k] + x[(i-1)*arrayCntXY + j*arrayCntX + k] - 6.0 * x[i*arrayCntXY + j*arrayCntX + k]) * constVal;
          x[i*arrayCntXY + j*arrayCntX + k] += delta * v[i*arrayCntXY + j*arrayCntX + k];
          k++;
        }

        for(; k+1<lastImportantField[2]; k+=2){
          rCenter = _mm_load_pd(&x[i*arrayCntXY + j*arrayCntX + k]);//printf("%d  %d  %d\n",i, j, k);
          
          rLeft = _mm_load_pd(&x[i*arrayCntXY + j*arrayCntX + k - 2]);
          rRight = _mm_load_pd(&x[i*arrayCntXY + j*arrayCntX + k + 2]);
          
          rLeft = _mm_shuffle_pd(rLeft, rCenter, 0x00000001);
          rRight = _mm_shuffle_pd(rCenter, rRight, 0x00000001);
          rCenter = _mm_mul_pd(rCenter, rConst6);

          rResult = _mm_add_pd(rRight, rLeft);//printf("werte: %d %d %d %d   %d\n",i,j,k,arrayCntX,i*arrayCntXY + (j-1)*arrayCntX + k);
          rResult = _mm_add_pd(rResult, _mm_load_pd(&x[i*arrayCntXY + (j-1)*arrayCntX + k]));
          rResult = _mm_add_pd(rResult, _mm_load_pd(&x[i*arrayCntXY + (j+1)*arrayCntX + k]));
          rResult = _mm_add_pd(rResult, _mm_load_pd(&x[(i+1)*arrayCntXY + j*arrayCntX + k]));
          rResult = _mm_add_pd(rResult, _mm_load_pd(&x[(i-1)*arrayCntXY + j*arrayCntX + k]));

          rResult = _mm_sub_pd(rResult, rCenter);
          rResult = _mm_mul_pd(rResult, rConstVal1);
          rResult = _mm_add_pd(rResult, _mm_load_pd(&v[i*arrayCntXY + j*arrayCntX + k]));
          _mm_store_pd(&v[i*arrayCntXY + j*arrayCntX + k], rResult);


          rResult = _mm_load_pd(&v[i*arrayCntXY + j*arrayCntX + k]);
          rResult = _mm_mul_pd(rResult, rConstVal2);
          rCenter = _mm_load_pd(&x[i*arrayCntXY + j*arrayCntX + k]);
          rResult = _mm_add_pd(rResult, rCenter);
          _mm_store_pd(&x[i*arrayCntXY + j*arrayCntX + k], rResult);
        }

        if(k+1 == lastImportantField[2]){
          v[i*arrayCntXY + j*arrayCntX + k] += (x[i*arrayCntXY + j*arrayCntX + k + 1] + x[i*arrayCntXY + j*arrayCntX + k - 1] + x[i*arrayCntXY + (j+1)*arrayCntX + k] + x[i*arrayCntXY + (j-1)*arrayCntX + k] + x[(i+1)*arrayCntXY + j*arrayCntX + k] + x[(i-1)*arrayCntXY + j*arrayCntX + k] - 6.0 * x[i*arrayCntXY + j*arrayCntX + k]) * constVal;
          x[i*arrayCntXY + j*arrayCntX + k] += delta * v[i*arrayCntXY + j*arrayCntX + k];
        }
      }
    }

    //x
    if(wvVec->edgeArray[0])
      stepsLocal[2]=steps-1;
    else
      stepsLocal[2]++;

    if(wvVec->edgeArray[1])
      lastImportantField[2]=wvVec->nx+steps;
    else
      lastImportantField[2]--;
    //y
    if(wvVec->edgeArray[2])
      stepsLocal[1]=steps-1;
    else
      stepsLocal[1]++;

    if(wvVec->edgeArray[3])
      lastImportantField[1]=wvVec->ny+steps;
    else
      lastImportantField[1]--;
    //z
    if(wvVec->edgeArray[4])
      stepsLocal[0]=steps-1;
    else
      stepsLocal[0]++;

    if(wvVec->edgeArray[5])
      lastImportantField[0]=wvVec->nz+steps;
    else
      lastImportantField[0]--;
  }
}


void
x_direction_data_exchange_wavestate3d_sse2(pwavestate3dAllign wvVec1, pwavestate3dAllign wvVec2, int steps){
  real *x1 = wvVec1->x;
  real *v1 = wvVec1->v;
  real *x2 = wvVec2->x;
  real *v2 = wvVec2->v;

  int nx = wvVec1->nx;
  int ny = wvVec1->ny;
  int nz = wvVec1->nz;

  int arrayCntX = wvVec1->nx + steps * 2;
  int arrayCntXY = wvVec1->ny + steps * 2;
  arrayCntXY *= arrayCntX;

  int lastImportantField[4];
  lastImportantField[0] = nz+steps;
  lastImportantField[1] = ny+steps;
  lastImportantField[2] = nx+steps;
  lastImportantField[3] = nx+steps*2;
  int cnt1, cnt2, cnt3;

#ifdef __SSE2__
__m128d rData;
#endif

  for(cnt1=steps;cnt1<lastImportantField[0];++cnt1){
    for(cnt2=steps;cnt2<lastImportantField[1];++cnt2){
      //left to right
      cnt3=nx;
      if((cnt1*arrayCntXY + cnt2*arrayCntX + cnt3) % 2){
        x2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx] = x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3];
        v2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx] = v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3];
        cnt3++;
      }

      for(;cnt3+1<lastImportantField[2];cnt3+=2){
        rData = _mm_load_pd(&x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3]);
        _mm_store_pd(&x2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx], rData);
        rData = _mm_load_pd(&v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3]);
        _mm_store_pd(&v2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx], rData);
      }

      if(cnt3+1 == lastImportantField[2]){
        x2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx] = x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3];
        v2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx] = v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3];
      }

      //right to left
      cnt3=nx;
      if((cnt1*arrayCntXY + cnt2*arrayCntX + cnt3) % 2){
        x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3] = x2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx];
        v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3] = v2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx];
        cnt3++;
      }

      for(;cnt3+1<lastImportantField[3];cnt3+=2){
        rData = _mm_load_pd(&x2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx]);
        _mm_store_pd(&x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3], rData);
        rData = _mm_load_pd(&v2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx]);
        _mm_store_pd(&v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3], rData);
      }

      if(cnt3+1 == lastImportantField[3]){
        x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3] = x2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx];
        v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3] = v2[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3-nx];
      }
    }
  }

}


void
y_direction_data_exchange_wavestate3d_sse2(pwavestate3dAllign wvVec1, pwavestate3dAllign wvVec2, int steps){
  real *x1 = wvVec1->x;
  real *v1 = wvVec1->v;
  real *x2 = wvVec2->x;
  real *v2 = wvVec2->v;

  int nx = wvVec1->nx;
  int ny = wvVec1->ny;
  int nz = wvVec1->nz;

  int arrayCntX = wvVec1->nx + steps * 2;
  int arrayCntXY = wvVec1->ny + steps * 2;
  arrayCntXY *= arrayCntX;

  int lastImportantField[4];
  lastImportantField[0] = nz+steps;
  lastImportantField[1] = ny+steps;
  lastImportantField[2] = ny+steps*2;
  lastImportantField[3] = nx+steps*2;
  int cnt1, cnt2, cnt3;

#ifdef __SSE2__
__m128d rData;
#endif

  for(cnt1=steps;cnt1<lastImportantField[0];++cnt1){
    for(cnt2=ny;cnt2<lastImportantField[1];++cnt2){
      for(cnt3=0;cnt3<lastImportantField[3];cnt3+=2){
        rData = _mm_load_pd(&x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3]);
        _mm_store_pd(&x2[cnt1*arrayCntXY + (cnt2-ny)*arrayCntX + cnt3], rData);
        rData = _mm_load_pd(&v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3]);
        _mm_store_pd(&v2[cnt1*arrayCntXY + (cnt2-ny)*arrayCntX + cnt3], rData);
      }
    }

    for(cnt2=ny+steps;cnt2<lastImportantField[2];++cnt2){
      for(cnt3=0;cnt3<lastImportantField[3];cnt3+=2){
        rData = _mm_load_pd(&x2[cnt1*arrayCntXY + (cnt2-ny)*arrayCntX + cnt3]);
        _mm_store_pd(&x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3], rData);
        rData = _mm_load_pd(&v2[cnt1*arrayCntXY + (cnt2-ny)*arrayCntX + cnt3]);
        _mm_store_pd(&v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3], rData);
      }
    }
  }

}


void
z_direction_data_exchange_wavestate3d_sse2(pwavestate3dAllign wvVec1, pwavestate3dAllign wvVec2, int steps){
  real *x1 = wvVec1->x;
  real *v1 = wvVec1->v;
  real *x2 = wvVec2->x;
  real *v2 = wvVec2->v;

  int nx = wvVec1->nx;
  int ny = wvVec1->ny;
  int nz = wvVec1->nz;

  int arrayCntX = wvVec1->nx + steps * 2;
  int arrayCntXY = wvVec1->ny + steps * 2;
  arrayCntXY *= arrayCntX;

  int lastImportantField[4];
  lastImportantField[0] = nz+steps;
  lastImportantField[1] = nz+steps*2;
  lastImportantField[2] = ny+steps*2;
  lastImportantField[3] = nx+steps*2;
  int cnt1, cnt2, cnt3;

#ifdef __SSE2__
__m128d rData;
#endif

  for(cnt1=nz;cnt1<lastImportantField[0];++cnt1){
    for(cnt2=0;cnt2<lastImportantField[2];++cnt2){
      for(cnt3=0;cnt3<lastImportantField[3];cnt3+=2){
        rData = _mm_load_pd(&x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3]);
        _mm_store_pd(&x2[(cnt1-nz)*arrayCntXY + cnt2*arrayCntX + cnt3], rData);
        rData = _mm_load_pd(&v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3]);
        _mm_store_pd(&v2[(cnt1-nz)*arrayCntXY + cnt2*arrayCntX + cnt3], rData);
      }
    }
  }

  for(cnt1=nz+steps;cnt1<lastImportantField[1];++cnt1){
    for(cnt2=0;cnt2<lastImportantField[2];++cnt2){
      for(cnt3=0;cnt3<lastImportantField[3];cnt3+=2){
        rData = _mm_load_pd(&x2[(cnt1-nz)*arrayCntXY + cnt2*arrayCntX + cnt3]);
        _mm_store_pd(&x1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3], rData);
        rData = _mm_load_pd(&v2[(cnt1-nz)*arrayCntXY + cnt2*arrayCntX + cnt3]);
        _mm_store_pd(&v1[cnt1*arrayCntXY + cnt2*arrayCntX + cnt3], rData);
      }
    }
  }

}
