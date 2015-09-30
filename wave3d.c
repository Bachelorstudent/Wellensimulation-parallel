
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
//#include <omp.h>
#include <mpi.h>

#include <immintrin.h>
#include <mm_malloc.h>

#include "wavestate3d.h"

/*
  ausführen:
  ./wave3d nx ny nz steps iterations xBlöcke yBlöcke zBlöcke

  nx muss gerade sein
  steps darf höchstens min(nx,ny,nz) sein
  xBlöcke, yBlöcke und zBlöcke müssen mindestens 1 sein 
  mindestens 2 Blöcke, sonst Endlosschleife

  nx, ny, nz bezeichnet die Kantenlängen eines Blockes(mx,my,mz aus der Arbeit). 
  */
int
main(int argc, char **argv)
{

  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  float time = 0.0, a_time = 0.0, b_time = 0.0, c_time = 0.0;
  clock_t s, e, a_s, a_e, b_s, b_e, c_s, c_e;

//  double time_omp = 0.0, a_time_omp = 0.0, b_time_omp = 0.0, c_time_omp = 0.0, s_omp, e_omp, a_s_omp, a_e_omp, b_s_omp, b_e_omp, c_s_omp, c_e_omp;

  double time_mpi = 0.0, a_time_mpi = 0.0, b_time_mpi = 0.0, c_time_mpi = 0.0, s_mpi, e_mpi, a_s_mpi, a_e_mpi, b_s_mpi, b_e_mpi, c_s_mpi, c_e_mpi;

  pwavestate3dAllign*** wvVecCuboid;
  real delta;
  int cnt, cnt2, cnt3, cnt4, cnt5, bufferCnt;
  int nx, ny, nz, steps, iterations, cuboidNumberX, cuboidNumberY, cuboidNumberZ;//, threads;

  //n = 400;
  //steps = 10;
  nx = atoi(argv[argc-8]);
  ny = atoi(argv[argc-7]);
  nz = atoi(argv[argc-6]);
  steps = atoi(argv[argc-5]);
  iterations = atoi(argv[argc-4]);
  cuboidNumberX = atoi(argv[argc-3]);
  cuboidNumberY = atoi(argv[argc-2]);
  cuboidNumberZ = atoi(argv[argc-1]);
//  threads = atoi(argv[argc-1]);
  delta = 0.0001;

  int first_mpi = (cuboidNumberX * cuboidNumberY * cuboidNumberZ * rank) / size;
  for(;first_mpi%cuboidNumberX;first_mpi++);
  int last_mpi = (cuboidNumberX * cuboidNumberY * cuboidNumberZ * (rank+1)) / size;
  for(;last_mpi%cuboidNumberX;last_mpi++);
//printf("rank: %d  first: %d  last: %d\n",rank,first_mpi,last_mpi);
  int first_mpi_array[size];
  int last_mpi_array[size];

  for(cnt=0;cnt<size;cnt++){
    first_mpi_array[cnt] = (cuboidNumberX * cuboidNumberY * cuboidNumberZ * cnt) / size;
    for(;first_mpi_array[cnt]%cuboidNumberX;first_mpi_array[cnt]++);
    last_mpi_array[cnt] = (cuboidNumberX * cuboidNumberY * cuboidNumberZ * (cnt+1)) / size;
    for(;last_mpi_array[cnt]%cuboidNumberX;last_mpi_array[cnt]++);
  }


/*
  pmpiDataExchangeStruct mpiBufferGetTopX;
  pmpiDataExchangeStruct mpiBufferSendTopX;
  pmpiDataExchangeStruct mpiBufferGetBottomX;
  pmpiDataExchangeStruct mpiBufferSendBottomX;
*/
  pmpiDataExchangeStruct mpiBufferGetTopY;
  pmpiDataExchangeStruct mpiBufferSendTopY;
  pmpiDataExchangeStruct mpiBufferGetBottomY;
  pmpiDataExchangeStruct mpiBufferSendBottomY;

  pmpiDataExchangeStruct mpiBufferGetTopZ;
  pmpiDataExchangeStruct mpiBufferSendTopZ;
  pmpiDataExchangeStruct mpiBufferGetBottomZ;
  pmpiDataExchangeStruct mpiBufferSendBottomZ;

  int bufferLength[4];
  bufferLength[0]=0;
  bufferLength[1]=0;
  bufferLength[2]=0;
  bufferLength[3]=0;
  int cuboidArray[4][(nx+steps*2)*(ny+steps*2)];
  cuboidArray[0][0]=0;
  cuboidArray[1][0]=0;
  cuboidArray[2][0]=0;
  cuboidArray[3][0]=0;

//  wvVecCuboid = new_wavestate3d_cuboid(nx, ny, nz, 1.0/(nx+1), 1.0, steps, cuboidNumberX, cuboidNumberY, cuboidNumberZ);
  wvVecCuboid = new_wavestate3d_cuboid(nx, ny, nz, 1.0/(nx+1), 1.0, steps, cuboidNumberX, cuboidNumberY, cuboidNumberZ, first_mpi, last_mpi);

//buffer
  for(cnt=first_mpi,cnt2=0;cnt-first_mpi<cuboidNumberY && cnt<last_mpi;cnt++){
    if(cnt-cuboidNumberX < first_mpi && cnt-cuboidNumberX >= 0 && (cnt%(cuboidNumberX*cuboidNumberY))/cuboidNumberX != 0){
      cuboidArray[0][cnt2]=cnt;
      cnt2++;
    }
  }
  mpiBufferGetTopY = new_mpiDataExchangeStruct(1, nx, ny, nz, steps, cnt2);
  mpiBufferSendTopY = new_mpiDataExchangeStruct(1, nx, ny, nz, steps, cnt2);
  bufferLength[0] = cnt2;

  for(cnt=last_mpi-1,cnt2=0;last_mpi-cnt<=cuboidNumberY && cnt>=first_mpi;cnt--){
    if(cnt+cuboidNumberX >= last_mpi && cnt+cuboidNumberX < cuboidNumberX*cuboidNumberY*cuboidNumberZ && (cnt%(cuboidNumberX*cuboidNumberY))/cuboidNumberX != cuboidNumberY-1){
      cuboidArray[1][cnt2]=cnt;
      cnt2++;
    }
  }
  mpiBufferGetBottomY = new_mpiDataExchangeStruct(1, nx, ny, nz, steps, cnt2);
  mpiBufferSendBottomY = new_mpiDataExchangeStruct(1, nx, ny, nz, steps, cnt2);
  bufferLength[1] = cnt2;

  for(cnt=first_mpi,cnt2=0;cnt-first_mpi<cuboidNumberX*cuboidNumberY && cnt<last_mpi;cnt++){
    if(cnt-(cuboidNumberX*cuboidNumberY) < first_mpi && cnt-(cuboidNumberX*cuboidNumberY) >= 0 && cnt/(cuboidNumberX*cuboidNumberY) != 0){
      cuboidArray[2][cnt2]=cnt;
      cnt2++;
    }
  }
  mpiBufferGetTopZ = new_mpiDataExchangeStruct(2, nx, ny, nz, steps, cnt2);
  mpiBufferSendTopZ = new_mpiDataExchangeStruct(2, nx, ny, nz, steps, cnt2);
  bufferLength[2] = cnt2;

  for(cnt=last_mpi-1,cnt2=0;last_mpi-cnt<=cuboidNumberX*cuboidNumberY && cnt>=first_mpi;cnt--){
    if(cnt+(cuboidNumberX*cuboidNumberY) >= last_mpi && cnt+(cuboidNumberX*cuboidNumberY) < cuboidNumberX*cuboidNumberY*cuboidNumberZ && cnt/(cuboidNumberX*cuboidNumberY) != cuboidNumberZ-1){
      cuboidArray[3][cnt2]=cnt;
      cnt2++;

    }
  }
  mpiBufferGetBottomZ = new_mpiDataExchangeStruct(2, nx, ny, nz, steps, cnt2);
  mpiBufferSendBottomZ = new_mpiDataExchangeStruct(2, nx, ny, nz, steps, cnt2);
  bufferLength[3] = cnt2;

  zero_wavestate3d_cuboid(wvVecCuboid, steps, cuboidNumberX, cuboidNumberY, cuboidNumberZ, first_mpi, last_mpi);
  
  if(rank==0)
  	boundary_wavestate3d_cuboid(wvVecCuboid, 0.0, steps);

  s = clock();
//  s_omp = omp_get_wtime();
  s_mpi = MPI_Wtime();
  for(cnt=0;cnt<iterations;cnt++){

    a_s = clock();
//    a_s_omp = omp_get_wtime();
    a_s_mpi = MPI_Wtime();

    for(cnt2=first_mpi;cnt2<last_mpi;++cnt2){
      leapfrog_wavestate3d_cuboid(wvVecCuboid[cnt2/(cuboidNumberX*cuboidNumberY)][(cnt2%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cnt2%cuboidNumberX], delta, steps);
    }

    a_e = clock();
//    a_e_omp = omp_get_wtime();
    a_e_mpi = MPI_Wtime();
    a_time += (double) (a_e - a_s) / (double) CLOCKS_PER_SEC;
//    a_time_omp += a_e_omp - a_s_omp;
    a_time_mpi += a_e_mpi - a_s_mpi;
    
    b_s = clock();
//    b_s_omp = omp_get_wtime();
    b_s_mpi = MPI_Wtime();
    

    //datenaustausch lokal
    for(cnt2=first_mpi;cnt2+1<last_mpi;++cnt2){
      if(cnt2%cuboidNumberX != cuboidNumberX-1){
        x_direction_data_exchange_wavestate3d_sse2(wvVecCuboid[cnt2/(cuboidNumberX*cuboidNumberY)][(cnt2%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cnt2%cuboidNumberX], wvVecCuboid[cnt2/(cuboidNumberX*cuboidNumberY)][(cnt2%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][(cnt2%cuboidNumberX)+1], steps);
      }
    }

    for(cnt2=first_mpi;cnt2+cuboidNumberX<last_mpi;++cnt2){
      if((cnt2%(cuboidNumberX*cuboidNumberY))/cuboidNumberX != cuboidNumberY-1){
        y_direction_data_exchange_wavestate3d_sse2(wvVecCuboid[cnt2/(cuboidNumberX*cuboidNumberY)][(cnt2%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cnt2%cuboidNumberX], wvVecCuboid[cnt2/(cuboidNumberX*cuboidNumberY)][((cnt2%(cuboidNumberX*cuboidNumberY))/cuboidNumberX)+1][cnt2%cuboidNumberX], steps);
      }
    }

    for(cnt2=first_mpi;cnt2+cuboidNumberX*cuboidNumberY<last_mpi;++cnt2){
      if(cnt2/(cuboidNumberX*cuboidNumberY) != cuboidNumberZ-1){
        z_direction_data_exchange_wavestate3d_sse2(wvVecCuboid[cnt2/(cuboidNumberX*cuboidNumberY)][(cnt2%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cnt2%cuboidNumberX], wvVecCuboid[(cnt2/(cuboidNumberX*cuboidNumberY))+1][(cnt2%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cnt2%cuboidNumberX], steps);
      }
    }
    

    b_e = clock();
//    b_e_omp = omp_get_wtime();
    b_e_mpi = MPI_Wtime();
    b_time += (double) (b_e - b_s) / (double) CLOCKS_PER_SEC;
//    b_time_omp += b_e_omp - b_s_omp;
    b_time_mpi += b_e_mpi - b_s_mpi;

    c_s = clock();
//    c_s_omp = omp_get_wtime();
    c_s_mpi = MPI_Wtime();
    

    //datenaustausch mpi
//y
    for(cnt2=0;cnt2<bufferLength[0];cnt2++){
      for(cnt3=0,bufferCnt=0;cnt3<nz;cnt3++){
        for(cnt4=0;cnt4<steps;cnt4++){
          for(cnt5=0;cnt5<(nx+steps*2);cnt5++,bufferCnt++){
            mpiBufferSendTopY->xv[bufferCnt + cnt2*2*(nx+steps*2)*steps*nz] = wvVecCuboid[cuboidArray[0][cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[0][cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[0][cnt2]%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*(steps+cnt4) + cnt5];
            mpiBufferSendTopY->xv[bufferCnt + (nx+steps*2)*steps*nz + cnt2*2*(nx+steps*2)*steps*nz] = wvVecCuboid[cuboidArray[0][cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[0][cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[0][cnt2]%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*(steps+cnt4) + cnt5];
          }
        }
      }
    }

    for(cnt2=0;cnt2<bufferLength[1];cnt2++){
      for(cnt3=0,bufferCnt=0;cnt3<nz;cnt3++){
        for(cnt4=0;cnt4<steps;cnt4++){
          for(cnt5=0;cnt5<(nx+steps*2);cnt5++,bufferCnt++){
            mpiBufferSendBottomY->xv[bufferCnt + cnt2*2*(nx+steps*2)*steps*nz] = wvVecCuboid[cuboidArray[1][bufferLength[1]-1-cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[1][bufferLength[1]-1-cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[1][bufferLength[1]-1-cnt2]%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*(ny+cnt4) + cnt5];
            mpiBufferSendBottomY->xv[bufferCnt + (nx+steps*2)*steps*nz + cnt2*2*(nx+steps*2)*steps*nz] = wvVecCuboid[cuboidArray[1][bufferLength[1]-1-cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[1][bufferLength[1]-1-cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[1][bufferLength[1]-1-cnt2]%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*(ny+cnt4) + cnt5];
          }
        }
      }
    }

//z
    for(cnt2=0;cnt2<bufferLength[2];cnt2++){
      for(cnt3=0,bufferCnt=0;cnt3<steps;cnt3++){
        for(cnt4=0;cnt4<(ny+steps*2);cnt4++){
          for(cnt5=0;cnt5<(nx+steps*2);cnt5++,bufferCnt++){
            mpiBufferSendTopZ->xv[bufferCnt + cnt2*2*(nx+steps*2)*(ny+steps*2)*steps] = wvVecCuboid[cuboidArray[2][cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[2][cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[2][cnt2]%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*cnt4 + cnt5];
            mpiBufferSendTopZ->xv[bufferCnt + (nx+steps*2)*(ny+steps*2)*steps + cnt2*2*(nx+steps*2)*(ny+steps*2)*steps] = wvVecCuboid[cuboidArray[2][cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[2][cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[2][cnt2]%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*cnt4 + cnt5];
          }
        }
      }
    }

    for(cnt2=0;cnt2<bufferLength[3];cnt2++){
      for(cnt3=0,bufferCnt=0;cnt3<steps;cnt3++){
        for(cnt4=0;cnt4<(ny+steps*2);cnt4++){
          for(cnt5=0;cnt5<(nx+steps*2);cnt5++,bufferCnt++){
            mpiBufferSendBottomZ->xv[bufferCnt + cnt2*2*(ny+steps*2)*(nx+steps*2)*steps] = wvVecCuboid[cuboidArray[3][bufferLength[3]-1-cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[3][bufferLength[3]-1-cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[3][bufferLength[3]-1-cnt2]%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(nz+cnt3) + (nx+steps*2)*cnt4 + cnt5];
            mpiBufferSendBottomZ->xv[bufferCnt + (ny+steps*2)*(nx+steps*2)*steps + cnt2*2*(ny+steps*2)*(nx+steps*2)*steps] = wvVecCuboid[cuboidArray[3][bufferLength[3]-1-cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[3][bufferLength[3]-1-cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[3][bufferLength[3]-1-cnt2]%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(nz+cnt3) + (nx+steps*2)*cnt4 + cnt5];
          }
        }
      }
    }

//send/receive
//y

    MPI_Request topY, bottomY;
    if((cuboidArray[0][0]-cuboidNumberX < first_mpi && cuboidArray[0][0]-cuboidNumberX >= 0 && (cuboidArray[0][0]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX != 0) && (cuboidArray[1][0]+cuboidNumberX >= last_mpi && cuboidArray[1][0]+cuboidNumberX < cuboidNumberX*cuboidNumberY*cuboidNumberZ && (cuboidArray[1][0]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX != cuboidNumberY-1)){
      MPI_Isend(mpiBufferSendTopY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[0],MPI_BYTE,rank-1,0,MPI_COMM_WORLD,&topY);
      MPI_Isend(mpiBufferSendBottomY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[1],MPI_BYTE,rank+1,0,MPI_COMM_WORLD,&bottomY);
      MPI_Irecv(mpiBufferGetTopY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[0],MPI_BYTE,rank-1,0,MPI_COMM_WORLD,&topY);
      MPI_Irecv(mpiBufferGetBottomY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[1],MPI_BYTE,rank+1,0,MPI_COMM_WORLD,&bottomY);
//      MPI_Recv(mpiBufferGetTopY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[0],MPI_BYTE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//      MPI_Recv(mpiBufferGetBottomY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[1],MPI_BYTE,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Wait(&topY,MPI_STATUS_IGNORE);
      MPI_Wait(&bottomY,MPI_STATUS_IGNORE);
    }else if(cuboidArray[0][0]-cuboidNumberX < first_mpi && cuboidArray[0][0]-cuboidNumberX >= 0 && (cuboidArray[0][0]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX != 0){
      MPI_Isend(mpiBufferSendTopY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[0],MPI_BYTE,rank-1,0,MPI_COMM_WORLD,&topY);
      MPI_Irecv(mpiBufferGetTopY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[0],MPI_BYTE,rank-1,0,MPI_COMM_WORLD,&topY);
//      MPI_Recv(mpiBufferGetTopY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[0],MPI_BYTE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Wait(&topY,MPI_STATUS_IGNORE);
    }else if(cuboidArray[1][0]+cuboidNumberX >= last_mpi && cuboidArray[1][0]+cuboidNumberX < cuboidNumberX*cuboidNumberY*cuboidNumberZ && (cuboidArray[1][0]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX != cuboidNumberY-1){
      MPI_Isend(mpiBufferSendBottomY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[1],MPI_BYTE,rank+1,0,MPI_COMM_WORLD,&bottomY);
      MPI_Irecv(mpiBufferGetBottomY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[1],MPI_BYTE,rank+1,0,MPI_COMM_WORLD,&bottomY);
//      MPI_Recv(mpiBufferGetBottomY->xv,(nx+steps*2)*steps*nz*8*2*bufferLength[1],MPI_BYTE,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Wait(&bottomY,MPI_STATUS_IGNORE);
    }

//Daten aus buffer in Blöcke kopieren
//y
    for(cnt2=0;cnt2<bufferLength[0];cnt2++){
      for(cnt3=0,bufferCnt=0;cnt3<nz;cnt3++){
        for(cnt4=0;cnt4<steps;cnt4++){
          for(cnt5=0;cnt5<(nx+steps*2);cnt5++,bufferCnt++){
            wvVecCuboid[cuboidArray[0][cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[0][cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[0][cnt2]%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*cnt4 + cnt5] = mpiBufferGetTopY->xv[bufferCnt + cnt2*2*(nx+steps*2)*steps*nz];
            wvVecCuboid[cuboidArray[0][cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[0][cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[0][cnt2]%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*cnt4 + cnt5] = mpiBufferGetTopY->xv[bufferCnt + (nx+steps*2)*steps*nz + cnt2*2*(nx+steps*2)*steps*nz];
          }
        }
      }
    }

    for(cnt2=0;cnt2<bufferLength[1];cnt2++){
      for(cnt3=0,bufferCnt=0;cnt3<nz;cnt3++){
        for(cnt4=0;cnt4<steps;cnt4++){
          for(cnt5=0;cnt5<(nx+steps*2);cnt5++,bufferCnt++){
            wvVecCuboid[cuboidArray[1][bufferLength[1]-1-cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[1][bufferLength[1]-1-cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[1][bufferLength[1]-1-cnt2]%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*(ny+steps+cnt4) + cnt5] = mpiBufferGetBottomY->xv[bufferCnt + cnt2*2*(nx+steps*2)*steps*nz];
            wvVecCuboid[cuboidArray[1][bufferLength[1]-1-cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[1][bufferLength[1]-1-cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[1][bufferLength[1]-1-cnt2]%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt3) + (nx+steps*2)*(ny+steps+cnt4) + cnt5] = mpiBufferGetBottomY->xv[bufferCnt + (nx+steps*2)*steps*nz + cnt2*2*(nx+steps*2)*steps*nz];
          }
        }
      }
    }

//send/receive
//z
    int tmpTargetRankTop = rank-1;
    int anotherTargetRankTop = 0;
    int tmpCuboidNumberTopRankA = bufferLength[2];
    int tmpCuboidNumberTopRankB = 0;
    int tmpTargetRankBottom = rank+1;
    int anotherTargetRankBottom = 0;
    int tmpCuboidNumberBottomRankA = bufferLength[3];
    int tmpCuboidNumberBottomRankB = 0;

    MPI_Request topZ1, topZ2, bottomZ1, bottomZ2;
    if((cuboidArray[2][0]-cuboidNumberX*cuboidNumberY < first_mpi && cuboidArray[2][0]-cuboidNumberX*cuboidNumberY >= 0 && cuboidArray[2][0]/(cuboidNumberX*cuboidNumberY) != 0) && (cuboidArray[3][0]+cuboidNumberX*cuboidNumberY >= last_mpi && cuboidArray[3][0]+cuboidNumberX*cuboidNumberY < cuboidNumberX*cuboidNumberY*cuboidNumberZ && cuboidArray[3][0]/(cuboidNumberX*cuboidNumberY) != cuboidNumberZ-1)){
      for(;cuboidArray[2][0]-cuboidNumberX*cuboidNumberY<first_mpi_array[tmpTargetRankTop];tmpTargetRankTop--);
      if(cuboidArray[2][bufferLength[2]-1]-cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankTop]){
        anotherTargetRankTop=1;
        tmpCuboidNumberTopRankA--;
        tmpCuboidNumberTopRankB++;
        for(cnt2=bufferLength[2]-2;cnt2>=0 && cuboidArray[2][cnt2]-cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankTop];cnt2--,tmpCuboidNumberTopRankA--,tmpCuboidNumberTopRankB++);
      }
      for(;cuboidArray[3][bufferLength[3]-1]+cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankBottom];tmpTargetRankBottom++);
      if(cuboidArray[3][0]+cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankBottom]){
        anotherTargetRankBottom=1;
        tmpCuboidNumberBottomRankA--;
        tmpCuboidNumberBottomRankB++;
        for(cnt2=1;cnt2<bufferLength[3] && cuboidArray[3][cnt2]+cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankBottom];cnt2++,tmpCuboidNumberBottomRankA--,tmpCuboidNumberBottomRankB++);
      }
      if(anotherTargetRankTop && anotherTargetRankBottom){
        MPI_Isend(mpiBufferSendTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Isend(mpiBufferSendTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,&topZ2);
        MPI_Isend(mpiBufferSendBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Isend(mpiBufferSendBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,&bottomZ2);
        MPI_Irecv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Irecv(mpiBufferGetTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,&topZ2);
        MPI_Irecv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Irecv(mpiBufferGetBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,&bottomZ2);
//        MPI_Recv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ1,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ2,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ1,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ2,MPI_STATUS_IGNORE);
      }else if(anotherTargetRankTop){
        MPI_Isend(mpiBufferSendTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Isend(mpiBufferSendTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,&topZ2);
        MPI_Isend(mpiBufferSendBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Irecv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Irecv(mpiBufferGetTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,&topZ2);
        MPI_Irecv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
//        MPI_Recv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ1,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ2,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ1,MPI_STATUS_IGNORE);
      }else if(anotherTargetRankBottom){
        MPI_Isend(mpiBufferSendTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Isend(mpiBufferSendBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Isend(mpiBufferSendBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,&bottomZ2);
        MPI_Irecv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Irecv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Irecv(mpiBufferGetBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,&bottomZ2);
//        MPI_Recv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ1,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ1,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ2,MPI_STATUS_IGNORE);
      }else{
        MPI_Isend(mpiBufferSendTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Isend(mpiBufferSendBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Irecv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Irecv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
//        MPI_Recv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ1,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ1,MPI_STATUS_IGNORE);
      }
    }else if(cuboidArray[2][0]-cuboidNumberX*cuboidNumberY < first_mpi && cuboidArray[2][0]-cuboidNumberX*cuboidNumberY >= 0 && cuboidArray[2][0]/(cuboidNumberX*cuboidNumberY) != 0){
      for(;cuboidArray[2][0]-cuboidNumberX*cuboidNumberY<first_mpi_array[tmpTargetRankTop];tmpTargetRankTop--);
      if(cuboidArray[2][bufferLength[2]-1]-cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankTop]){
        anotherTargetRankTop=1;
        tmpCuboidNumberTopRankA--;
        tmpCuboidNumberTopRankB++;
        for(cnt2=bufferLength[2]-2;cnt2>=0 && cuboidArray[2][cnt2]-cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankTop];cnt2--,tmpCuboidNumberTopRankA--,tmpCuboidNumberTopRankB++);
      }
      if(anotherTargetRankTop){
        MPI_Isend(mpiBufferSendTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Isend(mpiBufferSendTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,&topZ2);
        MPI_Irecv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Irecv(mpiBufferGetTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,&topZ2);
//        MPI_Recv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetTopZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberTopRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankB,MPI_BYTE,tmpTargetRankTop+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ1,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ2,MPI_STATUS_IGNORE);
      }else{
        MPI_Isend(mpiBufferSendTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
        MPI_Irecv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,&topZ1);
//        MPI_Recv(mpiBufferGetTopZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberTopRankA,MPI_BYTE,tmpTargetRankTop,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Wait(&topZ1,MPI_STATUS_IGNORE);
      }
    }else if(cuboidArray[3][0]+cuboidNumberX*cuboidNumberY >= last_mpi && cuboidArray[3][0]+cuboidNumberX*cuboidNumberY < cuboidNumberX*cuboidNumberY*cuboidNumberZ && cuboidArray[3][0]/(cuboidNumberX*cuboidNumberY) != cuboidNumberZ-1){
      for(;cuboidArray[3][bufferLength[3]-1]+cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankBottom];tmpTargetRankBottom++);
      if(cuboidArray[3][0]+cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankBottom]){
        anotherTargetRankBottom=1;
        tmpCuboidNumberBottomRankA--;
        tmpCuboidNumberBottomRankB++;
        for(cnt2=1;cnt2<bufferLength[3]-1 && cuboidArray[3][cnt2]+cuboidNumberX*cuboidNumberY>=last_mpi_array[tmpTargetRankTop];cnt2++,tmpCuboidNumberBottomRankA--,tmpCuboidNumberBottomRankB++);
      }
      if(anotherTargetRankBottom){
        MPI_Isend(mpiBufferSendBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Isend(mpiBufferSendBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,&bottomZ2);
        MPI_Irecv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Irecv(mpiBufferGetBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,&bottomZ2);
//        MPI_Recv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//        MPI_Recv(mpiBufferGetBottomZ->xv+(nx+steps*2)*(ny+steps*2)*steps*2*tmpCuboidNumberBottomRankA,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankB,MPI_BYTE,tmpTargetRankBottom+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ1,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ2,MPI_STATUS_IGNORE);
      }else{
        MPI_Isend(mpiBufferSendBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
        MPI_Irecv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,&bottomZ1);
//        MPI_Recv(mpiBufferGetBottomZ->xv,(nx+steps*2)*(ny+steps*2)*steps*8*2*tmpCuboidNumberBottomRankA,MPI_BYTE,tmpTargetRankBottom,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Wait(&bottomZ1,MPI_STATUS_IGNORE);
      }
    }    
    
//Daten aus buffer in Blöcke kopieren
//z
    for(cnt2=0;cnt2<bufferLength[2];cnt2++){
      for(cnt3=0,bufferCnt=0;cnt3<steps;cnt3++){
        for(cnt4=0;cnt4<(ny+steps*2);cnt4++){
          for(cnt5=0;cnt5<(nx+steps*2);cnt5++,bufferCnt++){
            wvVecCuboid[cuboidArray[2][cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[2][cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[2][cnt2]%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*cnt3 + (nx+steps*2)*cnt4 + cnt5] = mpiBufferGetTopZ->xv[bufferCnt + cnt2*2*(nx+steps*2)*(ny+steps*2)*steps];
            wvVecCuboid[cuboidArray[2][cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[2][cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[2][cnt2]%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*cnt3 + (nx+steps*2)*cnt4 + cnt5] = mpiBufferGetTopZ->xv[bufferCnt + (nx+steps*2)*(ny+steps*2)*steps + cnt2*2*(nx+steps*2)*(ny+steps*2)*steps];
          }
        }
      }
    }    

    for(cnt2=0;cnt2<bufferLength[3];cnt2++){
      for(cnt3=0,bufferCnt=0;cnt3<steps;cnt3++){
        for(cnt4=0;cnt4<(ny+steps*2);cnt4++){
          for(cnt5=0;cnt5<(nx+steps*2);cnt5++,bufferCnt++){
            wvVecCuboid[cuboidArray[3][bufferLength[3]-1-cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[3][bufferLength[3]-1-cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[3][bufferLength[3]-1-cnt2]%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(nz+steps+cnt3) + (nx+steps*2)*cnt4 + cnt5] = mpiBufferGetBottomZ->xv[bufferCnt + cnt2*2*(nx+steps*2)*(ny+steps*2)*steps];
            wvVecCuboid[cuboidArray[3][bufferLength[3]-1-cnt2]/(cuboidNumberX*cuboidNumberY)][(cuboidArray[3][bufferLength[3]-1-cnt2]%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][cuboidArray[3][bufferLength[3]-1-cnt2]%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(nz+steps+cnt3) + (nx+steps*2)*cnt4 + cnt5] = mpiBufferGetBottomZ->xv[bufferCnt + (nx+steps*2)*(ny+steps*2)*steps + cnt2*2*(nx+steps*2)*(ny+steps*2)*steps];
          }
        }
      }
    }

//Spezialfälle
    if(last_mpi-first_mpi>cuboidNumberX*cuboidNumberY && (last_mpi-first_mpi)%(cuboidNumberX*cuboidNumberY)){
      //ecke oben
      if(first_mpi%(cuboidNumberX*cuboidNumberY)){
        for(cnt2=0;cnt2<nz;cnt2++){
          for(cnt3=0;cnt3<steps;cnt3++){
            for(cnt4=0;cnt4<steps;cnt4++){
              //austausch zwischen first_mpi und first_mpi+cuboidNumberX*cuboidNumberY
              wvVecCuboid[(first_mpi+cuboidNumberX*cuboidNumberY)/(cuboidNumberX*cuboidNumberY)][((first_mpi+cuboidNumberX*cuboidNumberY)%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][(first_mpi+cuboidNumberX*cuboidNumberY)%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt2) + (nx+steps*2)*cnt3 + cnt4] = wvVecCuboid[first_mpi/(cuboidNumberX*cuboidNumberY)][(first_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][first_mpi%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt2) + (nx+steps*2)*(ny+cnt3) + cnt4];
              wvVecCuboid[(first_mpi+cuboidNumberX*cuboidNumberY)/(cuboidNumberX*cuboidNumberY)][((first_mpi+cuboidNumberX*cuboidNumberY)%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][(first_mpi+cuboidNumberX*cuboidNumberY)%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt2) + (nx+steps*2)*cnt3 + cnt4] = wvVecCuboid[first_mpi/(cuboidNumberX*cuboidNumberY)][(first_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][first_mpi%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt2) + (nx+steps*2)*(ny+cnt3) + cnt4];
            }
          }
        }
      }
      
      if(last_mpi%(cuboidNumberX*cuboidNumberY)){
        //ecke unten
        for(cnt2=0;cnt2<nz;cnt2++){
          for(cnt3=0;cnt3<steps;cnt3++){
            for(cnt4=0;cnt4<steps;cnt4++){
              //austausch zwischen last_mpi und last_mpi-cuboidNumberX*cuboidNumberY
              wvVecCuboid[(last_mpi-cuboidNumberX*cuboidNumberY)/(cuboidNumberX*cuboidNumberY)][((last_mpi-cuboidNumberX*cuboidNumberY)%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][(last_mpi-cuboidNumberX*cuboidNumberY)%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt2) + (nx+steps*2)*(steps+ny+cnt3) + steps + nx + cnt4] = wvVecCuboid[last_mpi/(cuboidNumberX*cuboidNumberY)][(last_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][last_mpi%cuboidNumberX]->x[(ny+steps*2)*(nx+steps*2)*(steps+cnt2) + (nx+steps*2)*(steps+cnt3) + steps + nx + cnt4];
              wvVecCuboid[(last_mpi-cuboidNumberX*cuboidNumberY)/(cuboidNumberX*cuboidNumberY)][((last_mpi-cuboidNumberX*cuboidNumberY)%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][(last_mpi-cuboidNumberX*cuboidNumberY)%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt2) + (nx+steps*2)*(steps+ny+cnt3) + steps + nx + cnt4] = wvVecCuboid[last_mpi/(cuboidNumberX*cuboidNumberY)][(last_mpi%(cuboidNumberX*cuboidNumberY))/cuboidNumberX][last_mpi%cuboidNumberX]->v[(ny+steps*2)*(nx+steps*2)*(steps+cnt2) + (nx+steps*2)*(steps+cnt3) + steps + nx + cnt4];
            }
          }
        }
      }
    }

    c_e = clock();
//    c_e_omp = omp_get_wtime();
    c_e_mpi = MPI_Wtime();
    c_time += (double) (c_e - c_s) / (double) CLOCKS_PER_SEC;
//    c_time_omp += c_e_omp - c_s_omp;
    c_time_mpi += c_e_mpi - c_s_mpi;

  }

  e = clock();
//  e_omp = omp_get_wtime();
  e_mpi = MPI_Wtime();
  time = (double) (e - s) / (double) CLOCKS_PER_SEC;
//  time_omp = e_omp - s_omp;
  time_mpi = e_mpi - s_mpi;
/*  printf("rank:%d alle würfel time: %f  time_omp: %f  time_mpi: %f\n", rank, time, time_omp, time_mpi);
  printf("\t Berechnung  alle würfel time: %f  time_omp: %f  time_mpi: %f\n", a_time, a_time_omp, a_time_mpi);
  printf("\t Austausch l alle würfel time: %f  time_omp: %f  time_mpi: %f\n", b_time, b_time_omp, b_time_mpi);
  printf("\t Austausch g alle würfel time: %f  time_omp: %f  time_mpi: %f\n", c_time, c_time_omp, c_time_mpi);
*/
  printf("rank:%d time: %f  time_mpi: %f\n", rank, time, time_mpi);
  printf("\t calculation time: %f  time_mpi: %f\n", a_time, a_time_mpi);
  printf("\t exchange local time: %f  time_mpi: %f\n", b_time, b_time_mpi);
  printf("\t exchange mpi time: %f  time_mpi: %f\n", c_time, c_time_mpi);
  
  del_wavestate3d_cuboid(wvVecCuboid, cuboidNumberX, cuboidNumberY, cuboidNumberZ, first_mpi, last_mpi);
  MPI_Finalize();
  return 0;
}
