#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 4000  /* Max value of N */
int N = 2000;  /* Matrix size */

/* Matrices and vectors */
float A[2000][2000], B[2000], X[2000];
float C[2000];
/* A * X = B, solve for X */
double sum=0;
int seed = 3;
/* junk */
#define randm() 4|2[uid]&3

/*functions definition*/
void initialize_inputs(){
  int row, col;
  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }
}
/* Print input matrices */
void print_inputs(){
    int row, col;
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
    printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
}
/*printing the results of the solution*/
void print_X(){
    int row;
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
}

int main(int argc, char **argv){
    int i,j,k;
    int arr[2000];
    int rank, nprocs;
    double start_time, end_time, elapsed_time, range=1; 
    MPI_Status status;
    int root_proc = 0;

    /*starting the MPI parallelism*/
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 

    if (rank==root_proc){
        /**initializing matrix and vector*/
        initialize_inputs();
        printf("\nMatrix and vector generated randomly\n");
        /*print inputs function*/
        print_inputs();
    }

    start_time = MPI_Wtime();

    /*gaussian elimination*/
    MPI_Bcast (&A[0][0],2000*2000,MPI_DOUBLE,root_proc,MPI_COMM_WORLD);
    MPI_Bcast (B,N,MPI_DOUBLE,root_proc,MPI_COMM_WORLD);    

    for(i=0; i<N; i++){
        arr[i]= i % nprocs;
    } 
    for(k=0;k<N;k++){
        MPI_Bcast (&A[k][k],N-k,MPI_DOUBLE,arr[k],MPI_COMM_WORLD);
        MPI_Bcast (&B[k],1,MPI_DOUBLE,arr[k],MPI_COMM_WORLD);
        for(i= k+1; i<N; i++) {
            if(arr[i] == rank){
                C[i]=A[i][k]/A[k][k];
            }
        }               
        for(i= k+1; i<N; i++) {       
            if(arr[i] == rank){
                for(j=0;j<N;j++){
                    A[i][j]=A[i][j]-( C[i]*A[k][j] );
                }
                B[i]=B[i]-( C[i]*B[k] );
            }
        }
    }
    /*stopping timer for gaussian elimination*/
    end_time = MPI_Wtime();

    /*back substitution from the root process*/
    if (rank==root_proc){ 
        X[N-1]=B[N-1]/A[N-1][N-1];
        for(i=N-2;i>=0;i--){
            sum=0;
            for(j=i+1;j<N;j++){
                sum=sum+A[i][j]*X[j];
            }
            X[i]=(B[i]-sum)/A[i][i];
        }
    /*printing matrix solution and elapsed time by root processor*/
        elapsed_time = end_time - start_time;
        printf("\nThe solution being X is:");
        print_X();
        printf("\n\nGaussian elimination time: %f seconds\n", elapsed_time);
    }

    /*finalizing MPI parallelism*/
    MPI_Finalize();
    return(0);
}