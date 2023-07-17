#include "mpi.h"
#include <stdio.h>

#define MAX 10000

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int px = atoi(argv[1]);

    int py = atoi(argv[2]);

    long long int N = atoll(argv[3]);

    int sendbuf[N+1][N];

    if(rank == 0)
    {
        for(int i=0;i<N+1;i++){
            for(int j=0;j<N;j++){
                sendbuf[i][j] = rand()%MAX;
                // sendbuf[i][j] = 1;
            }
        }
    }else{
        for(int i=0;i<N+1;i++){
            for(int j=0;j<N;j++){
                sendbuf[i][j] = 0;
            }
        }
    }

    MPI_Datatype newvtype[px*py];
    MPI_Status status;

    for(int i=0;i<px;i++){
        for(int j=0;j<py;j++){
            int count = N/py + 1;
            int blocklengths[count];
            int displacements[count];
            for(int k=0;k<count;k++){
                blocklengths[k]=N/px;
                displacements[k]=(rank/px)*(N/py)*N + (rank%px)*(N/px) + k*N;
                // if(rank==1){
                //     printf("%d %d %d\n", blocklengths[k], displacements[k], k);
                // }
            }
            MPI_Type_indexed(count, blocklengths, displacements, MPI_INT, &newvtype[i+j*px]);
            MPI_Type_commit(&newvtype[i+j*px]);
        }
    }

    for (int i=0;i<10;i++){
        double start = MPI_Wtime();
        // scatter sendbuf to from 0 to 
        if(rank == 0){
            for(int j=1;j<px*py;j++){
                MPI_Send(sendbuf, 1, newvtype[j], j, 0, MPI_COMM_WORLD);
            }
        }else{
            MPI_Recv(sendbuf, 1, newvtype[rank], 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        int x = (rank/px)*(N/py), y = (rank%px)*(N/px);

        for(int j=x; j<x+(N/py); j++){
            for (int k=y; k<y+(N/px); k++){
                if(k>=j)
                sendbuf[j][k]=sendbuf[j][k]-sendbuf[j+1][k];
            }
        }
        
        // gather from all process into sendbuf of process 0
        if(rank == 0){
            for(int j=1;j<px*py;j++){
                MPI_Recv(sendbuf, 1, newvtype[j], j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }else{
            MPI_Send(sendbuf, 1, newvtype[rank], 0, 1, MPI_COMM_WORLD);
        }

        // if(rank == 0){
        //     for (int j=0;j<N;j++){
        //         for (int k=0;k<N;k++){
        //             printf("%d ", sendbuf[j][k]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        double end = MPI_Wtime();

        // if(rank == 0){
            printf("%d, %lld, %d, %f\n", rank, N, i, end-start);
        // }
    }

    for(int i=0;i<px;i++){
        for(int j=0;j<py;j++){
            MPI_Type_free(&newvtype[i+j*px]);
        }
    }

    MPI_Finalize();
    return 0;
}
