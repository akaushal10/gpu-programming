#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <chrono>
using namespace std;

__host__ __device__ bool isColinear(int* x,int* y, int p1, int p2, int p3) {
    if(x[p2]!=x[p3] || y[p2]!=y[p3]){
        return (y[p1] - y[p2])*(x[p3] - x[p2]) == (y[p3] - y[p2])*(x[p1] - x[p2]);
    }
    if (y[p2] == y[p3]) {
        return y[p1] == y[p2];
    }
    return x[p1] == x[p2];
}


__global__ void dkernal(int* T_gpu,int* gpu_currentRound,int* x,int* y,int* hp,int* tempHp,int* score,int* semaphore,int* result,int* distArr){
    int currentRound = *gpu_currentRound;
    int T = *T_gpu;
    int tank_id = blockIdx.x;
    int victim_tank = threadIdx.x;
    if(victim_tank==0){
        distArr[tank_id] = INT_MAX;
        semaphore[tank_id] = 0;
        result[tank_id] = -1;
    }
    __syncthreads();
    int target_id = (tank_id + currentRound) % T;
    int x1 = x[tank_id];
    int y1 = y[tank_id];
    int x2 = x[target_id];
    int y2 = y[target_id];
    int dir = 0;
    if (x1 >= x2)
    {
        if (y1 > y2)
            dir = 1;
        else
            dir = 2;
    }
    else
    {
        if (y1 > y2)
            dir = 3;
        else
            dir = 4;
    }

    int x3 = x[victim_tank];
    int y3 = y[victim_tank];

    int newDir = 0;

    if (x1 >= x3)
    {
        if (y1 > y3)
            newDir = 1;
        else
            newDir = 2;
    }
    else
    {
        if (y1 > y3)
            newDir = 3;
        else
            newDir = 4;
    }

    if(hp[tank_id]>0 && hp[victim_tank]>0 && tank_id!=victim_tank && dir==newDir && isColinear(x,y,victim_tank,tank_id,target_id)){
        int currentDist = abs(x1 - x3) + abs(y1 - y3);
        int oldSemaphoreValue = 0;
        if(currentDist<distArr[tank_id]){
            do{
                oldSemaphoreValue=atomicCAS(&semaphore[tank_id],0,1);
                if(!oldSemaphoreValue){
                    if(currentDist<distArr[tank_id]){ 
                        distArr[tank_id]=currentDist;
                        result[tank_id]=victim_tank;
                    }
                    semaphore[tank_id]=0;
                }
            }while(oldSemaphoreValue!=0);
        }
    }

    __syncthreads();
    if(victim_tank==tank_id){

        if(result[tank_id]!=-1){
            score[tank_id]+=1;
            atomicAdd(&tempHp[result[tank_id]],-1);
        }
    }
    
}


__global__ void hpCopy(int* remainingTank,int* hp,int *tempHp){
    int tank_id = threadIdx.x;
    if(tempHp[tank_id]){
        hp[tank_id]+=tempHp[tank_id];
        if(hp[tank_id]<1){
            atomicAdd(remainingTank,-1);
        }
    }
    tempHp[tank_id] = 0;
}
__global__ void initHP(int* hp,int H){
    int id = threadIdx.x;
    hp[id] = H;
}
int main(int argc, char **argv) {
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL) {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int));  // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int));  // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++) {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    int* gpu_x;
    cudaMalloc(&gpu_x,sizeof(int)*T);
    cudaMemcpy(gpu_x,xcoord,sizeof(int)*T,cudaMemcpyHostToDevice);

    int* gpu_y;
    cudaMalloc(&gpu_y,sizeof(int)*T);
    cudaMemcpy(gpu_y,ycoord,sizeof(int)*T,cudaMemcpyHostToDevice);

    int* gpu_hp;
    cudaMalloc(&gpu_hp,sizeof(int)*T);
    initHP<<<1,T>>>(gpu_hp,H);

    int* gpu_temp_hp;
    cudaMalloc(&gpu_temp_hp,sizeof(int)*T);

    int* gpu_score;
    cudaMalloc(&gpu_score,sizeof(int)*T);
    cudaMemcpy(gpu_score,score,sizeof(int)*T,cudaMemcpyHostToDevice);

    int* gpu_lock;
    cudaMalloc(&gpu_lock,sizeof(int)*T);

    int* gpu_result;
    cudaMalloc(&gpu_result,sizeof(int)*T);

    int* gpu_min_dist;
    cudaMalloc(&gpu_min_dist,sizeof(int)*T);


    int* tankLeft;
    cudaHostAlloc(&tankLeft,sizeof(int),0);
    *tankLeft = T;

    int* T_gpu;
    cudaHostAlloc(&T_gpu,sizeof(int),0);
    *T_gpu = T;

    int* currentRound;
    cudaHostAlloc(&currentRound,sizeof(int),0);
    *currentRound = 1;
    while (*tankLeft > 1) {  
        if (!(*currentRound % T)) {
            *currentRound += 1 ;
            continue;
        }
        dkernal<<<T,T>>>(T_gpu,currentRound,gpu_x,gpu_y,gpu_hp,gpu_temp_hp,gpu_score,gpu_lock,gpu_result,gpu_min_dist);
        hpCopy<<<1,T>>>(tankLeft,gpu_hp,gpu_temp_hp);
        cudaDeviceSynchronize();
        *currentRound = *currentRound + 1 ;
    }
    cudaMemcpy(score,gpu_score,sizeof(int)*T,cudaMemcpyDeviceToHost);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++) {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
