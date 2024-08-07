/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

texture<float, cudaTextureType2D, cudaReadModeElementType> myTexture;


/**
 * This kernel will apply filter on any matrix parallely.
 * @param mat : given input matrix.
 * @param filter : given filter matrix.
 * @param result : output matrix on which result store.
 * @param m : number of rows in given matrix.
 * @param n : number of column in given matrix.
 * @param k : size of filter matrix.  
 * **/

__global__ void dkernel(long int* mat, long int* filter, long int* result,int m, int n, int k) {
    // filter matrix stored in shared memory 
    extern __shared__ long int filter_mat[];

    // id on which we perform filter parallely
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int x = globalId / n; // row number of input matrix
    int y = globalId % n; // column number of input matrix

    if(threadIdx.x == 0){
        // initiallizing shared memory with given filter 
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                filter_mat[i * k + j] = filter[i * k + j];
            }
        }
    }

    __syncthreads(); // because we want to make a barriar for filling shared memory.
    int center = k / 2; // offset on which we perform operation on matrix by assuming this element filter[center][center] will multiply by mat[id].
    int x1 = x - center;
    int y1 = y - center;

    long int total = 0; // store the sum for the given cell

    // applying filter on matrix
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            int i_cap = x1 + i; // row offset of resultant matrix
            int j_cap = y1 + j; // column offset of resultant matrix
            if (i_cap < m && i_cap >= 0 && j_cap < n && j_cap >= 0 ) {
                // if offsets are not overflowing the range than perform filter. 
                total += mat[i_cap * n + j_cap] * filter_mat[i * k + j];
            }
        }
    }
    // store filtered value in resultant matrix
    result[globalId] = total;
}

int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    
    
    long int* g_mat;
    long int* g_filter;
    long int* g_ans;

    cudaMalloc(&g_mat, m * n * sizeof(long int)); // allocating memory for input matrix on GPU
    cudaMemcpy(g_mat, h_mat, m * n * sizeof(long int), cudaMemcpyHostToDevice); 

    cudaMalloc(&g_filter, k * k * sizeof(long int));  // allocating memory for filter matrix on GPU
    cudaMemcpy(g_filter, h_filter, k * k * sizeof(long int), cudaMemcpyHostToDevice);

    cudaMalloc(&g_ans, m * n * sizeof(long int));  // allocating memory for output matrix on GPU


  
    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
    int memorySize = k*k*sizeof(long int); 
    dkernel<<<m, n, memorySize >>>(g_mat, g_filter, g_ans, m, n, k);
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    
    
    cudaDeviceSynchronize();

    cudaMemcpy(h_ans, g_ans, m * n * sizeof(long int), cudaMemcpyDeviceToHost);

 
    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}