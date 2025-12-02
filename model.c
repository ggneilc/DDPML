/**
 * Architecture  : Regression
 * Loss function : Cross-entropy
 * Optimization  : Stochastic GD
 * 
 * N = number of features   
 * K = number of classes    
 * D = number of datapoints 
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
//#include <mpi.h>
#include "matrix.h"
#include "dataloader.h"

#define N 12
#define K 10
#define LR 0.005
#define EP 1e-7
#define EPOCHS 100000
struct timespec train_s, train_e, start, end;

matrix log_regressor();
matrix forward_pass_linear(const matrix*, const matrix*);
matrix forward_pass_softmax(const matrix*);
double loss_fn(const matrix*, const matrix*);
matrix optimize_gd(const matrix*, const matrix*, const matrix*, matrix*);
void train(const matrix* X, const matrix* Y, matrix* W, int epochs, int, int);
void test(const matrix*, const vector*, const matrix*);


long ram_usage(){
    // Example rough C structure:
    FILE* status_file = fopen("/proc/self/status", "r");
    char line[256];
    long rss_kb = 0;

    while (fgets(line, sizeof(line), status_file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            // Parse the numerical value here
            sscanf(line, "VmRSS: %ld kB", &rss_kb);
            break;
        }
    }
    fclose(status_file);
    return rss_kb;
}


int main(int argc, char const *argv[]){

    int comm_sz  = 1;
    int my_rank = 0;
//    MPI_Init (NULL ,NULL);
 //   MPI_Comm_size ( MPI_COMM_WORLD , & comm_sz );
  //  MPI_Comm_rank ( MPI_COMM_WORLD , & my_rank );
    char* filepath = "wine-data.csv";
    dataloader* data = NULL;
    matrix one_hot_y;

    printf("ram usage: %ld\n", ram_usage());


    if (my_rank == 0){
        data = load_data(filepath);

        // augment vector of labels to one-hot encoded matrix
        one_hot_y = new_matrix(data->samples, 10); // D x K
        for (int i = 1; i < data->samples; i++){
            int class = vget(data->y, i);
            mget(one_hot_y, i, class) = 1;
        }
    }

    // every process stores weights
    matrix weights = log_regressor();

    if (my_rank == 0){
        printf("W : %d x %d\n", weights.rows, weights.cols);
        printf("X : %d x %d\n", data->X.rows, data->X.cols);
    }
    clock_gettime(_POSIX_MONOTONIC_CLOCK, &train_s);
    train(&(data->X), &one_hot_y, &weights, EPOCHS, my_rank, comm_sz);
    clock_gettime(_POSIX_MONOTONIC_CLOCK, &train_e);
    double elapsed_sec = (train_e.tv_sec - train_s.tv_sec) + 
                    (train_e.tv_nsec - train_s.tv_nsec) / 1.0e9;
    printf("training time: %f\n", elapsed_sec);

    test(&(data->X), &(data->y), &weights);
    return 0;
}




/* ==== Logistic Regressor ==== */

/**
 * Weights for each feature = 11 (+1) = 12
 * with 0-10 quality -> 10 classes 
 * @return W = R^{10 x 12}
 */
matrix log_regressor(){
    srand(1234);
    matrix W = new_matrix(K, N); 
    // initialize weights between 0-1
    for (int i = 1; i <= K; i++ ){
        for (int j = 1; j <= N; j++){
            mget(W, i, j) = (double)rand() / RAND_MAX;
        }
    }
    return W;
}


/**
 * Row x Col 
 * Forward Pass
 * Linear  : computes linear score z_i 
 * @param  W : K x N  10 x 12
 * @param  X : D x N  D  x 12 
 *
 * computes XW^T : (D x N) (N x K)
 * 
 * @return Z : D x K
 */
matrix forward_pass_linear(const matrix* W, const matrix* X){
    matrix wt = transpose_matrix(W);
    return matrix_mult(X, &wt);
}

/**
 * Softmax : softmax on Z (Stabilized)
 * @param Z  : D x K  
 * @return y : D x K (normalized probabilities)
 */
matrix forward_pass_softmax(const matrix* Z){
    int D = Z->rows;
    matrix P = new_matrix(D, K);

    for (int i = 1; i <= D; i++){
        
        // --- 1. Find Max Score for numerical stability ---
        double max_z = mgetp(Z, i, 1);
        for (int j = 2; j <= K; j++){
            if (mgetp(Z, i, j) > max_z) {
                max_z = mgetp(Z, i, j);
            }
        }

        // --- 2. Calculate sum_exp using stabilized scores ---
        double sum_exp = 0.0;
        for (int j = 1; j <= K; j++){
            // Subtract max_z before exponentiation
            sum_exp += exp(mgetp(Z, i, j) - max_z);
        }

        // --- 3. Normalize each value in column (Stabilized) ---
        for (int j = 1; j <= K; j++){
            // Use stabilized exp(z - z_max) in the numerator
            mget(P, i, j) = exp(mgetp(Z, i, j) - max_z) / sum_exp;
        }
    }
    return P;
}

/**
 * Cross-entropy Loss (negative loglikelihood)
 * @param P : predictions   D x K
 * @param Y : 1-hot y       D x K
 * 
 * E(W) = - (1/N) * sum_n( sum_k( Y_nk * log(P_nk) ) )
 * 
 * @return err 
 */
double loss_fn(const matrix* P, const matrix* Y){
    double err = 0.0;
    int D = P->rows;
    for (int i = 1; i <= D; i++){
        for (int j = 1; j <= K; j++){
            if (mgetp(Y, i, j) == 1.0){
                double p_nk = mgetp(P, i, j);
                // clamp to 1.0 or EP
                if (p_nk < EP){ p_nk = EP;}
                if (p_nk > (1.0 - EP)){p_nk = (1.0 - EP);}
                err += log(p_nk);
            }
        }
    }
    return -(err/D);
}


/**
 * Gradient Descent:
 * @param X : D x N 
 * @param Y : D x K
 * @param P : D x K
 * @param W : K x N  
 * @param lr : learning rate
 * 
 * G = 1/N (X^T) (P - Y)
 * W = W - lr(G)
 * 
 * @return updated weight matrix W
 */
matrix optimize_gd(
    const matrix* X, const matrix* Y,
    const matrix* P, matrix* W_old
    ){
    /* == calculate gradient == */ 
    // 4092 x 10 
    matrix err = matrix_sub(P, Y);
    // 12 x 4092
    matrix xt = transpose_matrix(X);
    // (12 x 4092) * (4092 x 10) = (12 x 10)
    matrix grad = matrix_mult(&xt, &err);
    double scale = 1.0 / X->rows;
    matrix scaled_grad = matrix_scaler(&grad, scale);
    matrix orient_grad = transpose_matrix(&scaled_grad);
    matrix update_grad = matrix_scaler(&orient_grad, LR);
    /* == update step == */
    // W : 10 x 12 
    matrix W = matrix_sub(W_old, &update_grad);
    return W;
}

/**
 * Training Loop
 * for epoch: 
    * 1. forward passes
    * 2. calculate loss
    * 3. optimize
 */
void train(const matrix* X, const matrix* Y, matrix* W,
    int epochs, int my_rank, int comm_sz){
    double loss = 0.0;
    for (int i = 0; i < epochs; i ++){
        matrix Z = forward_pass_linear(W, X);
        matrix P = forward_pass_softmax(&Z);
        loss = loss_fn(&P, Y);
        matrix new_weight = optimize_gd(X, Y, &P, W);
        *W = new_weight;
        free_matrix(&Z);
        free_matrix(&P);
    }
}

/**
 * Tests the accuracy of the model
 * @param X : D x N
 * @param Y : 1 x D
 * @param W : K x N
 * 
 * computes yhat - y : number of misclassified examples
 * yhat = argmax(P_i) : the highest probability class
 * 
 */
void test(const matrix* X, const vector* Y, const matrix* W){
    matrix Z = forward_pass_linear(W, X);
    matrix P = forward_pass_softmax(&Z);
    vector preds = new_vector(Y->size);
    // argmax each P
    int total_errors = 0;
    for (int i = 1; i <= P.rows; i++){
        double class = -1; double max = -1.0;
        for (int j = 1; j < P.cols; j++){
            if (mget(P, i, j) > max) {
                max = mget(P, i, j) ;
                class = j;
            } 
        }
        vget(preds, i) = class;

        double true_pred = vgetp(Y, i);

        if (class != true_pred){
            total_errors++;
        }
    }

    double accuracy = 1.0 - ((double)total_errors / Y->size);
    printf("Test Accuracy: %.4f\n", accuracy);
    printf("Errors: %d\n", total_errors);
}