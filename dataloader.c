#include "dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>  // for feature scaling


/**
 * Given the filepath, assumed to be a csv with header row.
 * Returns Matrix of [1, x1, x2, ..., xn, y]
 * Assumes target is the final column
 */
dataloader* load_data(char* filepath){
    dataloader* loader = (dataloader*)malloc(sizeof(dataloader));
    FILE* file_ptr = fopen(filepath, "r");
    char meta[1024] = "";

    /** -------------------------
     * Count (rows, cols) for X, y
     * extract labels into meta[] 
     */
    char line[1024];
    int rows = 0;
    int cols = 0;
    while(fgets(line, sizeof(line), file_ptr)){
        rows += 1;
        if (rows == 1){ // headers -> set meta
            char *token = strtok(line, ";");
            while (token){
                cols += 1;
                strcat(meta, token);
                token = strtok(NULL, ";");
            }
        }
    }
    /** ------------------------- */


    rewind(file_ptr);
    // call once to skip header
    fgets(line, sizeof(line), file_ptr);


    /** ------------------------- *
     * Create Dataloader object
     */
    int samples = rows-1;
    int biased_features = cols;
    loader->X = new_matrix(samples, biased_features);
    loader->y = new_vector(samples);
    loader->features = biased_features;
    loader->samples = samples;
    loader->labels = meta;
    /** ------------------------- */



    /** ------------------------- *
     * Turn row[cols-1] to X matrix,
     * row[cols] to y vector
     */
    int row = 0;
    while(fgets(line, sizeof(line), file_ptr)){
        row += 1;
        // set bias
        mget(loader->X, row, 1) = 1.0; // Sets X[i, 1] = 1.0
        char *token = strtok(line, ";");  
        for (int col = 2; col <= cols; col++){
            mget(loader->X, row, col) = atof(token);
            token = strtok(NULL, ";");
        }
        // the next token is the final column
        vget(loader->y, row) = atoi(token);
    }
    /** ------------------------- */


    /**
     * --------------------------
     * Feature Normalization
     * --------------------------
     */

    int D = loader->samples; // Number of data points
    int N_feat = loader->features; // Total number of columns (12: Bias + 11 features)

    // 1. Loop through each FEATURE column (start at column 2, skip bias at 1)
    for (int j = 2; j <= N_feat; j++){
        
        // --- Calculate Mean (mu) ---
        double mu = 0.0;
        for (int i = 1; i <= D; i++){
            mu += mget(loader->X, i, j);
        }
        mu /= D;

        // --- Calculate Standard Deviation (sigma) ---
        double sigma = 0.0;
        for (int i = 1; i <= D; i++){
            double diff = mget(loader->X, i, j) - mu;
            sigma += diff * diff;
        }
        sigma = sqrt(sigma / D);
        
        // Handle case where sigma is zero (constant feature) to prevent division by zero
        if (sigma == 0.0) {
            sigma = 1.0; // By convention, if sigma is 0, leave values as 0 (mu - mu) / 1.0
        }

        // --- Apply Z-Score Transformation ---
        for (int i = 1; i <= D; i++){
            double normalized_value = (mget(loader->X, i, j) - mu) / sigma;
            mget(loader->X, i, j) = normalized_value;
        }
    }

    /** ------------------------- */

    return loader;
}