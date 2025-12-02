#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "matrix.h"

matrix new_matrix(const int rows, const int cols){
    matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    assert(rows>0); assert(cols>0);
    mat.val = (double*)malloc(sizeof(double)*rows*cols);

    for (int i=0; i < (rows*cols); i++){
        mat.val[i] = 0.0;
    }

    return mat;
}

void free_matrix(matrix* A){
    free(A->val);
}
void free_vector(vector* x){
    free(x->val);
}

void print_matrix_full(const matrix* mat, char* varname){
    assert(mat->rows>0); assert(mat->cols>0);
    printf("\n %.100s = \n", &varname[1]);
    for(int i=1; i<mat->rows; i++){
        printf("  |  ");
        for(int j=1; j<=mat->cols; j++){
            printf("%10.3e", mgetp(mat,i,j));
            if (j<mat->cols) {printf(", ");}
            else {printf(" ");}
        }
        printf("|\n");
    }
        printf("\n");
}


matrix matrix_add(const matrix* A, const matrix* B){
    const int rows = A->rows;
    const int cols = A->cols;
    assert(rows==B->rows); assert(cols==B->cols);
    matrix C = new_matrix(rows, cols);

    for(int i=1; i<=rows; i++){
        for (int j=1; j<=cols; j++){
            mget(C,i,j) = mgetp(A,i,j)+mgetp(B,i,j);
        }
    }
    return C;
}


/* -------------- MODEL FUNCTIONS --------------------*/

/**
 * TODO: parallelize with OpenMP
 */
matrix matrix_scaler(const matrix* A, const double s){
    int rows = A->rows;
    int cols = A->cols;
    matrix tmp = new_matrix(rows, cols);
    #pragma omp parallel for
    for (int i = 1; i <= rows; i++){
        for (int j = 1; j <= cols; j++){
            mget(tmp, i, j) = mgetp(A, i, j) * s;
        }
    }
    return tmp;
}


/**
 * TODO: Parallelize with OpenMP
 */ 
matrix matrix_sub(const matrix* A, const matrix* B){
    const int rows = A->rows;
    const int cols = A->cols;
    assert(rows==B->rows); assert(cols==B->cols);
    matrix C = new_matrix(rows, cols);

    #pragma omp parallel for
    for(int i=1; i<=rows; i++){
        for (int j=1; j<=cols; j++){
            mget(C,i,j) = mgetp(A,i,j)-mgetp(B,i,j);
        }
    }
    return C;
}


/**
 * TODO: Parallelize with OpenMP
 */ 
matrix matrix_mult(const matrix* A, const matrix* B){
    const int rowsA = A->rows; const int colsA = A->cols;
    const int rowsB = B->rows; const int colsB = B->cols;
    assert(colsA==rowsB); 
    matrix C = new_matrix(rowsA, colsB);

    #pragma omp parallel for collapse(2)
    for(int i=1; i<=rowsA; i++)
        for(int j=1; j<=colsB; j++)
            for(int k=1; k<=colsA; k++){
                mget(C,i,j) += mgetp(A,i,k)*mgetp(B,k,j);
            }
    return C;
}



/**
 * TODO: Parallelize with OpenMP
 */ 
matrix transpose_matrix(const matrix* A){
    matrix T = new_matrix(A->cols, A->rows);

    #pragma omp parallel for
    for (int i=1; i < A->rows+1; i++){
        for (int j=1; j < A->cols+1; j++){
            mget(T,j,i) = mgetp(A,i,j);
        }
    }
    return T;
}

/* -----------------------------------------------------*/

matrix matrix_dot_mult(const matrix* A, const matrix* B){
    const int rows = A->rows;
    const int cols = A->cols;
    assert(rows==B->rows); assert(cols==B->cols);
    matrix C = new_matrix(rows, cols);

    for(int i=1; i<=rows; i++){
        for (int j=1; j<=cols; j++){
            mget(C,i,j) = mgetp(A,i,j)*mgetp(B,i,j);
        }
    }
    return C;
}

matrix invert(const matrix* A){
    int n = A->rows;
    matrix inv = new_matrix(n, n);

    // temporary b 
    vector e = new_vector(n);

    for (int j = 1; j <= n; ++j) {
        // Set e to the j-th standard basis vector
        for (int i = 1; i <= n; ++i)
            vget(e, i) = (i == j) ? 1.0 : 0.0;

        // Solve A x = e_j
        vector x = solve(A, &e);

        // Store result as j-th column of inv
        for (int i = 1; i <= n; ++i)
            mget(inv, i, j) = vget(x, i);

        free(x.val);  // if dynamically allocated
    }

    free(e.val);
    return inv;
}


vector new_vector(const int size){
    vector vec;
    vec.size = size;
    assert(size>0);
    vec.val = (double*)malloc(sizeof(double)*size);

    for (int i=0; i < size; i++){
        vec.val[i] = 0.0;
    }
    return vec;
}

void print_vector_full(const vector* vec, char* varname){
    assert(vec->size>0);
    printf("\n");
    printf("\n %.100s = \n", &varname[1]);
    printf("  |  ");
    for(int i=1; i<vec->size+1; i++){
        printf("%10.3e", vgetp(vec,i));
        if (i<vec->size) {printf(", ");}
    }
    printf("  |^T\n\n");
}

/**
 * TODO: Parallelize
 */ 
vector vector_add(const vector* x, const vector* y){
    const int size = x->size;
    assert(size==y->size);
    vector z = new_vector(size);

    for(int i=1; i<=size; i++){
        vget(z,i) = vgetp(x,i)+vgetp(y,i);
    }
    return z;
}

vector vector_sub(const vector* x, const vector* y){
    const int size = x->size;
    assert(size==y->size);
    vector z = new_vector(size);

    for(int i=1; i<=size; i++){
        vget(z,i) = vgetp(x,i)-vgetp(y,i);
    }
    return z;
}

double vector_dot_mult(const vector* x, const vector* y){
    const int size = x->size;
    assert(size==y->size);
    double z = 0.0;

    for(int i=1; i<=size; i++){
        z += vgetp(x,i)*vgetp(y,i);
    }
    return z;
}

void print_scalar_full(const double* z, char* varname){
    printf("\n %.100s = %10.3e \n\n", &varname[1], *z);
}

vector matrix_vector_mult(const matrix* A, const vector* x){
    const int rows = A->rows; const int cols = A->cols;
    const int size = x->size;
    assert(cols==size);
    vector Ax = new_vector(rows);

    for (int i=1; i<=rows; i++){
        double tmp = 0.0;
        for (int j=1; j<=size; j++){
            tmp += mgetp(A,i,j) * vgetp(x,j);
        }
        vget(Ax,i) = tmp;
    }

    return Ax;
}

/**
 * Gaussian Elimination
 * with Partial Pivoting 
 * and Back substitution
 */ 
vector solve(const matrix* A, const vector* b){
    const int rows = A->rows;
    const int cols = A->cols;
    const int size = b->size;
    assert(rows==cols); assert(rows==size);

    vector x = new_vector(rows);

    matrix Acopy = new_matrix(A->rows, A->cols);
    vector Bcopy = new_vector(b->size);

    // deep copy A and B
    for(int i=1; i<A->rows+1; i++){
        for(int j=0; j<A->cols+1; j++){
            mget(Acopy, i, j) = mgetp(A, i, j);
        }
    }
    for (int i=1; i<b->size+1; i++){
        vget(Bcopy, i) = vgetp(b, i);
    }

    // Loop over each column
    for (int i=1; i<=(size-1); i++){
        //select largest pivor in current column
        int p=i; double maxA = -100.0e0;
        for(int j=i; j<=size; j++){
            double tmp = fabs(mget(Acopy,j,i));
            if (tmp > maxA) {p=j; maxA=tmp;}
        }

        // See if matrix is singular
        if (maxA <= 1.0e-14)
        {printf("Matrix is singular\n"); exit(1);}

        //Pivot (swap rows)
        if (p!=i) {
            for(int j=i; j<=size; j++){
                double tmp = mget(Acopy,i,j);
                mget(Acopy,i,j) = mget(Acopy,p,j);
                mget(Acopy,p,j) = tmp;
            }
            double tmp = vget(Bcopy,i);
            vget(Bcopy,i) = vget(Bcopy,p);
            vget(Bcopy,p) = tmp;
        }

        //Eliminate below diagonal
        for(int j=i+1; j<=size; j++){
            double m = mget(Acopy,j,i)/mget(Acopy,i,i);
            for(int k=i; k<=size; k++){
                mget(Acopy,j,k) = mget(Acopy,j,k) - m*mget(Acopy,i,k);
            }
            vget(Bcopy,j) = vget(Bcopy,j) - m*vget(Bcopy,i);
        }

        // Backward substitution
        vget(x,size) = vget(Bcopy,size)/mget(Acopy,size,size);
        for(int j=1; j<=(size-1); j++){
            int i = size-j;
            double sum = 0.0e0;
            for(int k=i+1; k<=size; k++){
                sum = sum + mget(Acopy,i,k)*vget(x,k);
            }
            vget(x,i) = (vget(Bcopy,i)-sum)/mget(Acopy,i,i);
        }
    }

    return x;
}

double power_iteration(const vector* v, const matrix* A, const double TOL, const int maxiter){
    const int size = v->size;
    assert(A->rows==A->cols); assert(A->rows==size);

    vector b_k = new_vector(size);
    for (int i=1; i<=size; i++){
        vget(b_k,i) = vgetp(v,i);
    }

    double lambda = 0.0;
    for (int iter=0; iter<maxiter; iter++){
        // calculate the matrix-by-vector product Ab
        vector Ab = matrix_vector_mult(A, &b_k);

        // calculate the norm
        double norm = 0.0;
        for (int i=1; i<=size; i++){
            norm += vget(Ab,i)*vget(Ab,i);
        }
        norm = sqrt(norm);

        // re normalize the vector
        for (int i=1; i<=size; i++){
            vget(b_k,i) = vget(Ab,i)/norm;
        }

        // Rayleigh quotient
        double lambda_new = 0.0;
        vector Ab_new = matrix_vector_mult(A, &b_k);
        for (int i=1; i<=size; i++){
            lambda_new += vget(b_k,i)*vget(Ab_new,i);
        }

        free(Ab.val);
        free(Ab_new.val);

        // check convergence
        if (fabs(lambda_new - lambda) < TOL){
            lambda = lambda_new;
            break;
        }
        lambda = lambda_new;
        printf("Iter %d: lambda = %f\n", iter, lambda);
    }

    for (int i=1; i<=size; i++){
        vgetp(v,i) = vget(b_k,i);
    }
    free(b_k.val);
    return lambda;
}

double inverse_iteration(const double mu, const vector* v, const matrix* A, const double TOL, const int maxiter){
    const int size = v->size;
    assert(A->rows==A->cols); assert(A->rows==size);

    vector b_k = new_vector(size);
    for (int i=1; i<=size; i++){
        vget(b_k,i) = vgetp(v,i);
    }

    double lambda = 0.0;
    for (int iter=0; iter<maxiter; iter++){
        // Solve (A - mu*I)x = b
        matrix A_shifted = new_matrix(size, size);
        for (int i=1; i<=size; i++){
            for (int j=1; j<=size; j++){
                if (i == j){
                    mget(A_shifted,i,j) = mgetp(A,i,j) - mu;
                } else {
                    mget(A_shifted,i,j) = mgetp(A,i,j);
                }
            }
        }

        vector x = solve(&A_shifted, &b_k);
        free(A_shifted.val);

        // calculate the norm
        double norm = 0.0;
        for (int i=1; i<=size; i++){
            norm += vget(x,i)*vget(x,i);
        }
        norm = sqrt(norm);

        // re normalize the vector
        for (int i=1; i<=size; i++){
            vget(b_k,i) = vget(x,i)/norm;
        }
        free(x.val);

        // Rayleigh quotient
        double lambda_new = 0.0;
        vector Ab_new = matrix_vector_mult(A, &b_k);
        for (int i=1; i<=size; i++){
            lambda_new += vget(b_k,i)*vget(Ab_new,i);
        }
        free(Ab_new.val);

        // check convergence
        if (fabs(lambda_new - lambda) < TOL){
            lambda = lambda_new;
            break;
        }
        lambda = lambda_new;
        printf("Iter %d: lambda = %f\n", iter, lambda);
    }

    for (int i=1; i<=size; i++){
        vgetp(v,i) = vget(b_k,i);
    }
    free(b_k.val);
    return lambda;
}