#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "util.hpp"
#include "types.hpp"
#include "evd.hpp"

void MatMul(double *, double *, double *, int);

void evd_classic(struct matrix_t Data_matr, struct vector_t Eigen_values, int epoch) {

	double *A = Data_matr.ptr;
	const int m = Data_matr.rows;
	
	double *E = Eigen_values.ptr;
	
	// Build the auxiliary matrices
	
	double *P, *P_t, *temp;
	P = (double *)malloc(sizeof(double) * m * m);
	P_t = (double *)malloc(sizeof(double) * m * m);
	temp = (double *)malloc(sizeof(double) * m * m);
	
	for(int ep = 1; ep <= epoch; ep++) {
		
		double val = 0.0;
		int i_max, j_max;
		double alpha, beta, cos_t, sin_t;
		
		identity(P, m);
		
		// Find the larget non-diagonal element in the
		// upper triangular matrix
		
		for(int i = 0; i < m; i++) {
			for (int j = i+1; j < m; j++) {
				if(fabs(A[i*m + j]) > val) {
					i_max = i;
					j_max = j;
					val = A[i*m + j];
				}
			}
		}
	
		// Compute cos_t and sin_t for the rotation matrix
	
		alpha = 2.0 * sign(A[i_max*m + i_max] - A[j_max*m + j_max]) * A[i_max*m + j_max];
		beta = fabs(A[i_max*m + i_max] - A[j_max*m + j_max]);
		cos_t = sqrt(0.5 * (1 + beta / sqrt(alpha*alpha + beta*beta)));
		// sin_t = (1 / 2*cos_t) * (alpha / sqrt(alpha*alpha + beta*beta));
		sin_t = sign(alpha) * sqrt(1 - cos_t * cos_t);

		// Initialize the rotation parameters in the identity matrix
		
		P[i_max*m + i_max] = P[j_max*m + j_max] = cos_t;
		P[j_max*m + i_max] = sin_t;
		P[i_max*m + j_max] = -1 * sin_t;
	
		// Perform the operation A(i) = P_t * A(i-1) * P
		// corresponding to Jacobi iteration i
		
		transpose(P, P_t, m);
		MatMul(temp, P_t, A, m);
		MatMul(A, temp, P, m);
	}
	
	// Store the generated eigen values in the vector
	for(int i = 0; i < m; i++) {
		E[i] = A[i*m + i];
	}

}

void MatMul(double *P, double *Q, double *R, int n) {
	double sum = 0.0;
	for(int i = 0; i < n ; i++) {
		for(int j = 0; j < n; j++) {
			for(int k = 0; k < n; k++) {
				sum += Q[i*n + k] * R[k*n + j];
			}
			P[i*n + j] = sum;
			sum = 0.0;
		}
	}
}
