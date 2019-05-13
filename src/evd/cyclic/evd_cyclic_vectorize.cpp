#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "evd_cyclic.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "util.hpp"

void evd_cyclic_vectorize(struct matrix_t Data_matr, struct matrix_t Data_matr_copy, struct matrix_t Eigen_vectors,
                          struct vector_t Eigen_values, int epoch) {
    assert(Data_matr.rows == Data_matr.cols);
    double* A = Data_matr_copy.ptr;
    // Create a copy of the matrix to prevent modification of the original matrix
    memcpy(A, Data_matr.ptr, Data_matr.rows * Data_matr.cols * sizeof(double));

    double* V = Eigen_vectors.ptr;
    double* E = Eigen_values.ptr;
    const size_t m = Data_matr.rows;

    matrix_identity(Eigen_vectors);

    int is_not_diagonal = 0;

    for (int ep = 1; ep <= epoch; ep++) {
        double alpha, beta, cos_t, sin_t;

        for (size_t i = 0; i < m; i++) {
            for (size_t j = i + 1; j < m; j++) {
                if (A[i * m + j] != 0.0) {
                    is_not_diagonal = 1;
                    break;
                }
            }
        }

        if (!is_not_diagonal) break;

        for (size_t row = 0; row < m; row++) {
            for (size_t col = row + 1; col < m; col++) {
                size_t n = m;
                __m256d sin_vec, cos_vec;

                // Compute cos_t and sin_t for the rotation
                alpha = 2.0 * sign(A[row * m + row] - A[col * m + col]) * A[row * m + col];
                beta = fabs(A[row * m + row] - A[col * m + col]);
                cos_t = sqrt(0.5 * (1 + beta / sqrt(alpha * alpha + beta * beta)));
                // sin_t = (1 / 2*cos_t) * (alpha / sqrt(alpha*alpha + beta*beta));
                sin_t = sign(alpha) * sqrt(1 - cos_t * cos_t);

                sin_vec = _mm256_set1_pd(sin_t);
                cos_vec = _mm256_set1_pd(cos_t);

                if (m % 4 != 0) n = m - (m % 4);

                for (size_t i = 0; i < n; i += 4) {
                    __m256d Ac_row, Ac_col, Ac_rcopy;
                    __m256d sin_row, sin_col, cos_row, cos_col;

                    // Compute the eigen values by updating the columns until convergence
                    Ac_row = _mm256_set_pd(A[m * i + row], A[m * i + m + row], A[m * i + m * 2 + row],
                                           A[m * i + m * 3 + row]);
                    Ac_rcopy = Ac_row;
                    Ac_col = _mm256_set_pd(A[m * i + col], A[m * i + m + col], A[m * i + m * 2 + col],
                                           A[m * i + m * 3 + col]);

                    cos_row = _mm256_mul_pd(Ac_row, cos_vec);
                    sin_col = _mm256_mul_pd(Ac_col, sin_vec);
                    Ac_row = _mm256_add_pd(cos_row, sin_col);
                    double* Ac_row_updated = (double*) &Ac_row;
                    A[m * i + row] = Ac_row_updated[3];
                    A[m * i + m + row] = Ac_row_updated[2];
                    A[m * i + m * 2 + row] = Ac_row_updated[1];
                    A[m * i + m * 3 + row] = Ac_row_updated[0];

                    cos_col = _mm256_mul_pd(Ac_col, cos_vec);
                    sin_row = _mm256_mul_pd(Ac_rcopy, sin_vec);
                    Ac_col = _mm256_sub_pd(cos_col, sin_row);
                    double* Ac_col_updated = (double*) &Ac_col;
                    A[m * i + col] = Ac_col_updated[3];
                    A[m * i + m + col] = Ac_col_updated[2];
                    A[m * i + m * 2 + col] = Ac_col_updated[1];
                    A[m * i + m * 3 + col] = Ac_col_updated[0];
                }

                if (m % 4 != 0) {
                    for (size_t i = 0; i < m - n; i++) {
                        double A_i_r = A[m * (n + i) + row];
                        A[m * (n + i) + row] = cos_t * A[m * (n + i) + row] + sin_t * A[m * (n + i) + col];
                        A[m * (n + i) + col] = cos_t * A[m * (n + i) + col] - sin_t * A_i_r;
                    }
                }

                for (size_t i = 0; i < n; i += 4) {
                    __m256d A_row, A_col, A_rcopy, V_row, V_col, V_rcopy;
                    __m256d sin_row, sin_col, cos_row, cos_col;

                    // Compute the eigen values by updating the rows until convergence
                    A_row = _mm256_loadu_pd(A + m * row + i);
                    A_rcopy = A_row;
                    A_col = _mm256_loadu_pd(A + m * col + i);

                    cos_row = _mm256_mul_pd(A_row, cos_vec);
                    sin_col = _mm256_mul_pd(A_col, sin_vec);
                    A_row = _mm256_add_pd(cos_row, sin_col);
                    _mm256_storeu_pd(A + m * row + i, A_row);

                    cos_col = _mm256_mul_pd(A_col, cos_vec);
                    sin_row = _mm256_mul_pd(A_rcopy, sin_vec);
                    A_col = _mm256_sub_pd(cos_col, sin_row);
                    _mm256_storeu_pd(A + m * col + i, A_col);

                    // Compute the eigen vectors similarly by updating the eigen vector matrix
                    V_row = _mm256_loadu_pd(V + m * row + i);
                    V_rcopy = V_row;
                    V_col = _mm256_loadu_pd(V + m * col + i);

                    cos_row = _mm256_mul_pd(V_row, cos_vec);
                    sin_col = _mm256_mul_pd(V_col, sin_vec);
                    V_row = _mm256_add_pd(cos_row, sin_col);
                    _mm256_storeu_pd(V + m * row + i, V_row);

                    cos_col = _mm256_mul_pd(V_col, cos_vec);
                    sin_row = _mm256_mul_pd(V_rcopy, sin_vec);
                    V_col = _mm256_sub_pd(cos_col, sin_row);
                    _mm256_storeu_pd(V + m * col + i, V_col);
                }

                if (m % 4 != 0) {
                    for (size_t i = 0; i < m - n; i++) {
                        double A_r_i = A[m * row + n + i];
                        A[m * row + n + i] = cos_t * A[m * row + n + i] + sin_t * A[m * col + n + i];
                        A[m * col + n + i] = cos_t * A[m * col + n + i] - sin_t * A_r_i;

                        double V_r_i = V[m * row + n + i];
                        V[m * row + n + i] = cos_t * V[m * row + n + i] + sin_t * V[m * col + n + i];
                        V[m * col + n + i] = cos_t * V[m * col + n + i] - sin_t * V_r_i;
                    }
                }
            }
        }
        is_not_diagonal = 0;
    }

    matrix_transpose({V, m, m}, {V, m, m});

    // Store the generated eigen values in the vector
    for (size_t i = 0; i < m; i++) {
        E[i] = A[i * m + i];
    }

    reorder_decomposition(Eigen_values, &Eigen_vectors, 1, greater);
}
