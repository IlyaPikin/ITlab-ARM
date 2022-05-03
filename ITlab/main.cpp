#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <cstdint>
#include <cstdbool>
#include <cmath>
#include <arm_neon.h>

using namespace std;

//typedef float float32_t;
//typedef unsigned int uint32_t;

void matrix_multiply_c(float32_t* A, float32_t* B, float32_t* C, size_t n, size_t m, size_t k) {
    for (size_t i_idx = 0; i_idx < n; i_idx++) {
        for (size_t j_idx = 0; j_idx < m; j_idx++) {
            C[n * j_idx + i_idx] = 0;
            for (size_t k_idx = 0; k_idx < k; k_idx++) {
                C[n * j_idx + i_idx] += A[n * k_idx + i_idx] * B[k * j_idx + k_idx];
            }
        }
    }
}
void matrix_multiply_neon(float32_t* A, float32_t* B, float32_t* C, size_t n, size_t m, size_t k) {
    /*
     * Multiply matrices A and B, store the result in C.
     * It is the user's responsibility to make sure the matrices are compatible.
     */

    size_t A_idx;
    size_t B_idx;
    size_t C_idx;

    // these are the columns of a 4x4 sub matrix of A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    // these are the columns of a 4x4 sub matrix of B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    // these are the columns of a 4x4 sub matrix of C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    for (size_t i_idx = 0; i_idx < n; i_idx += 4) {
        for (size_t j_idx = 0; j_idx < m; j_idx += 4) {
            // Zero accumulators before matrix op
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            for (size_t k_idx = 0; k_idx < k; k_idx += 4) {
                // Compute base index to 4x4 block
                A_idx = i_idx + n * k_idx;
                B_idx = k * j_idx + k_idx;

                // Load most current A values in row 
                A0 = vld1q_f32(A + A_idx);
                A1 = vld1q_f32(A + A_idx + n);
                A2 = vld1q_f32(A + A_idx + 2 * n);
                A3 = vld1q_f32(A + A_idx + 3 * n);

                // Multiply accumulate in 4x1 blocks, i.e. each column in C
                B0 = vld1q_f32(B + B_idx);
                C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
                C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
                C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
                C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

                B1 = vld1q_f32(B + B_idx + k);
                C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
                C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
                C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
                C1 = vfmaq_laneq_f32(C1, A3, B1, 3);

                B2 = vld1q_f32(B + B_idx + 2 * k);
                C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
                C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
                C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
                C2 = vfmaq_laneq_f32(C2, A3, B2, 3);

                B3 = vld1q_f32(B + B_idx + 3 * k);
                C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
                C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
                C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
                C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
            }
            // Compute base index for stores
            C_idx = n * j_idx + i_idx;
            vst1q_f32(C + C_idx, C0);
            vst1q_f32(C + C_idx + n, C1);
            vst1q_f32(C + C_idx + 2 * n, C2);
            vst1q_f32(C + C_idx + 3 * n, C3);
        }
    }
}

bool f32comp_noteq(float32_t a, float32_t b) {
    if (fabs(a - b) < 0.000001) {
        return false;
    }
    return true;
}

bool matrix_comp(float32_t* A, float32_t* B, size_t rows, size_t cols) {
    float32_t a;
    float32_t b;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            a = A[rows * j + i];
            b = B[rows * j + i];

            if (f32comp_noteq(a, b)) {
                printf("i=%d, j=%d, A=%f, B=%f\n", i, j, a, b);
                return false;
            }
        }
    }
    return true;
}


int main(int argc, char* argv[])
{
    // Initialization
    size_t n, k, m, n4, k4, m4;
    float32_t l, h;

    //cout << "Enter the dimensions of the matrices [n, k, m]:" << endl;
    //cin >> n >> k >> m;
    //cout << "Enter a range of values [low, high]:" << endl;
    //cin >> l >> h;
    //cout << endl;
    cout << "argc = " << argc << endl;
    if (argc == 4)
    {
        n = strtoul(argv[1], NULL, 10);
        k = strtoul(argv[2], NULL, 10);
        m = strtoul(argv[3], NULL, 10);
        l = 0.0;
        h = 100.0;
    }
    if (argc == 6)
    {
        n = strtoul(argv[1], NULL, 10);
        k = strtoul(argv[2], NULL, 10);
        m = strtoul(argv[3], NULL, 10);
        l = atof(argv[4]);
        h = atof(argv[5]);
    }
    else { cout << "Wrong number of parameters." << endl; return 0; }

    // ƒополнение размеров матриц до размеров, кратных 4
    n4 = n + (4 - n % 4) % 4;
    k4 = k + (4 - k % 4) % 4;
    m4 = m + (4 - m % 4) % 4;

    float32_t* A = new float32_t[n4*k4];
    float32_t* B = new float32_t[k4*m4];
    float32_t* C = new float32_t[n4*m4];
    float32_t* D = new float32_t[n4*m4];

    for (size_t j = 0; j < k4; j++)
    {
        for (size_t i = 0; i < n4; i++)
        {
            if (i < n && j < k)
                A[n4 * j + i] = l + static_cast <float32_t> (rand()) / (static_cast <float32_t> (RAND_MAX / (h - l)));
            else
                A[n4 * j + i] = 0;
        
        }
    }
    for (size_t j = 0; j < m4; j++)
    {
        for (size_t i = 0; i < k4; i++)
        {
            if (i < k && j < m)
                B[k4 * j + i] = l + static_cast <float32_t> (rand()) / (static_cast <float32_t> (RAND_MAX / (h - l)));
            else
                B[k4 * j + i] = 0;

        }
    }

    // Computing and time clocking
    
    // C time
    double start_c; // time in seconds
    double end_c;
    start_c = omp_get_wtime();
    //... work to be timed ...
    matrix_multiply_c(A, B, C, n4, m4, k4);
    end_c = omp_get_wtime();
    printf("Time of C-code multiplication:  %f seconds\n", end_c - start_c);

    // Neon time
    double start_neon; // time in seconds
    double end_neon;
    start_neon = omp_get_wtime();
    //... work to be timed ...
    //matrix_multiply_neon(A, B, D, n4, m4, k4);
    end_neon = omp_get_wtime();
    printf("Time of multiplication with NEON: %f seconds\n", end_neon - start_neon);


    //// Print values
    if (n < 5 && k < 5 && m < 5)
    {
        cout << endl << "Matrix A" << endl;
        for (uint32_t i = 0; i < n4; i++)
        {
            for (uint32_t j = 0; j < k4; j++)
            {
                cout << A[n4 * j + i] << " ";
            }
            cout << endl;
        }
        cout << endl << "Matrix B" << endl;
        for (uint32_t i = 0; i < k4; i++)
        {
            for (uint32_t j = 0; j < m4; j++)
            {
                cout << B[k4 * j + i] << " ";
            }
            cout << endl;
        }
        cout << endl << "Matrix C" << endl;
        for (uint32_t i = 0; i < n4; i++)
        {
            for (uint32_t j = 0; j < m4; j++)
            {
                cout << C[n4 * j + i] << " ";
            }
            cout << endl;
        }
        cout << endl << "Matrix D" << endl;
        for (uint32_t i = 0; i < n4; i++)
        {
            for (uint32_t j = 0; j < m4; j++)
            {
                cout << D[n4 * j + i] << " ";
            }
            cout << endl;
        }
    }
    cout << endl;

    // Comparison
    if(matrix_comp(C, D, n4, m4))
        cout << "Correct calculation." << endl;
    else cout << "Incorrect calculation!" << endl;

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    return 0;
}
