#include "evd.hpp"
#include <math.h>
#include <vector>
#include <list>
#include "gtest/gtest.h"
#include "types.hpp"
#include "tsc_x86.h"
#include <stdio.h>

using namespace std;

#define CYCLES_REQUIRED 1e7
#define REP 100

/*
* Computes and reports and returns the total number of cycles required for the operation
*/
void perf_test_evd_classic(matrix_t Data_matr, vector_t E_vals, matrix_t E_vecs)
{
    double cycles = 0.;
    size_t num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            evd_classic(Data_matr, E_vecs, E_vals, 100);
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

    } while (multiplier > 2);

    list<double> cyclesList;

    // Actual performance measurements repeated REP times.
    // We simply store all results and compute medians during post-processing.
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (size_t i = 0; i < num_runs; ++i) {
            evd_classic(Data_matr, E_vecs, E_vals, 100);
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;

        cyclesList.push_back(cycles);
    }

    cyclesList.sort();
    cycles = cyclesList.front();
    printf("The operation required %lf cycles on average\n", cycles);
}

//Include the tests and call the perf_test_evd_classic function at the end

TEST(evdperf, random_matrix) {

    size_t n = 4;
    std::vector<double> A = {7.0, 3.0, 2.0, 1.0,
                             3.0, 9.0, -2.0, 4.0,
                             2.0, -2.0, -4.0, 2.0,
                             1.0, 4.0, 2.0, 3.0};
    std::vector<double> e(n);
    std::vector<double> V(n * n, 0);

    matrix_t Data_matr = {&A[0], n, n};
    vector_t E_vals = {&e[0], n};
    matrix_t E_vecs = {&V[0], n, n};
    perf_test_evd_classic(Data_matr, E_vals, E_vecs);
}
