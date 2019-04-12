#pragma once
/**
 * This is a modified version of the original measurement code published with the below license.
 */

/**
 *      _________   _____________________  ____  ______
 *     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
 *    / /_  / /| | \__ \ / / / /   / / / / / / / __/
 *   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
 *  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
 *
 *  http://www.acl.inf.ethz.ch/teaching/fastcode
 *  How to Write Fast Numerical Code 263-2300 - ETH Zurich
 *  Copyright (C) 2019
 *                   Tyler Smith        (smitht@inf.ethz.ch)
 *                   Alen Stojanov      (astojanov@inf.ethz.ch)
 *                   Gagandeep Singh    (gsingh@inf.ethz.ch)
 *                   Markus Pueschel    (pueschel@inf.ethz.ch)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see http://www.gnu.org/licenses/.
 */

#include <string>
#include <utility>
#include <vector>
#include "tsc_x86.h"

/**
 * A template variadic runtime measurement function which runs the given function with the given arguments a number
 * of times, and returns all the measured cycle values. Before performing the actual benchmarking, this function runs
 * the given function a couple of times to adjust the measured cycles.
 *
 * @tparam n_reps Number of times the function will run. This is also the size of the returned vector.
 * @tparam cycles_required The code will be executed for at least this number of cycles. This reduces timing overhead
 * when measuring small runtimes.
 * @param fn Any function to benchmark.
 * @param Args All the arguments to be passed when calling fn.
 * @return std::vector containing n_reps many measured cycle values.
 */
template <size_t n_reps = 100, size_t cycles_required = static_cast<size_t>(1e7), typename Func, typename... Args>
std::vector<double> measure_cycles(Func fn, Args... args) {
    double cycles = 0.;
    size_t num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least cycles_required cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            fn(std::forward<Args>(args)...);
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (cycles_required) / (cycles);
    } while (multiplier > 2);

    std::vector<double> cycles_vec;
    // Actual performance measurements repeated n_reps times.
    // We simply store all results and compute medians during post-processing.
    for (size_t j = 0; j < n_reps; j++) {
        start = start_tsc();
        for (size_t i = 0; i < num_runs; ++i) {
            fn(std::forward<Args>(args)...);
        }
        end = stop_tsc(start);
        cycles = ((double)end) / num_runs;
        cycles_vec.push_back(cycles);
    }

    return cycles_vec;
}

/**
 * Benchmark all the given functions using the same set of parameters.
 */
template <typename FuncType, typename... Args>
void run_all(const std::vector<FuncType>& fn_vec, const std::vector<std::string>& fn_names, Args... args) {
    assert(fn_vec.size() == fn_names.size());
    for (size_t i = 0; i < fn_vec.size(); ++i) {
        std::vector<double> cycles = measure_cycles(fn_vec[i], std::forward<Args>(args)...);
        auto median_it = cycles.begin() + cycles.size() / 2;
        std::nth_element(cycles.begin(), median_it, cycles.end());
        std::cout << fn_names[i] << " requires " << *median_it << " cycles (median)" << std::endl;
    }
}
