/*
 * Course: Distributed Systems / Parallel Programming
 * Assignment: MPI - Distributed Deep Learning (Question 2)
 *
 * Problem Statement:
 * 1. Generate a matrix X of 1000 elements on the root process.
 * 2. Scatter elements to all processes.
 * 3. Calculate local means on each process.
 * 4. Compute the global mean using the local means.
 * 5. Analyze performance differences with varying process counts.
 *
 * Mathematical Challenge:
 * It is known that "The arithmetic mean of a set of numbers does not equal
 * the mean of the arithmetic means of the subsets" unless subset sizes are equal.
 *
 * Solution Strategy:
 * 1. Padding: We pad the array with zeros so it divides evenly across P processes.
 * This ensures every process receives the exact same number of elements (send_count).
 * 2. Mean of Means: Since weights (counts) are now identical, averaging the local
 * means gives a mathematically valid "Padded Global Mean".
 * 3. Correction: We multiply the Padded Mean by a ratio (TotalPaddedSize / OriginalSize)
 * to eliminate the effect of the padding zeros and recover the true mean.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

 // The target size of the dataset as requested
#define N 1000
// Toggle this to 10000000 to demonstrate speedup in the report
// #define N 10000000 

int main(int argc, char** argv) {
    int rank, size;

    // Memory Pointers
    int* full_data = NULL;       // Stores the complete array (Rank 0 only)
    int* local_data = NULL;      // Stores the chunk received by this process
    double* gathered_means = NULL; // Stores local means from all processes (Rank 0 only)

    // variables for padding logic
    int remainder, padding = 0;
    int send_count;

    // Timing variables
    double start_time, end_time;

    // 1. MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2. Data Generation Phase (Rank 0 Only)
    if (rank == 0) {
        // Calculate padding to ensure equal division for Scatter
        remainder = N % size;
        if (remainder != 0) {
            padding = size - remainder;
        }

        // Allocate memory for N + padding
        full_data = (int*)malloc((N + padding) * sizeof(int));
        if (!full_data) {
            fprintf(stderr, "Memory allocation failed on Rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate buffer to receive means from all workers later
        gathered_means = (double*)malloc(size * sizeof(double));

        // Generate Random Data
        srand((unsigned int)time(NULL));
        printf("[Rank 0] Generating %d elements (Padding: %d)...\n", N, padding);

        for (int i = 0; i < N; i++) {
            full_data[i] = rand() % 100; // Random int 0-99
        }
        // Fill padding area with zeros (neutral for summation)
        for (int i = N; i < N + padding; i++) {
            full_data[i] = 0;
        }
    }

    // --- Performance Timer Start ---
    // Barrier ensures all processes start the clock simultaneously
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // 3. Broadcast Metadata
    // All processes need to know the 'padding' value to calculate 'send_count'
    MPI_Bcast(&padding, 1, MPI_INT, 0, MPI_COMM_WORLD);

    send_count = (N + padding) / size;
    local_data = (int*)malloc(send_count * sizeof(int));

    // 4. Scatter Data (Requirement 1)
    // Distributes the full_data array in equal chunks of 'send_count'
    MPI_Scatter(full_data, send_count, MPI_INT,
        local_data, send_count, MPI_INT,
        0, MPI_COMM_WORLD);

    // 5. Local Computation (Requirement 2)
    long long local_sum = 0;
    for (int i = 0; i < send_count; i++) {
        local_sum += local_data[i];
    }

    // Calculate local mean. 
    // Note: This satisfies the requirement to find the mean of received elements locally.
    double local_mean = (double)local_sum / send_count;

    // 6. Global Aggregation (Requirement 3)
    // We gather the partial means
    MPI_Gather(&local_mean, 1, MPI_DOUBLE,
        gathered_means, 1, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // --- Performance Timer Stop ---
    end_time = MPI_Wtime();

    // 7. Final Calculation and Reporting (Rank 0)
    if (rank == 0) {
        double sum_of_means = 0;

        printf("--------------------------------------------------\n");
        printf("[Rank 0] Collected Means from Workers:\n");
        for (int i = 0; i < size; i++) {
            sum_of_means += gathered_means[i];
            // Sample output to verify distribution
            if (i < 4) printf("   Proc %d Mean: %.2f\n", i, gathered_means[i]);
        }
        if (size > 4) printf("   ...\n");

        // Calculate "Padded Global Mean" (Simple average of averages)
        // This is valid because all local means have equal weight (send_count)
        double padded_global_mean = sum_of_means / size;

        // CORRECTION STEP:
        // Adjust for the padding zeros we added. 
        // Formula: TrueMean = PaddedMean * (TotalPaddedSize / OriginalSize)
        double total_padded_size = (double)(N + padding);
        double true_final_mean = padded_global_mean * (total_padded_size / N);

        printf("--------------------------------------------------\n");
        printf("Avg of Avgs (Padded): %.4f\n", padded_global_mean);
        printf("Correction Ratio:     %.4f\n", total_padded_size / N);
        printf("FINAL TRUE MEAN:      %.4f\n", true_final_mean);
        printf("Execution Time:       %f seconds\n", end_time - start_time);
        printf("--------------------------------------------------\n");

        // Verification (Sequential Calculation)
        // This proves that our Distributed Mean-of-Means logic is correct
        long long verify_sum = 0;
        for (int i = 0; i < N; i++) verify_sum += full_data[i];
        printf("Verification (Serial): %.4f\n", (double)verify_sum / N);

        // Cleanup Rank 0 memory
        free(full_data);
        free(gathered_means);
    }

    // Cleanup Worker memory
    free(local_data);

    MPI_Finalize();
    return 0;
}