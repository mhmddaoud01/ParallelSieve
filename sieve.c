#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void sieve(int n, int rank, int size) {
    int local_start = 2 + rank * (n / size);
    int local_end = (rank == size - 1) ? n : local_start + (n / size) - 1;
    if (local_start < 2) local_start = 2;

    int range_size = local_end - local_start + 1;
    int *is_prime = malloc(range_size * sizeof(int));
    for (int i = 0; i < range_size; i++) is_prime[i] = 1;

    for (int i = 2; i <= sqrt(n); i++) {
        int first_multiple = (local_start / i) * i;
        if (first_multiple < local_start) first_multiple += i;
        if (first_multiple == i) first_multiple += i;

        for (int j = first_multiple; j <= local_end; j += i) {
            is_prime[j - local_start] = 0;
        }
    }

    int local_prime_count = 0;
    for (int i = 0; i < range_size; i++) {
        if (is_prime[i]) local_prime_count++;
    }

    int *counts = NULL, *displacements = NULL;
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displacements = malloc(size * sizeof(int));
    }
    MPI_Gather(&local_prime_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_primes = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) total_primes += counts[i];
        int offset = 0;
        for (int i = 0; i < size; i++) {
            displacements[i] = offset;
            offset += counts[i];
        }
    }

    int *local_primes = malloc(local_prime_count * sizeof(int));
    int idx = 0;
    for (int i = 0; i < range_size; i++) {
        if (is_prime[i]) local_primes[idx++] = local_start + i;
    }

    int *global_primes = NULL;
    if (rank == 0) global_primes = malloc(total_primes * sizeof(int));

    MPI_Gatherv(local_primes, local_prime_count, MPI_INT, global_primes, counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total primes up to %d: %d\nPrimes:\n", n, total_primes);
        for (int i = 0; i < total_primes; i++) printf("%d ", global_primes[i]);
        printf("\n");

        free(global_primes);
        free(counts);
        free(displacements);
    }

    free(is_prime);
    free(local_primes);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size, n;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the upper limit n: ");
        scanf("%d", &n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    sieve(n, rank, size);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Execution Time: %.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
