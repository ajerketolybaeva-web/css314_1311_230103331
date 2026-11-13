
import multiprocessing as mp
from multiprocessing import shared_memory
import array
import time
import statistics

from harness import emit_verify_tag

STUDENT_ID = 230103331
N = 384
P_PHYS = 8

TILE_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

WARMUP_RUNS = 1
TIMED_RUNS = 3


def gemm_worker(args):
    name_a, name_b, name_c, row_start, row_end, variant, block_size = args

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)
    shm_c = shared_memory.SharedMemory(name=name_c)

    A = shm_a.buf.cast("d")
    B = shm_b.buf.cast("d")
    C = shm_c.buf.cast("d")

    if variant == "naive":
        for i in range(row_start, row_end):
            for j in range(N):
                dot = 0.0
                for k in range(N):
                    dot += A[i * N + k] * B[k * N + j]
                C[i * N + j] = dot

    elif variant == "reordered":
        for i in range(row_start, row_end):
            for k in range(N):
                a_ik = A[i * N + k]
                for j in range(N):
                    C[i * N + j] += a_ik * B[k * N + j]

    elif variant == "blocked":
        bs = block_size

        for ii in range(row_start, row_end, bs):
            i_end = min(ii + bs, row_end)

            for kk in range(0, N, bs):
                k_end = min(kk + bs, N)

                for jj in range(0, N, bs):
                    j_end = min(jj + bs, N)

                    for i in range(ii, i_end):
                        for k in range(kk, k_end):
                            a_ik = A[i * N + k]
                            for j in range(jj, j_end):
                                C[i * N + j] += a_ik * B[k * N + j]

    del A, B, C

    shm_a.close()
    shm_b.close()
    shm_c.close()


def reset_c(shm_c):
    C = shm_c.buf.cast("d")
    for i in range(N * N):
        C[i] = 0.0
    del C


def run_variant(pool, tasks, shm_c):
    reset_c(shm_c)
    pool.map(gemm_worker, tasks)


def benchmark(pool, tasks, shm_c):
    for _ in range(WARMUP_RUNS):
        run_variant(pool, tasks, shm_c)

    times = []

    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        run_variant(pool, tasks, shm_c)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return statistics.median(times)


def create_tasks(shm_a, shm_b, shm_c, variant, block_size=None):
    tasks = []
    rows_per_worker = N // P_PHYS

    for worker in range(P_PHYS):
        start = worker * rows_per_worker
        end = (worker + 1) * rows_per_worker

        tasks.append((
            shm_a.name,
            shm_b.name,
            shm_c.name,
            start,
            end,
            variant,
            block_size
        ))

    return tasks


def verify_results(shm_c, reference):
    C = shm_c.buf.cast("d")

    max_error = 0.0

    for i in range(N * N):
        error = abs(C[i] - reference[i])
        if error > max_error:
            max_error = error

    del C
    return max_error


def main():
    print("=" * 70)
    print("PHASE 3B: MATRIX MULTIPLICATION OPTIMIZATION")
    print("=" * 70)
    print(f"Matrix size: {N} x {N}")
    print(f"Processes: {P_PHYS}")
    print(f"Tile sizes: {TILE_SIZES}")

    total_elements = N * N

    # Create deterministic input matrices
    A_data = array.array("d", (
        ((i * 7 + 3) % 100) / 100.0
        for i in range(total_elements)
    ))

    B_data = array.array("d", (
        ((i * 11 + 5) % 100) / 100.0
        for i in range(total_elements)
    ))

    shm_a = shared_memory.SharedMemory(
        create=True, size=A_data.itemsize * total_elements
    )
    shm_b = shared_memory.SharedMemory(
        create=True, size=B_data.itemsize * total_elements
    )
    shm_c = shared_memory.SharedMemory(
        create=True, size=8 * total_elements
    )

    shm_a.buf[:] = A_data.tobytes()
    shm_b.buf[:] = B_data.tobytes()

    results = {}
    reference = None

    try:
        with mp.Pool(processes=P_PHYS) as pool:

            # Variant 1: Naive i-j-k
            print("\n1. NAIVE i-j-k GEMM")
            tasks = create_tasks(shm_a, shm_b, shm_c, "naive")

            naive_time = benchmark(pool, tasks, shm_c)
            naive_gflops = (2 * N**3) / (naive_time / 1000) / 1e9

            print(f"Median time: {naive_time:.3f} ms")
            print(f"Throughput: {naive_gflops:.6f} GFLOP/s")

            # Variant 2: Reordered i-k-j
            print("\n2. REORDERED i-k-j GEMM")
            tasks = create_tasks(shm_a, shm_b, shm_c, "reordered")

            reordered_time = benchmark(pool, tasks, shm_c)
            reordered_gflops = (2 * N**3) / (reordered_time / 1000) / 1e9

            print(f"Median time: {reordered_time:.3f} ms")
            print(f"Throughput: {reordered_gflops:.6f} GFLOP/s")

            # Save reference result from reordered version
            reference = array.array("d", shm_c.buf.cast("d"))

            # Variant 3: Blocked GEMM
            print("\n3. BLOCKED GEMM — TILE SIZE SWEEP")
            print("-" * 70)

            best_gflops = 0.0
            best_bs = None
            best_time = None

            for bs in TILE_SIZES:
                tasks = create_tasks(
                    shm_a, shm_b, shm_c, "blocked", bs
                )

                median_ms = benchmark(pool, tasks, shm_c)
                gflops = (2 * N**3) / (median_ms / 1000) / 1e9

                error = verify_results(shm_c, reference)

                results[bs] = (median_ms, gflops, error)

                print(
                    f"BS={bs:3d} | "
                    f"Time={median_ms:10.3f} ms | "
                    f"GFLOP/s={gflops:.6f} | "
                    f"Max error={error:.3e}"
                )

                if gflops > best_gflops:
                    best_gflops = gflops
                    best_bs = bs
                    best_time = median_ms

        print("\n" + "=" * 70)
        print("FINAL GEMM RESULTS")
        print("=" * 70)

        print(f"Naive:       {naive_gflops:.6f} GFLOP/s")
        print(f"Reordered:   {reordered_gflops:.6f} GFLOP/s")
        print(f"Best tile:   BS={best_bs}")
        print(f"Best time:   {best_time:.3f} ms")
        print(f"Best tile:   {best_gflops:.6f} GFLOP/s")

        best_error = results[best_bs][2]
        print(f"Maximum numerical error: {best_error:.3e}")

        print("\nTILE WORKING-SET FOOTPRINT")
        for bs in TILE_SIZES:
            footprint = 3 * bs * bs * 8
            print(f"BS={bs:3d} | Working set={footprint:,} bytes")

        print("\nVERIFICATION HASH")
        print(emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            naive_gflops,
            best_gflops
        ))

    finally:
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()
        shm_c.close()
        shm_c.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
