
import multiprocessing as mp
import os
import time
import statistics
from harness import emit_verify_tag

STUDENT_ID = 230103331
N = 12_000
P = 8
CHUNK_SIZES = [1, 4, 16, 64, 256, 1024]
WARMUP_RUNS = 1
TIMED_RUNS = 3


def task_cost(i):
    return 1 + int((i / 60) ** 2)


def compute_indices(indices):
    total = 0.0

    for i in indices:
        k = task_cost(i)
        x = 1.0 + i * 0.0001

        for _ in range(k):
            x = x * 1.000001 + 0.000001

        total += x

    return total


def process_chunk(indices):
    worker_id = os.getpid()

    start = time.perf_counter()
    compute_indices(indices)
    elapsed = time.perf_counter() - start

    work = sum(task_cost(i) for i in indices)

    return worker_id, elapsed, work


def make_static_chunks():
    return [
        list(range(i * N // P, (i + 1) * N // P))
        for i in range(P)
    ]


def make_cyclic_chunks():
    return [
        list(range(worker, N, P))
        for worker in range(P)
    ]


def make_dynamic_chunks(chunk_size):
    return [
        list(range(start, min(start + chunk_size, N)))
        for start in range(0, N, chunk_size)
    ]


def run_static(pool):
    results = pool.map(process_chunk, make_static_chunks())
    return collect_results(results)


def run_cyclic(pool):
    results = pool.map(process_chunk, make_cyclic_chunks())
    return collect_results(results)


def run_dynamic(pool, chunk_size):
    chunks = make_dynamic_chunks(chunk_size)

    results = list(
        pool.imap_unordered(process_chunk, chunks, chunksize=1)
    )

    return collect_results(results)


def collect_results(results):
    worker_times = {}
    total_work = 0

    for worker_id, elapsed, work in results:
        worker_times[worker_id] = (
            worker_times.get(worker_id, 0.0) + elapsed
        )
        total_work += work

    times = list(worker_times.values())

    if not times:
        omega = 0.0
    else:
        omega = max(times) / statistics.mean(times)

    return omega, total_work


def benchmark(strategy, pool=None, chunk_size=None):
    def execute():
        if strategy == "static":
            return run_static(pool)
        elif strategy == "cyclic":
            return run_cyclic(pool)
        else:
            return run_dynamic(pool, chunk_size)

    for _ in range(WARMUP_RUNS):
        execute()

    times = []
    omegas = []

    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        omega, _ = execute()
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        omegas.append(omega)

    return statistics.median(times), statistics.median(omegas)


def benchmark_serial():
    for _ in range(WARMUP_RUNS):
        compute_indices(range(N))

    times = []

    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        compute_indices(range(N))
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def main():
    print("=" * 70)
    print("PHASE 4: LOAD BALANCING AND SCHEDULING")
    print("=" * 70)
    print(f"Tasks: {N:,}")
    print(f"Processes: {P}")
    print(f"Dynamic chunk sizes: {CHUNK_SIZES}")

    print("\n1. SERIAL BASELINE")
    serial_time = benchmark_serial()
    print(f"Serial time: {serial_time * 1000:.3f} ms")

    results = {}

    with mp.Pool(processes=P) as pool:

        print("\n2. STATIC BLOCK SCHEDULING")
        static_time, static_omega = benchmark("static", pool)
        static_speedup = serial_time / static_time

        results["Static"] = (
            static_time, static_speedup, static_omega
        )

        print(
            f"Time={static_time * 1000:.3f} ms | "
            f"Speedup={static_speedup:.4f} | "
            f"Omega={static_omega:.4f}"
        )

        print("\n3. CYCLIC SCHEDULING")
        cyclic_time, cyclic_omega = benchmark("cyclic", pool)
        cyclic_speedup = serial_time / cyclic_time

        results["Cyclic"] = (
            cyclic_time, cyclic_speedup, cyclic_omega
        )

        print(
            f"Time={cyclic_time * 1000:.3f} ms | "
            f"Speedup={cyclic_speedup:.4f} | "
            f"Omega={cyclic_omega:.4f}"
        )

        print("\n4. DYNAMIC SCHEDULING")
        print("-" * 70)

        for chunk_size in CHUNK_SIZES:
            elapsed, omega = benchmark(
                "dynamic", pool, chunk_size
            )

            speedup = serial_time / elapsed

            results[f"Dynamic-{chunk_size}"] = (
                elapsed, speedup, omega
            )

            print(
                f"Chunk={chunk_size:4d} | "
                f"Time={elapsed * 1000:9.3f} ms | "
                f"Speedup={speedup:.4f} | "
                f"Omega={omega:.4f}"
            )

    best_name = min(results, key=lambda name: results[name][0])
    best_time, best_speedup, best_omega = results[best_name]

    print("\n5. FINAL RESULTS")
    print("=" * 70)

    for name, (elapsed, speedup, omega) in results.items():
        print(
            f"{name:15s} | "
            f"Time={elapsed * 1000:9.3f} ms | "
            f"Speedup={speedup:.4f} | "
            f"Omega={omega:.4f}"
        )

    print("-" * 70)
    print(f"Best strategy: {best_name}")
    print(f"Best time: {best_time * 1000:.3f} ms")
    print(f"Best speedup: {best_speedup:.4f}")
    print(f"Best load imbalance (Omega): {best_omega:.4f}")

    print("\nVERIFICATION HASH")
    print(
        emit_verify_tag(
            "PHASE4",
            STUDENT_ID,
            best_speedup,
            best_omega
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
