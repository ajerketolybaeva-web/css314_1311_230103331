
import multiprocessing as mp
import math
import time

from harness import emit_verify_tag


STUDENT_ID = 230103331

N_STRONG = 1_600_000
N0_WEAK = 250_000

PROCESS_COUNTS = list(range(1, 9))

WARMUP_RUNS = 1
TIMED_RUNS = 3


def serial_stage(n):
    """Serial reduction and prefix filter."""
    total = 0.0
    count = int(0.05 * n)

    for i in range(count):
        x = i * 0.0001 + 1.0

        for _ in range(12):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)

        total += x

    return total


def parallel_worker(args):
    start, end = args

    total = 0.0

    for i in range(start, end):
        x = i * 0.0001 + 1.0

        for _ in range(35):
            x = x * 1.000001 + 0.000001
            x = math.sin(x) + math.cos(x)

        total += x

    return total


def make_chunks(n, p):
    chunks = []

    for t in range(p):
        start = t * n // p
        end = (t + 1) * n // p

        chunks.append((start, end))

    return chunks


def run_trial(pool, n, p):
    stage_a_start = time.perf_counter()

    serial_stage(n)

    stage_a_time = time.perf_counter() - stage_a_start

    chunks = make_chunks(n, p)

    pool.map(parallel_worker, chunks)

    return stage_a_time


def benchmark(pool, n, p):
    for _ in range(WARMUP_RUNS):
        run_trial(pool, n, p)

    samples = []
    stage_a_samples = []

    for _ in range(TIMED_RUNS):
        start = time.perf_counter()

        stage_a_time = run_trial(pool, n, p)

        end = time.perf_counter()

        samples.append((end - start) * 1000)
        stage_a_samples.append(stage_a_time * 1000)

    samples.sort()
    stage_a_samples.sort()

    median_time = samples[len(samples) // 2]
    median_stage_a = stage_a_samples[len(stage_a_samples) // 2]

    return median_time, median_stage_a


def main():
    print("=" * 65)
    print("PHASE 2: STRONG SCALING AND WEAK SCALING")
    print("=" * 65)

    print(f"Strong scaling N: {N_STRONG:,}")
    print(f"Weak scaling N0: {N0_WEAK:,}")
    print(f"Processes: {PROCESS_COUNTS}")

    # ==========================================
    # EXPERIMENT 2A: STRONG SCALING
    # ==========================================

    print("\nEXPERIMENT 2A: STRONG SCALING")
    print("-" * 65)

    strong_results = {}

    for p in PROCESS_COUNTS:
        with mp.Pool(processes=p) as pool:
            median_time, stage_a_time = benchmark(
                pool, N_STRONG, p
            )

        strong_results[p] = {
            "time": median_time,
            "stage_a": stage_a_time
        }

        print(
            f"p={p} | "
            f"Time={median_time:.3f} ms | "
            f"Stage A={stage_a_time:.3f} ms"
        )

    t1 = strong_results[1]["time"]

    strong_speedups = {}

    print("\nSTRONG SCALING METRICS")
    print("-" * 65)

    for p in PROCESS_COUNTS:
        tp = strong_results[p]["time"]

        speedup = t1 / tp
        efficiency = speedup / p

        strong_speedups[p] = speedup

        print(
            f"p={p} | "
            f"Speedup={speedup:.4f} | "
            f"Efficiency={efficiency:.4f}"
        )

    # ==========================================
    # AMDahl's LAW
    # ==========================================

    serial_fraction = (
        strong_results[1]["stage_a"] /
        strong_results[1]["time"]
    )

    theoretical_max = 1 / serial_fraction

    theoretical_8 = 1 / (
        serial_fraction +
        (1 - serial_fraction) / 8
    )

    print("\nAMDAHL'S LAW")
    print("-" * 65)

    print(f"Serial fraction: {serial_fraction:.6f}")
    print(f"Theoretical maximum speedup: {theoretical_max:.4f}")
    print(f"Theoretical speedup at p=8: {theoretical_8:.4f}")
    print(f"Observed speedup at p=8: {strong_speedups[8]:.4f}")

    # ==========================================
    # EXPERIMENT 2B: WEAK SCALING
    # ==========================================

    print("\nEXPERIMENT 2B: WEAK SCALING")
    print("-" * 65)

    weak_results = {}

    for p in PROCESS_COUNTS:
        n = p * N0_WEAK

        with mp.Pool(processes=p) as pool:
            median_time, stage_a_time = benchmark(
                pool, n, p
            )

        weak_results[p] = {
            "time": median_time,
            "stage_a": stage_a_time,
            "n": n
        }

        print(
            f"p={p} | "
            f"N={n:,} | "
            f"Time={median_time:.3f} ms | "
            f"Stage A={stage_a_time:.3f} ms"
        )

    t1_weak = weak_results[1]["time"]

    print("\nWEAK SCALING METRICS")
    print("-" * 65)

    gustafson_8 = None

    for p in PROCESS_COUNTS:
        tp = weak_results[p]["time"]

        efficiency = t1_weak / tp

        epsilon = (
            weak_results[p]["stage_a"] / tp
        )

        gustafson = p - epsilon * (p - 1)

        print(
            f"p={p} | "
            f"Weak Efficiency={efficiency:.4f} | "
            f"Epsilon={epsilon:.6f} | "
            f"Gustafson={gustafson:.4f}"
        )

        if p == 8:
            gustafson_8 = gustafson

    # ==========================================
    # VERIFICATION HASH
    # ==========================================

    print("\nVERIFICATION HASH")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedups[8],
            gustafson_8
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
