
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103331
M = 24_000_000

P_PHYS = 8
PROCESS_COUNTS = [1, 2, 4, 8]
STRIDES = [1, 2, 4, 8, 16, 32]


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc

    del arr
    shm.close()


def run_false_sharing(p, stride):
    shm = shared_memory.SharedMemory(
        create=True,
        size=4096
    )

    processes = []

    try:
        for t in range(p):
            proc = mp.Process(
                target=fs_worker,
                args=(
                    shm.name,
                    t * stride,
                    M // p
                )
            )
            processes.append(proc)

        start = time.perf_counter_ns()

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        end = time.perf_counter_ns()

        for proc in processes:
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {proc.exitcode}"
                )

        return (end - start) / 1e6

    finally:
        shm.close()
        shm.unlink()


def benchmark_case(p, stride):
    stats = run_benchmark(
        lambda: run_false_sharing(p, stride),
        warmup_runs=2,
        timed_runs=7
    )

    throughput = M / (stats.median_ms * 1000)

    return stats, throughput


def main():
    print("=" * 55)
    print("PHASE 1: FALSE SHARING")
    print("=" * 55)

    print(f"Total operations: {M:,}")
    print(f"Physical cores: {P_PHYS}")

    stride_results = {}

    print("\nTABLE 1A: STRIDE SENSITIVITY")
    print("-" * 55)

    for stride in STRIDES:
        stats, throughput = benchmark_case(P_PHYS, stride)

        stride_results[stride] = stats.median_ms

        print(
            f"Stride={stride:2d} | "
            f"Bytes={stride * 8:3d} | "
            f"Median={stats.median_ms:10.3f} ms | "
            f"IQR={stats.iqr_ms:9.3f} ms | "
            f"Throughput={throughput:.4f} Mops/s"
        )

    print("\nTABLE 1B: PROCESS SCALING")
    print("-" * 55)

    scaling_results = {}

    for stride in [1, 16]:
        print(f"\nStride = {stride}")

        times = {}

        for p in PROCESS_COUNTS:
            stats, throughput = benchmark_case(p, stride)
            times[p] = stats.median_ms

        baseline = times[1]

        for p in PROCESS_COUNTS:
            speedup = baseline / times[p]
            efficiency = speedup / p

            if p == 1:
                karp_flatt = float("nan")
            else:
                karp_flatt = (
                    (1 / speedup - 1 / p) /
                    (1 - 1 / p)
                )

            print(
                f"p={p} | "
                f"Time={times[p]:.3f} ms | "
                f"Speedup={speedup:.4f} | "
                f"Efficiency={efficiency:.4f} | "
                f"Karp-Flatt={karp_flatt:.6f}"
            )

        scaling_results[stride] = times

    print("\nVERIFICATION HASHES")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            stride_results[1],
            stride_results[16]
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
