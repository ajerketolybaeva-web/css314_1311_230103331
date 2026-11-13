
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics

P_PHYS = 8
SIZE = 96 * 1024 * 1024  # 96 MB per buffer
FLOPS_PER_WORKER = 80_000_000
WARMUP_RUNS = 1
TIMED_RUNS = 3


def copy_worker(args):
    name_a, name_b, start, end = args

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)

    mv_a = shm_a.buf
    mv_b = shm_b.buf

    mv_a[start:end] = mv_b[start:end]

    del mv_a, mv_b
    shm_a.close()
    shm_b.close()


def compute_worker(iterations):
    a = 1.000001
    b = 0.000001

    x0 = 1.0
    x1 = 1.1
    x2 = 1.2
    x3 = 1.3
    x4 = 1.4
    x5 = 1.5
    x6 = 1.6
    x7 = 1.7
    x8 = 1.8
    x9 = 1.9
    x10 = 2.0
    x11 = 2.1
    x12 = 2.2
    x13 = 2.3
    x14 = 2.4
    x15 = 2.5

    for _ in range(iterations):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b

    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15


def median_time(fn, warmup=WARMUP_RUNS, runs=TIMED_RUNS):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)

    return statistics.median(times)


def main():
    print("=" * 65)
    print("PHASE 3A: EMPIRICAL ROOFLINE MODEL")
    print("=" * 65)
    print(f"Physical processes: {P_PHYS}")
    print(f"Buffer size: {SIZE / (1024 * 1024):.0f} MB each")

    # -------------------------------------------------
    # 1. Sustained DRAM bandwidth
    # -------------------------------------------------
    print("\n1. MEASURING DRAM BANDWIDTH")

    shm_a = shared_memory.SharedMemory(create=True, size=SIZE)
    shm_b = shared_memory.SharedMemory(create=True, size=SIZE)

    # Initialize source buffer
    shm_b.buf[:] = b"\x01" * SIZE

    chunk = SIZE // P_PHYS
    tasks = []

    for i in range(P_PHYS):
        start = i * chunk
        end = (i + 1) * chunk if i < P_PHYS - 1 else SIZE
        tasks.append((shm_a.name, shm_b.name, start, end))

    try:
        with mp.Pool(processes=P_PHYS) as pool:
            def copy_all():
                pool.map(copy_worker, tasks)

            copy_seconds = median_time(copy_all)

        bandwidth = (2 * SIZE) / copy_seconds / 1e9

        print(f"Median copy time: {copy_seconds * 1000:.3f} ms")
        print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")

    finally:
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()

    # -------------------------------------------------
    # 2. Sustained Python compute ceiling
    # -------------------------------------------------
    print("\n2. MEASURING PYTHON COMPUTE CEILING")

    iterations = FLOPS_PER_WORKER // 32

    with mp.Pool(processes=P_PHYS) as pool:
        def compute_all():
            pool.map(compute_worker, [iterations] * P_PHYS)

        compute_seconds = median_time(compute_all)

    total_flops = FLOPS_PER_WORKER * P_PHYS
    compute_gflops = total_flops / compute_seconds / 1e9

    print(f"FLOPs per worker: {FLOPS_PER_WORKER:,}")
    print(f"Median compute time: {compute_seconds * 1000:.3f} ms")
    print(f"Sustained compute ceiling: {compute_gflops:.6f} GFLOP/s")

    # -------------------------------------------------
    # 3. Empirical ridge point
    # -------------------------------------------------
    ridge_point = compute_gflops / bandwidth

    print("\n3. EMPIRICAL ROOFLINE RESULTS")
    print("-" * 65)
    print(f"DRAM Bandwidth:       {bandwidth:.4f} GB/s")
    print(f"Python Compute:       {compute_gflops:.6f} GFLOP/s")
    print(f"Ridge Point:          {ridge_point:.6f} FLOPs/Byte")
    print("=" * 65)


if __name__ == "__main__":
    mp.freeze_support()
    main()
