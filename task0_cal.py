
import time
import multiprocessing as mp

from harness import emit_verify_tag


STUDENT_ID = 230103331
ITERATIONS = 2_000_000


def main():
    print("PHASE 0: TIMER CALIBRATION")
    print("-" * 40)

    logical_cores = mp.cpu_count()

    print(f"Logical CPU cores: {logical_cores}")
    print(f"Timer reads: {ITERATIONS:,}")

    # Measure timer call latency
    start = time.perf_counter_ns()

    for _ in range(ITERATIONS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_time_ns = end - start
    average_latency_ns = total_time_ns / ITERATIONS

    print(f"Total time: {total_time_ns:,} ns")
    print(f"Average latency: {average_latency_ns:.4f} ns")

    # Generate verification hash
    verify = emit_verify_tag(
        "PHASE0",
        STUDENT_ID,
        average_latency_ns,
        float(logical_cores)
    )

    print("\nVerification:")
    print(verify)


if __name__ == "__main__":
    main()
