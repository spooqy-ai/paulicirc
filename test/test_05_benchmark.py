"""
Microbenchmarks for the performance-critical Pauli-gadget operations.

Run with::

    pytest test/test_05_benchmark.py --benchmark-only

The numba-jitted kernels are warmed up at import time so that the reported
timings measure steady-state execution rather than one-off JIT compilation.
"""

import numpy as np
import pytest

from paulicirc import Circuit, Gadget

RNG_SEED = 0


def _make_circuit(num_gadgets: int, num_qubits: int, seed: int = RNG_SEED) -> Circuit:
    return Circuit.random(num_gadgets, num_qubits, rng=seed)


def _rand_codes(num_gadgets: int, seed: int = RNG_SEED) -> np.ndarray:
    # Non-zero codes (1..7) so every adjacent pair is actually commuted.
    return np.random.default_rng(seed).integers(1, 8, num_gadgets // 2, dtype=np.uint8)


def _commute_rounds(circ: Circuit, num_rounds: int, seed: int = RNG_SEED) -> Circuit:
    """Apply `num_rounds` of random non-zero commutations (the obfuscation workload)."""
    rng = np.random.default_rng(seed)
    for _ in range(num_rounds):
        codes = rng.integers(1, 8, len(circ) // 2, dtype=np.uint8)
        circ = circ.commute(codes)
    return circ


# --- Warm up the JIT kernels so benchmarks exclude compilation cost ----------

_warm = _make_circuit(8, 8)
_warm.commute(_rand_codes(8))
Gadget.random(8, rng=0).commute_past(Gadget.random(8, rng=1), 5)
del _warm


# --- Single-pair gadget commutation ------------------------------------------


@pytest.mark.parametrize("num_qubits", [8, 32, 128])
def test_bench_gadget_commute_past(benchmark, num_qubits: int) -> None:
    g = Gadget.random(num_qubits, rng=0)
    h = Gadget.random(num_qubits, rng=1)
    # Ensure odd overlap (the non-trivial path) by retrying seeds if needed.
    seed = 1
    while g.commutes_with(h):
        seed += 1
        h = Gadget.random(num_qubits, rng=seed)
    benchmark(g.commute_past, h, 5)


# --- Batched circuit commutation (THE obfuscation primitive) -----------------


@pytest.mark.parametrize(
    "num_gadgets,num_qubits",
    [
        (100, 8),
        (1000, 8),
        (1000, 32),
        (1000, 128),
        (5000, 32),
    ],
)
def test_bench_circuit_commute(benchmark, num_gadgets: int, num_qubits: int) -> None:
    circ = _make_circuit(num_gadgets, num_qubits)
    codes = _rand_codes(num_gadgets)
    result = benchmark(circ.commute, codes)
    # Sanity: commutation preserves or grows gadget count (2->3 on odd overlap).
    assert len(result) >= num_gadgets


# --- Iterated commutation: the actual obfuscation loop -----------------------


@pytest.mark.parametrize(
    "num_gadgets,num_qubits,num_rounds",
    [
        (64, 16, 5),
        (64, 64, 5),
        (256, 32, 5),
    ],
)
def test_bench_obfuscation_loop(
    benchmark, num_gadgets: int, num_qubits: int, num_rounds: int
) -> None:
    circ = _make_circuit(num_gadgets, num_qubits)
    result = benchmark(_commute_rounds, circ, num_rounds)
    # Repeated commutation inflates the circuit; record the growth factor.
    benchmark.extra_info["start_gadgets"] = num_gadgets
    benchmark.extra_info["final_gadgets"] = len(result)
    benchmark.extra_info["growth_factor"] = len(result) / max(1, num_gadgets)


# --- Construction baselines --------------------------------------------------


@pytest.mark.parametrize(
    "num_gadgets,num_qubits",
    [(1000, 32), (10000, 32), (1000, 128)],
)
def test_bench_circuit_random(benchmark, num_gadgets: int, num_qubits: int) -> None:
    benchmark(Circuit.random, num_gadgets, num_qubits, rng=RNG_SEED)
