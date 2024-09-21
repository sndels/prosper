import math
import cmath
from typing import List, Tuple


# Based on
# High Performance Discrete Fourier Transforms on Graphics Processors
# By Govindaraju et al.

def mulI(c: complex) -> complex:
    return c * complex(0, 1)


def decimate(v: List[complex]) -> None:
    if len(v) == 2:
        v0 = v[0]
        v1 = v[1]
        v[0] = v0 + v1
        v[1] = v0 - v1
    elif len(v) == 4:
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        v3 = v[3]
        v[0] = v0 + v1 + v2 + v3
        v[1] = v0 - mulI(v1) - v2 + mulI(v3)
        v[2] = v0 - v1 + v2 - v3
        v[3] = v0 + mulI(v1) - v2 - mulI(v3)
    else:
        raise ValueError("Unsupported radix")


def expand(idxL: int, N1: int, N2: int) -> int:
    return (idxL // N1) * N1 * N2 + (idxL % N1)


def fftIteration(
   i:int, j: int, N: int, R: int, Ns: int, data0: List[List[complex]], data1: List[List[complex]]
) -> None:
    v = [complex(0, 0) for i in range(R)]
    idxS = j
    angle = -2.0 * math.pi * (j % Ns) / (Ns * R)
    for r in range(R):
        v[r] = data0[i][idxS + r * (N // R)]
        # TODO: sin/cos are expensive. lookups or some approx better?
        v[r] *= complex(math.cos(r * angle), math.sin(r * angle))

    decimate(v)

    idxD = expand(j, Ns, R)
    for r in range(R):
        data1[i][idxD + r * Ns] = v[r]


def isPowerOf(N: int, R: int) -> bool:
    v = R
    while v < N:
        v *= R
    return v == N


def fft(transpose: bool, data: List[List[complex]]) -> List[List[complex]]:
    N = len(data)
    assert math.log2(N).is_integer(), "Input should be a power of two"
    assert len(data) == len(data[0])
    tmp = [[complex(0, 0) for i in range(N)] for j in range(N)]
    R = 4
    iterations = 0
    Ns = 1
    G = 64
    if not isPowerOf(N, R):
        # Input is a power of two so we need at most one iteration at R=2 to be able to run R=4 for the rest
        R = 2
        for b in range(0, ((N-1) // G) + 1):
            for i in range(N):
                for t in range(0, G):
                    j = b * G + t
                    col = i if transpose else j
                    row = j if transpose else i
                    if col < N // R and row < N:
                        fftIteration(row, col, N, R, Ns, data, tmp)
        data, tmp = tmp, data
        Ns *= R
        iterations += 1
        R = 4
    assert N == Ns or isPowerOf(N // Ns, R)
    while Ns < N:
        for b in range(0, ((N-1) // G) + 1):
            for i in range(N):
                for t in range(0, G):
                    j = b * G + t
                    col = i if transpose else j
                    row = j if transpose else i
                    if col < N // R and row < N:
                        fftIteration(row, col, N, R, Ns, data, tmp)
        data, tmp = tmp, data
        Ns *= R
        iterations += 1
    print(f"{iterations} total iterations")

    return data


def f(i: int, n: int) -> float:
    return math.sin(40 * float(i) / n) + math.sin(200 * float(i) / n)


def main():
    N = 2
    while N < 20000:
        print(f"N={N}")
        fv = [[f(i, N) for i in range(N)] for j in range(N)]

        data = [[complex(v, 0) for v in row] for row in fv]
        data = fft(False,  data)
        data = fft(True,  data)

        data = [[complex(c.imag, c.real) for c in row] for row in data]
        data = fft(False, data)
        data = fft(True, data)
        data = [[complex(c.imag, c.real) for c in row] for row in data]
        data = [[c / (N*N) for c in row] for row in data]

        delta = max([abs(v - c) for (v, c) in zip([v for row in fv for v in row], [v for row in data for v in row])])
        assert delta <  1e-14
        print(delta)
        N *= 2


if __name__ == "__main__":
    main()
