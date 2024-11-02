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

# Accessed with twiddleLut[R][Ns][j][r]
twiddleLut = dict()

def fillTwiddleLut(N: int):
    global twiddleLut

    twiddleLut[2] = dict()
    twiddleLut[4] = dict()

    R=4
    Ns=1
    while Ns < N:
        twiddleLut[R][Ns] = dict()
        for j in range(0, N // R):
            angle = -2.0 * math.pi * (j % Ns) / (Ns * R)
            twiddleLut[R][Ns][j] = [complex(math.cos(r * angle), math.sin(r * angle)) for r in range(0, R)]
        Ns *= R

    # Radix-2 is only used as the first iteration
    R=2
    Ns=1
    twiddleLut[R][Ns] = dict()
    for j in range(0, N // R):
        angle = -2.0 * math.pi * (j % Ns) / (Ns * R)
        twiddleLut[R][Ns][j] = [complex(math.cos(r * angle), math.sin(r * angle)) for r in range(0, R)]
    Ns *= R

    # Radix-4 Ns differ when Radix-2 is applied on the first pass
    R=4
    while Ns < N:
        twiddleLut[R][Ns] = dict()
        for j in range(0, N // R):
            angle = -2.0 * math.pi * (j % Ns) / (Ns * R)
            twiddleLut[R][Ns][j] = [complex(math.cos(r * angle), math.sin(r * angle)) for r in range(0, R)]
        Ns *= R


def twiddle(j:int, r:int, Ns:int, R:int):
    return twiddleLut[R][Ns][j][r]

def fftIteration(
    j: int, N: int, R: int, Ns: int, data0: List[complex], data1: List[complex]
) -> None:
    v = [complex(0, 0) for i in range(R)]
    idxS = j
    for r in range(R):
        v[r] = data0[idxS + r * (N // R)]
        v[r] *= twiddle(j, r, Ns, R);

    decimate(v)

    idxD = expand(j, Ns, R)
    for r in range(R):
        data1[idxD + r * Ns] = v[r]


def isPowerOf(N: int, R: int) -> bool:
    v = R
    while v < N:
        v *= R
    return v == N


def fft(N: int, data: List[complex]) -> List[complex]:
    assert math.log2(N).is_integer(), "Input should be a power of two"
    tmp = [complex(0, 0) for i in range(len(data))]
    R = 4
    i = 0
    Ns = 1
    G = 64
    if not isPowerOf(N, R):
        print("R=2 first")
        # Input is a power of two so we need at most one iteration at R=2 to be able to run R=4 for the rest
        R = 2
        for b in range(0, ((N-1) // G) + 1):
            for t in range(0, G):
                j = b * G + t
                if j < N // R:
                    fftIteration(j, N, R, Ns, data, tmp)
        data, tmp = tmp, data
        Ns *= R
        i += 1
        R = 4
    assert N == Ns or isPowerOf(N // Ns, R)
    while Ns < N:
        for b in range(0, ((N-1) // G) + 1):
            for t in range(0, G):
                j = b * G + t
                if j < N // R:
                    fftIteration(j, N, R, Ns, data, tmp)
        data, tmp = tmp, data
        Ns *= R
        i += 1
    print(f"{i} total iterations")

    return data


def ifft(N: int, data: List[complex]) -> List[complex]:
    # NOTE:
    # This swap really easy to do if real and complex are stored separately
    # Probably the way to go for the gpu version?
    data = [complex(c.imag, c.real) for c in data]
    data = fft(N, data)
    data = [complex(c.imag, c.real) for c in data]
    data = [c / N for c in data]
    return data


def f(i: int, n: int) -> float:
    return math.sin(40 * float(i) / n) + math.sin(200 * float(i) / n)


def main():
    N = 2
    while N < 20000:
        print(f"N={N}")
        fillTwiddleLut(N)
        fv = [f(i, N) for i in range(N)]
        data = [complex(v, 0) for v in fv]
        data = fft(N, data)
        data = ifft(N, data)
        delta = sum([abs(v - c) for (v, c) in zip(fv, data)])
        assert delta <  1e-10
        print(delta)
        N *= 2


if __name__ == "__main__":
    main()
