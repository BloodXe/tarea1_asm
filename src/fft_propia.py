"""
DFT directa por definición y FFT propia
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
    x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+j*2*pi*k*n/N)
"""
import numpy as np


def es_potencia_de_dos(N):
    """True si N = 2^m con m >= 0."""
    return N > 0 and (N & (N - 1)) == 0


def dft_directa(x):
    """DFT calculada directamente de la definición. Costo O(N^2)."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N)  # matriz de Fourier N x N
    return W @ x


def _indices_bit_reverso(N):
    """Permutación de bits invertidos necesaria para la FFT en el tiempo."""
    bits = int(np.log2(N))
    idx = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def fft(x):
    """
    FFT radix-2 (decimación en el tiempo). Costo O(N log2 N).
    N debe ser potencia de dos.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if not es_potencia_de_dos(N):
        raise ValueError("N debe ser potencia de dos")
    X = x[_indices_bit_reverso(N)].copy()
    tam = 2
    while tam <= N:                      # una etapa por cada potencia de 2
        mitad = tam // 2
        W = np.exp(-2j * np.pi * np.arange(mitad) / tam)   # factores twiddle
        X = X.reshape(-1, tam)
        par = X[:, :mitad].copy()
        impar = X[:, mitad:] * W
        X[:, :mitad] = par + impar       # mariposa
        X[:, mitad:] = par - impar
        X = X.reshape(-1)
        tam *= 2
    return X


def ifft(X):
    """IFFT usando la FFT: x = conj(FFT(conj(X))) / N."""
    X = np.asarray(X, dtype=complex)
    return np.conj(fft(np.conj(X))) / len(X)


def medidas_lobulos(w, zp=64):
    """
    Mide una ventana w a partir de su DFT densa (relleno de ceros):
      ancho a -3 dB del lóbulo principal [bins],
      ancho nulo a nulo del lóbulo principal [bins],
      nivel del mayor lóbulo lateral respecto del pico [dB].
    """
    w = np.asarray(w, dtype=float)
    N = len(w)
    M = N * zp
    W = np.abs(fft(np.r_[w, np.zeros(M - N)]))[: M // 2 + 1]
    W = W / W[0]
    pos = np.arange(M // 2 + 1) / zp                   # en bins
    i3 = int(np.argmax(W < 1 / np.sqrt(2)))            # primer cruce bajo -3 dB
    p3 = pos[i3 - 1] + (W[i3 - 1] - 1 / np.sqrt(2)) / (W[i3 - 1] - W[i3]) * (pos[i3] - pos[i3 - 1])
    i0 = int(np.argmax(np.diff(W) > 0))                # primer mínimo (nulo)
    return {
        "ancho_3dB_bins": 2 * float(p3),
        "ancho_nulo_a_nulo_bins": 2 * float(pos[i0]),
        "lobulo_lateral_dB": float(20 * np.log10(W[i0:].max())),
    }


def espectro(x, fs, ventana=None):
    """
    Espectro unilateral de un bloque real x (longitud potencia de dos).
    Devuelve frecuencias [Hz], magnitud |X[k]| y fase arg(X[k]) [rad]
    para k = 0..N/2.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if ventana is not None:
        x = x * ventana
    X = fft(x)[: N // 2 + 1]
    f = np.arange(N // 2 + 1) * fs / N
    return f, np.abs(X), np.angle(X)
