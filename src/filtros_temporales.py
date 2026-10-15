"""
Diseño y aplicación de filtros temporales (FIR / IIR)

Ecuación de diferencias general:
    y[n] = sum_{k=0}^{M} b_k x[n-k]  -  sum_{k=1}^{N} a_k y[n-k]
    H(z) = ( sum b_k z^-k ) / ( 1 + sum a_k z^-k )
"""
import numpy as np
from scipy import signal


# Diseño
def disenar_fir_kaiser(fs, f_pasa, f_rechazo, atenuacion_db):
    """
    FIR pasa bajas de fase lineal tipo I, longitud impar con ventana Kaiser.
    f_pasa: fin de la banda de paso Hz
    f_rechazo: inicio de la banda de rechazo Hz
    atenuacion_db: atenuación mínima en rechazo dB
    Con Kaiser el rizado de paso y de rechazo es el mismo delta, por lo que
    la atenuación de 60 dB implica rizado de paso apoximado de 0.0087 dB.
    """
    ancho = (f_rechazo - f_pasa) / (fs / 2.0)           # ancho de transición
    numtaps, beta = signal.kaiserord(atenuacion_db, ancho)
    if numtaps % 2 == 0:                                # tipo I: M par
        numtaps += 1
    fc = 0.5 * (f_pasa + f_rechazo)                     # corte a mitad
    b = signal.firwin(numtaps, fc, window=("kaiser", beta), fs=fs)
    return b, {"numtaps": numtaps, "beta": beta, "fc_hz": fc}


def orden_iir_equivalente(fs, f_pasa, f_rechazo, rizado_db, atenuacion_db):
    """Orden mínimo de IIR clásicos para la misma especificación (referencia)."""
    wp, ws = f_pasa / (fs / 2), f_rechazo / (fs / 2)
    return {
        "Butterworth": signal.buttord(wp, ws, rizado_db, atenuacion_db)[0],
        "Chebyshev I": signal.cheb1ord(wp, ws, rizado_db, atenuacion_db)[0],
        "Elíptico": signal.ellipord(wp, ws, rizado_db, atenuacion_db)[0],
    }

# Aplicación: ecuación de diferencias
def ecuacion_diferencias(b, a, x):
    """
    Aplica y[n] = sum b_k x[n-k] - sum_{k>=1} a_k y[n-k] con condiciones
    iniciales nulas. Se normaliza para que a[0] = 1. Implementación directa
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    b = b / a[0]
    a = a / a[0]
    x = np.asarray(x, dtype=float)
    L = len(x)
    M = len(b) - 1
    Na = len(a) - 1
    y = np.zeros(L)
    for n in range(L):
        acc = 0.0
        for k in range(M + 1):
            if n - k >= 0:
                acc += b[k] * x[n - k]
        for k in range(1, Na + 1):
            if n - k >= 0:
                acc -= a[k] * y[n - k]
        y[n] = acc
    return y


def compensar_retardo(y, D):
    """Adelanta la salida D muestras retardo de grupo del FIR de fase lineal."""
    out = np.zeros_like(y)
    out[: len(y) - D] = y[D:]
    return out


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------
def respuesta_frecuencia(b, a, fs, npts=8192):
    """Devuelve f [Hz] y H(e^jw)."""
    w, H = signal.freqz(b, a, worN=npts, fs=fs)
    return w, H


def retardo_grupo(b, a, fs, npts=8192):
    """Retardo de grupo en muestras; f [Hz]."""
    w, gd = signal.group_delay((b, a), w=npts, fs=fs)
    return w, gd


def polos_ceros(b, a=(1.0,)):
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    ceros = np.roots(b)
    polos = np.roots(a)
    extra = (len(b) - 1) - (len(a) - 1)
    if extra > 0:                                  # polos en el origen
        polos = np.concatenate([polos, np.zeros(extra)])
    elif extra < 0:                                # ceros en el origen
        ceros = np.concatenate([ceros, np.zeros(-extra)])
    return ceros, polos


def es_estable(polos, tol=1e-12):
    """Sistema causal estable, todos los polos con |p| < 1."""
    return bool(np.all(np.abs(polos) < 1.0 - tol))


def especificaciones_medidas(b, fs, f_pasa, f_rechazo, npts=16384):

    f, H = respuesta_frecuencia(b, [1.0], fs, npts)
    mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-300))
    paso = f <= f_pasa
    rech = f >= f_rechazo
    return {
        "rizado_paso_dB": float(np.max(np.abs(mag_db[paso]))),
        "atenuacion_rechazo_dB": float(-np.max(mag_db[rech])),
    }


def ganancia_en(b, fs, f0):

    w = 2 * np.pi * f0 / fs
    H = np.sum(np.asarray(b) * np.exp(-1j * w * np.arange(len(b))))
    return 20 * np.log10(abs(H))


# Cuantización (para el microcontrolador)
def cuantizar_coeficientes(b, bits):
    """Cuantiza a punto fijo con 'bits' bits (formato Q1.(bits-1)); devuelve
    los coeficientes cuantizados (float) y los enteros."""
    esc = 2 ** (bits - 1)
    q = np.clip(np.round(np.asarray(b) * esc), -esc, esc - 1).astype(np.int64)
    return q / esc, q
