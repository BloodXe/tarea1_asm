"""
metricas.py
"""
import numpy as np
from scipy.signal import get_window


def energia(x):
    """E_x = sum |x[n]|^2."""
    return float(np.sum(np.abs(x) ** 2))


def mse(s, s_hat):
    """MSE = (1/L) sum (s - s_hat)^2."""
    s, s_hat = np.asarray(s), np.asarray(s_hat)
    return float(np.mean((s - s_hat) ** 2))


def snr_db_ref(s, s_hat):
    """SNR_dB = 10 log10( sum s^2 / sum (s - s_hat)^2 )."""
    s, s_hat = np.asarray(s), np.asarray(s_hat)
    return float(10 * np.log10(np.sum(s ** 2) / np.sum((s - s_hat) ** 2)))


def delta_snr(snr_salida, snr_entrada):
    """Mejora de SNR en dB."""
    return snr_salida - snr_entrada


def energia_util_conservada(s, s_hat):
    """
    Porcentaje de energía útil conservada: proyección de la salida sobre la
    señal limpia, 100 * <s_hat, s> / <s, s>.
    Es 100 % si la salida contiene toda la señal útil con ganancia 1, y baja
    en la proporción de energía útil que se eliminó (exacto si cada tono se
    conserva completo o se elimina completo).
    """
    s, s_hat = np.asarray(s), np.asarray(s_hat)
    return float(100.0 * np.dot(s_hat, s) / np.dot(s, s))


def potencia_banda(x, fs, f1, f2):
    """Energía espectral de x en [f1, f2] Hz (periodograma con ventana Hann)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    w = get_window("hann", N)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(N, 1.0 / fs)
    m = (f >= f1) & (f <= f2)
    return float(np.sum(np.abs(X[m]) ** 2))


def atenuacion_interferencia_db(n, error, fs, banda):
    """
    Atenuación de la interferencia dominante en la banda dada [Hz]:
        10 log10( P_banda(n) / P_banda(s_hat - s) )
    n      : ruido original (entrada)
    error  : s_hat - s (lo que queda del ruido, más la distorsión)
    """
    f1, f2 = banda
    return 10 * np.log10(potencia_banda(n, fs, f1, f2) /
                         max(potencia_banda(error, fs, f1, f2), 1e-300))


def recortar(*senales, E):
    """Descarta E muestras en cada extremo de todas las señales."""
    return [np.asarray(s)[E:len(s) - E] for s in senales]


def evaluar(s, n, x, y, fs, banda_interf, E):
    """
    Evalúa una salida y respecto de la limpia s, el ruido n y la entrada
    contaminada x (todas alineadas en el tiempo). Devuelve un dict.
    """
    s_, n_, x_, y_ = recortar(s, n, x, y, E=E)
    snr_in = snr_db_ref(s_, x_)
    snr_out = snr_db_ref(s_, y_)
    return {
        "MSE": mse(s_, y_),
        "SNR_entrada_dB": snr_in,
        "SNR_salida_dB": snr_out,
        "Delta_SNR_dB": delta_snr(snr_out, snr_in),
        "energia_util_conservada_pct": energia_util_conservada(s_, y_),
        "atenuacion_interferencia_dB": atenuacion_interferencia_db(
            n_, y_ - s_, fs, banda_interf),
    }
