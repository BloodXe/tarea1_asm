"""
Reducción de ruido por bloques mediante FFT / IFFT con solapamiento

Parámetros adaptativos se recalculan en cada bloque a partir de los datos:
sigma: nivel de ruido estimado = mediana de |X[k]|
T: umbral = max(alpha*sigma, tau*max|X[k]|); se conservan los bins con |X[k]| >= T más ext bins vecinos lóbulo principal de la ventana
f_int: frecuencia interferente estimada = pico más fuerte fuera de la banda util, si sobresale prom veces del promedio de sus vecinos 
w: ancho de rechazo (en bins) alrededor de f_int
Parámetros de ajuste: alpha, tau, ext, prom, w.
"""
import numpy as np
from fft_propia import fft as fft_propia, ifft as ifft_propia, es_potencia_de_dos


def ventana_hann(N):
    """Hann periódica: con salto N/2 cumple COLA (suma exactamente 1)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)


PARAMETROS_POR_DEFECTO = {
    "N": 512,              # tamaño de bloque - potencia de dos
    "alpha": 4.0,          # umbral sobre el nivel de ruido - mediana
    "tau": 0.30,           # umbral relativo al pico máximo del bloque
    "ext": 2,              # bins vecinos que se conservan junto a cada pico
    "prom": 6.0,           # relación pico/vecinos para declarar interferencia tonal
    "w": 3,             # ancho de rechazo en bins a cada lado
    "banda_util_hz": (300.0, 2300.0),
}


def procesar_bloque(xb, fs, par, fft=fft_propia, ifft=ifft_propia):
    """
    Procesa un bloque de N muestras. Devuelve (yb, info).
    yb ya está multiplicado por la ventana de análisis (listo para sumar).
    """
    N = par["N"]
    df = fs / N
    win = ventana_hann(N)
    X = fft(xb * win)
    mag = np.abs(X[: N // 2 + 1])               # medio espectro k = 0..N/2

    #Estimación del ruido y umbral adaptativo
    sigma = float(np.median(mag))
    T = max(par["alpha"] * sigma, par["tau"] * float(mag.max()))
    conservar = mag >= T
    # Un tono con ventana Hann ocupa +- 2 bins, lóbulo principal: se conservan
    # 'ext' bins vecinos de cada bin retenido para no recortar el tono.
    if par["ext"] > 0:
        conservar = np.convolve(conservar.astype(int),
                                np.ones(2 * par["ext"] + 1, dtype=int),
                                mode="same") > 0
    G = conservar.astype(float)# ganancia por bin: 1 = conservar, 0 = anular

    #Detección de interferencia tonal fuera de la banda útil
    f_lo, f_hi = par["banda_util_hz"]
    k_lo, k_hi = int(np.ceil(f_lo / df)), int(np.floor(f_hi / df))
    fuera = np.ones(N // 2 + 1, dtype=bool)
    fuera[k_lo:k_hi + 1] = False
    candidato = np.where(fuera, mag, 0.0)
    kp = int(np.argmax(candidato))
    f_int = np.nan
    # prominencia: el pico debe sobresalir de sus vecinos (un tono es angosto;
    # un pico de ruido no destaca de los bins que lo rodean)
    vec = np.r_[mag[max(kp - par["w"] - 8, 0): max(kp - par["w"], 0)],
                mag[min(kp + par["w"] + 1, N // 2 + 1): kp + par["w"] + 9]]
    vecinos = float(np.mean(vec)) if len(vec) else 0.0
    if candidato[kp] > par["prom"] * vecinos:
        f_int = kp * df                          # frecuencia interferente estimada
        a, b = max(kp - par["w"], 0), min(kp + par["w"], N // 2)
        G[a:b + 1] = 0.0                         # rechazo de ancho 2w+1 bins

    #Simetría conjugada: G[N-k] = G[k] 
    Gf = np.empty(N)
    Gf[: N // 2 + 1] = G
    Gf[N // 2 + 1:] = G[1: N // 2][::-1]
    Y = X * Gf
    yb = ifft(Y)
    info = {
        "sigma": sigma, "T": T, "f_int": f_int,
        "frac_bins_conservados": float(np.mean(G > 0)),
        "max_imag": float(np.max(np.abs(yb.imag))),
    }
    return yb.real, info


def filtrar_espectral(x, fs, par=None, fft=fft_propia, ifft=ifft_propia):
    """
    Filtrado espectral por bloques de N muestras con salto N/2 (50 % de
    solapamiento). La ventana Hann de análisis con salto N/2 suma 1, por lo
    que con ganancia 1 la salida reproduce la entrada (se verifica en tests).
    Devuelve (y, info) con info = dict de listas por bloque.
    """
    p = dict(PARAMETROS_POR_DEFECTO)
    if par:
        p.update(par)
    N = p["N"]
    if not es_potencia_de_dos(N):
        raise ValueError("N debe ser potencia de dos")
    hop = N // 2
    x = np.asarray(x, dtype=float)
    y = np.zeros(len(x))
    infos = []
    for ini in range(0, len(x) - N + 1, hop):
        yb, inf = procesar_bloque(x[ini:ini + N], fs, p, fft, ifft)
        y[ini:ini + N] += yb
        infos.append(inf)
    info = {k: np.array([d[k] for d in infos]) for k in infos[0]}
    return y, info


# Diagnóstico Nodo 1: características del ruido
def periodograma_promedio(x, fs, N=512):
  
    win = ventana_hann(N)
    acum = np.zeros(N // 2 + 1)
    nb = 0
    for ini in range(0, len(x) - N + 1, N // 2):
        X = fft_propia(x[ini:ini + N] * win)[: N // 2 + 1]
        acum += np.abs(X) ** 2
        nb += 1
    return np.arange(N // 2 + 1) * fs / N, acum / nb


def caracteristicas_espectrales(x, fs, N=512):
    """
    Características para identificar automáticamente el tipo de ruido
    Se calculan sobre la señal que se analice: ruido solo o contaminada.
    """
    f, P = periodograma_promedio(x, fs, N)
    P = P[1:]                                   # sin DC
    f = f[1:]
    Pn = P / P.sum()
    planitud = float(np.exp(np.mean(np.log(P + 1e-300))) / np.mean(P))
    centroide = float(np.sum(f * Pn))
    c = np.cumsum(Pn)
    f_lo = float(f[np.searchsorted(c, 0.005)])
    f_hi = float(f[min(np.searchsorted(c, 0.995), len(f) - 1)])
    return {
        "planitud_espectral": planitud,
        "centroide_hz": centroide,
        "ancho_banda_99pct_hz": f_hi - f_lo,
        "f_inf_hz": f_lo, "f_sup_hz": f_hi,
        "pico_sobre_mediana_dB": float(10 * np.log10(P.max() / np.median(P))),
    }


def clasificar_ruido(x, fs, banda_util_hz=(300.0, 2300.0), N=512):
    """
    Clasificador simple del Nodo 1 usando SOLO la señal contaminada

    Mira la región por encima de la banda útil (donde no hay señal útil):
      ocupacion = ancho (Hz) de los bins con potencia > 10 % del máximo
      
    Reglas:
      ocupacion < 150 Hz               -> interferencia tonal
      ocupacion >= 80 % de la región   -> ruido de banda ancha
      en otro caso                     -> ruido de banda limitada
    """
    f, P = periodograma_promedio(x, fs, N)
    f_hi = banda_util_hz[1]
    m = f > f_hi
    Pr = P[m]
    df = fs / N
    ocupacion = float(np.sum(Pr > 0.10 * Pr.max()) * df)
    ancho_region = float(m.sum() * df)
    if ocupacion < 150.0:
        tipo = "interferencia tonal"
    elif ocupacion >= 0.8 * ancho_region:
        tipo = "ruido de banda ancha"
    else:
        tipo = "ruido de banda limitada"
    return tipo, {"ocupacion_hz": ocupacion, "region_hz": ancho_region,
                  "f_pico_hz": float(f[m][np.argmax(Pr)])}
