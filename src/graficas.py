"""
graficas.py
-----------
Funciones de graficación usadas por el notebook. Todas guardan la figura en
'ruta' (PNG) y la devuelven. Convención: ejes con magnitud y unidades.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fft_propia as fp
from filtrado_espectral import ventana_hann

C_LIMPIA = "#1b9e77"
C_CONT = "#8c8c8c"
C_RUIDO = "#c0392b"
C_FIR = "#1f77b4"
C_ESP = "#ff7f0e"
C_REF = "#444444"


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 170, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False,
    })


estilo()


def _guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def _db(x, piso=-200.0):
    return np.maximum(20 * np.log10(np.maximum(np.abs(x), 1e-300)), piso)


def dft_denso(x, zp=16):
    """DFT con relleno de ceros (zero-padding) para ver la forma continua.
    Devuelve la posición en bins del bloque original y X."""
    N = len(x)
    M = N * zp
    X = fp.fft(np.r_[x, np.zeros(M - N)])[: M // 2 + 1]
    return np.arange(M // 2 + 1) / zp, X



# 2.1 Fundamentos
def fig_resolucion(fs, f1, f2, Ns, ruta):
    """Dos tonos cercanos analizados con distintos N (ventana Hann)."""
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 4.6), sharey=True)
    for ax, N in zip(axs.ravel(), Ns):
        n = np.arange(N)
        x = np.cos(2 * np.pi * f1 * n / fs) + np.cos(2 * np.pi * f2 * n / fs)
        w = ventana_hann(N)
        pos, Xd = dft_denso(x * w, zp=32)
        norm = np.sum(w) / 2
        ax.plot(pos * fs / N, _db(Xd / norm, -80), color=C_REF, lw=1.0,
                label="forma continua (relleno de ceros)")
        Xk = fp.fft(x * w)[: N // 2 + 1]
        k = np.arange(N // 2 + 1)
        ax.plot(k * fs / N, _db(Xk / norm, -80), "o", ms=3.5, color=C_FIR,
                label="bins de la DFT de N puntos")
        for f0 in (f1, f2):
            ax.axvline(f0, color=C_RUIDO, lw=0.8, ls="--")
        ax.set_xlim(f1 - 100, f2 + 100)
        ax.set_ylim(-60, 5)
        ax.set_title(f"N = {N}  ($\\Delta f$ = {fs / N:.2f} Hz, T = {1000 * N / fs:.0f} ms)")
        ax.set_xlabel("Frecuencia [Hz]")
    axs[0, 0].set_ylabel("Magnitud [dB]")
    axs[1, 0].set_ylabel("Magnitud [dB]")
    axs[0, 0].legend(loc="lower left")
    _guardar(fig, ruta)
    return fig


def fig_ventanas_magnitud(N, fs, k_al, k_fu, phi, ruta, zp=16):
    """Tono alineado (k_al) y entre bins (k_fu): ventana rectangular vs Hann."""
    n = np.arange(N)
    ventanas = {"Rectangular": np.ones(N), "Hann": ventana_hann(N)}
    colores = {"Rectangular": C_FIR, "Hann": C_ESP}
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.0), sharey="row")
    for i, (k0, nom) in enumerate([(k_al, "alineado con un bin"),
                                   (k_fu, "entre dos bins")]):
        x = np.cos(2 * np.pi * k0 * n / N + phi)
        for j, (rango, tit) in enumerate([(12, "zoom $\\pm$12 bins"),
                                          (N // 2, "todos los bins")]):
            ax = axs[i, j]
            for nomw, w in ventanas.items():
                norm = np.sum(w) / 2
                Xk = fp.fft(x * w)[: N // 2 + 1]
                if j == 0:     # zoom: forma continua (relleno de ceros) + bins reales
                    pos, Xd = dft_denso(x * w, zp=zp)
                    ax.plot(pos, _db(Xd / norm, -140), color=colores[nomw], lw=0.9,
                            label=nomw)
                    ax.plot(np.arange(N // 2 + 1), _db(Xk / norm, -140), "o", ms=3,
                            color=colores[nomw])
                else:          # vista completa: solo los bins de la DFT de N puntos
                    ax.plot(np.arange(N // 2 + 1), _db(Xk / norm, -140), "o-", ms=1.5,
                            lw=0.5, color=colores[nomw], label=nomw)
            ax.set_ylim(-140, 8)
            if j == 0:
                ax.set_xlim(k0 - rango, k0 + rango)
            else:
                ax.set_xlim(0, N // 2)
            ax.set_title(("Alineado" if i == 0 else "Entre bins") + f" (k = {k0}): {tit}", fontsize=9)
            ax.set_xlabel("Bin k")
        axs[i, 0].set_ylabel("Magnitud normalizada [dB]")
    axs[0, 1].legend(loc="upper right", markerscale=3)
    _guardar(fig, ruta)
    return fig


def fig_ventanas_fase(N, k_al, k_fu, phi, ruta, umbral_db=-100.0):
    """Fase de los bins significativos (|X| > umbral) para cada caso."""
    n = np.arange(N)
    ventanas = {"Rectangular": np.ones(N), "Hann": ventana_hann(N)}
    colores = {"Rectangular": C_FIR, "Hann": C_ESP}
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, (k0, nom) in zip(axs, [(k_al, "alineado (k = %d)" % k_al),
                                    (k_fu, "entre bins (k = %.1f)" % k_fu)]):
        x = np.cos(2 * np.pi * k0 * n / N + phi)
        for nomw, w in ventanas.items():
            X = fp.fft(x * w)[: N // 2 + 1]
            mag = _db(X / (np.sum(w) / 2), -300)
            k = np.arange(N // 2 + 1)
            m = mag > umbral_db
            ax.stem(k[m], np.angle(X[m]), linefmt=colores[nomw], markerfmt="o",
                    basefmt=" ", label=f"{nomw} ({m.sum()} bin{'s' if m.sum() != 1 else ''} sobre {umbral_db:.0f} dB)")
        ax.axhline(phi, color=C_REF, ls=":", lw=0.8)
        ax.set_xlim(k0 - 8, k0 + 8)
        ax.set_title("Tono " + nom)
        ax.set_xlabel("Bin k")
        ax.legend(loc="lower right")
    axs[0].set_ylabel("Fase [rad]")
    axs[0].text(0.02, 0.92, "línea punteada: fase inicial del tono", transform=axs[0].transAxes,
                fontsize=7, color=C_REF)
    _guardar(fig, ruta)
    return fig


def fig_simetria(N, x, ruta):
    """Magnitud (par) y fase (impar) del espectro completo de una señal real."""
    X = fp.fft(x)
    k = np.arange(N)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    axs[0].stem(k, np.abs(X), linefmt=C_FIR, markerfmt="o", basefmt=" ")
    axs[0].axvline(N / 2, color=C_REF, ls="--", lw=0.8)
    axs[0].set_title("|X[k]|: simetría par, |X[N-k]| = |X[k]|")
    axs[0].set_xlabel("Bin k")
    axs[0].set_ylabel("Magnitud")
    m = np.abs(X) > 1e-6
    axs[1].stem(k[m], np.angle(X[m]), linefmt=C_ESP, markerfmt="o", basefmt=" ")
    axs[1].axvline(N / 2, color=C_REF, ls="--", lw=0.8)
    axs[1].set_title("arg X[k]: simetría impar, arg X[N-k] = -arg X[k]")
    axs[1].set_xlabel("Bin k")
    axs[1].set_ylabel("Fase [rad]")
    _guardar(fig, ruta)
    return fig

# 2.2 Caracterización de señales
def espectro_amplitud(x, fs):
    """Espectro unilateral de amplitud (Hann, FFT propia) en dB re 1 (amplitud
    de una senoidal). Devuelve f [Hz], dB, fase [rad] y |X| normalizado."""
    L = len(x)
    w = ventana_hann(L)
    X = fp.fft(x * w)[: L // 2 + 1]
    amp = np.abs(X) * 2 / np.sum(w)
    return np.arange(L // 2 + 1) * fs / L, _db(amp, -140), np.angle(X), amp


def energia_por_bandas(x, fs, ancho=500.0):
    """Porcentaje de energía por bandas de 'ancho' Hz (FFT con rectangular)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    P = np.abs(X) ** 2
    bordes = np.arange(0, fs / 2 + ancho, ancho)
    e = np.array([P[(f >= a) & (f < b)].sum() for a, b in zip(bordes[:-1], bordes[1:])])
    e[-1] += P[f >= bordes[-1]].sum()
    return bordes, 100 * e / e.sum()


def fig_caso(titulo, fs, s, n, x, ruta, t_ms=25.0, f_utiles=()):
    """Señal temporal, magnitud, fase y energía por bandas."""
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.4))
    t = np.arange(len(s)) / fs * 1000
    m = t <= t_ms
    ax = axs[0, 0]
    ax.plot(t[m], x[m], color=C_CONT, lw=0.8, label="contaminada")
    ax.plot(t[m], s[m], color=C_LIMPIA, lw=1.4, label="limpia")
    ax.set_xlabel("Tiempo [ms]")
    ax.set_ylabel("Amplitud [u.a., FS = 1]")
    ax.set_title("Dominio del tiempo (primeros %d ms)" % t_ms)
    ax.legend(loc="upper right", ncol=2)

    f, dbx, phx, ampx = espectro_amplitud(x, fs)
    _, dbs, phs, amps = espectro_amplitud(s, fs)
    ax = axs[0, 1]
    ax.plot(f, dbx, color=C_CONT, lw=0.6, label="contaminada")
    ax.plot(f, dbs, color=C_LIMPIA, lw=0.8, label="limpia")
    for f0 in f_utiles:
        ax.axvline(f0, color=C_REF, ls=":", lw=0.7)
    ax.set_ylim(-100, 5)
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Amplitud [dB re 1]")
    ax.set_title("Magnitud (Hann, $\\Delta f$ = %.3f Hz)" % (fs / len(x)))
    ax.legend(loc="upper right", ncol=2)

    ax = axs[1, 0]
    dom = ampx > 0.05 * ampx.max()
    ax.plot(f[~dom], phx[~dom], ".", ms=1.2, color="#bbbbbb", label="bins de ruido (fase aleatoria)")
    ax.plot(f[dom], phx[dom], "o", ms=3, color=C_RUIDO, label="bins dominantes (> 5 % del máximo)")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Fase [rad]")
    ax.set_xlim(0, fs / 2)
    ax.set_title("Fase de la señal contaminada")
    ax.legend(loc="lower right", markerscale=2)

    ax = axs[1, 1]
    bordes, es = energia_por_bandas(s, fs)
    _, en = energia_por_bandas(n, fs)
    cen = 0.5 * (bordes[:-1] + bordes[1:])
    ax.bar(cen - 100, es, width=190, color=C_LIMPIA, label="señal limpia")
    ax.bar(cen + 100, en, width=190, color=C_RUIDO, label="ruido")
    ax.set_xlabel("Banda [Hz] (centro)")
    ax.set_ylabel("Energía por banda [%]")
    ax.set_xticks(cen)
    ax.set_xticklabels([f"{int(c)}" for c in cen], fontsize=7)
    ax.set_title("Distribución de energía en bandas de 500 Hz")
    ax.legend(loc="upper center")
    fig.suptitle(titulo, fontsize=11, y=1.0)
    _guardar(fig, ruta)
    return fig


# 2.3 Filtro temporal
def fig_respuesta_fir(f, H, f_gd, gd, fs, f_pasa, f_rechazo, A_rech, ruta, f_int=None):
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0))
    ax = axs[0]
    ax.plot(f, 20 * np.log10(np.maximum(np.abs(H), 1e-12)), color=C_FIR)
    ax.axvspan(f_rechazo, fs / 2, color=C_RUIDO, alpha=0.08, label="banda de rechazo")
    ax.axvspan(0, f_pasa, color=C_LIMPIA, alpha=0.10, label="banda de paso")
    ax.axhline(-A_rech, color=C_REF, ls="--", lw=0.8)
    if f_int is not None:
        ax.axvline(f_int, color=C_RUIDO, ls=":", lw=1.0)
    ax.set_ylim(-110, 5)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("$|H(e^{j\\omega})|$ [dB]")
    ax.set_title("Magnitud")
    ax.legend(loc="lower left")
    ax = axs[1]
    ax.plot(f, np.unwrap(np.angle(H)), color=C_FIR)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Fase [rad]")
    ax.set_title("Fase (lineal en la banda de paso)")
    ax = axs[2]
    ax.plot(f_gd, gd, color=C_FIR)
    ax.set_ylim(0, 2 * np.median(gd))
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Retardo de grupo [muestras]")
    ax.set_title("Retardo de grupo constante")
    _guardar(fig, ruta)
    return fig


def fig_polos_ceros(ceros, polos, ruta):
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=C_REF, lw=0.8)
    ax.plot(ceros.real, ceros.imag, "o", mfc="none", color=C_FIR, label=f"ceros ({len(ceros)})")
    ax.plot(polos.real, polos.imag, "x", color=C_RUIDO, ms=8,
            label=f"polos en z = 0 (multiplicidad {len(polos)})")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginaria")
    ax.set_title("Plano z: ceros y polos del FIR")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=1)
    _guardar(fig, ruta)
    return fig


def fig_espectros_salida(datos, fs, ruta, f_utiles=()):
    """datos: lista de (titulo, x, y_fir, y_esp). Magnitud en dB para cada caso."""
    fig, axs = plt.subplots(len(datos), 1, figsize=(7.4, 2.5 * len(datos)), sharex=True)
    for ax, (tit, x, yf, ye) in zip(np.atleast_1d(axs), datos):
        f, dx, _, _ = espectro_amplitud(x, fs)
        _, df_, _, _ = espectro_amplitud(yf, fs)
        _, de, _, _ = espectro_amplitud(ye, fs)
        ax.plot(f, dx, color=C_CONT, lw=0.6, label="entrada contaminada")
        ax.plot(f, df_, color=C_FIR, lw=0.7, label="salida FIR")
        ax.plot(f, de, color=C_ESP, lw=0.7, label="salida espectral")
        for f0 in f_utiles:
            ax.axvline(f0, color=C_REF, ls=":", lw=0.7)
        ax.set_ylim(-100, 5)
        ax.set_xlim(0, fs / 2)
        ax.set_ylabel("Amplitud [dB re 1]")
        ax.set_title(tit)
    np.atleast_1d(axs)[0].legend(loc="upper right", ncol=3)
    np.atleast_1d(axs)[-1].set_xlabel("Frecuencia [Hz]")
    _guardar(fig, ruta)
    return fig


def fig_tiempo_salidas(datos, fs, ruta, t0_ms=100.0, dur_ms=15.0):
    """datos: lista de (titulo, s, x, yf, ye) ya alineadas; zoom temporal."""
    fig, axs = plt.subplots(len(datos), 1, figsize=(7.4, 2.4 * len(datos)), sharex=True)
    t = np.arange(len(datos[0][1])) / fs * 1000
    m = (t >= t0_ms) & (t <= t0_ms + dur_ms)
    for ax, (tit, s, x, yf, ye) in zip(np.atleast_1d(axs), datos):
        ax.plot(t[m], x[m], color=C_CONT, lw=0.7, label="contaminada")
        ax.plot(t[m], s[m], color=C_LIMPIA, lw=1.6, label="limpia (referencia)")
        ax.plot(t[m], yf[m], color=C_FIR, lw=1.0, ls="--", label="salida FIR")
        ax.plot(t[m], ye[m], color=C_ESP, lw=1.0, ls="-.", label="salida espectral")
        ax.set_ylabel("Amplitud [u.a.]")
        ax.set_title(tit)
    np.atleast_1d(axs)[0].legend(loc="upper right", ncol=4)
    np.atleast_1d(axs)[-1].set_xlabel("Tiempo [ms]")
    _guardar(fig, ruta)
    return fig


# 2.4 Filtrado espectral
def fig_bloque_espectral(fs, N, xb, yb, T, f_int, ruta, titulo=""):
    """Un bloque: espectro de entrada, umbral T y espectro de salida.
    xb: bloque de entrada sin ventana; yb: salida del bloque (ya ventaneada)."""
    win = ventana_hann(N)
    X = np.abs(fp.fft(xb * win)[: N // 2 + 1])
    Y = np.abs(fp.fft(yb)[: N // 2 + 1])
    f = np.arange(N // 2 + 1) * fs / N
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(f, _db(X, -60), color=C_CONT, lw=0.9, label="|X[k]| entrada")
    ax.plot(f, _db(Y, -60), color=C_ESP, lw=1.0, label="|Y[k]| tras modificar el espectro")
    ax.axhline(20 * np.log10(T), color=C_RUIDO, ls="--", lw=0.9, label="umbral adaptativo T")
    if not np.isnan(f_int):
        ax.axvline(f_int, color=C_REF, ls=":", lw=1.0)
        ax.text(f_int + 30, ax.get_ylim()[1] - 12, f"$f_{{int}}$ est. = {f_int:.0f} Hz", fontsize=8)
    ax.set_ylim(-20, _db(X, -60).max() + 8)
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Magnitud [dB]")
    ax.set_title(titulo)
    ax.legend(loc="upper right", ncol=3)
    _guardar(fig, ruta)
    return fig


def fig_compromiso(res_alpha, res_tau, ruta):
    """
    res_*: dict con 'x' (valores del parámetro), 'dsnr', 'eutil', 'eliminada'
    (porcentaje de energía de la entrada que se elimina).
    """
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, r, xl, tit in [(axs[0], res_alpha, "$\\alpha$ (umbral sobre el nivel de ruido)",
                            "Caso 2 (ruido blanco): barrido de $\\alpha$"),
                           (axs[1], res_tau, "$\\tau$ (umbral relativo al pico)",
                            "Caso 3 (banda superpuesta): barrido de $\\tau$")]:
        ax.plot(r["x"], r["eliminada"], "s-", ms=3, color=C_RUIDO, label="energía de entrada eliminada [%]")
        ax.plot(r["x"], r["eutil"], "o-", ms=3, color=C_LIMPIA, label="energía útil conservada [%]")
        ax.set_xlabel(xl)
        ax.set_ylabel("Porcentaje [%]")
        ax.set_ylim(-5, 110)
        ax.set_title(tit)
        ax2 = ax.twinx()
        ax2.plot(r["x"], r["dsnr"], "^-", ms=3, color=C_FIR, label="$\\Delta$SNR [dB]")
        ax2.set_ylabel("$\\Delta$SNR [dB]")
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        i = int(np.argmax(r["dsnr"]))
        ax2.plot(r["x"][i], r["dsnr"][i], "*", ms=11, color=C_FIR)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7)
    _guardar(fig, ruta)
    return fig


# 2.5 Comparación
def fig_barras_metricas(df, ruta):
    """Barras de MSE y ΔSNR por caso y método (df con columnas caso, metodo)."""
    casos = list(df["caso"].unique())
    metodos = ["Entrada", "FIR", "Espectral"]
    colores = {"Entrada": C_CONT, "FIR": C_FIR, "Espectral": C_ESP}
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ancho = 0.26
    for j, (col, yl, log) in enumerate([("MSE", "MSE [u.a.$^2$] (escala log)", True),
                                         ("Delta_SNR_dB", "$\\Delta$SNR [dB]", False)]):
        ax = axs[j]
        for i, m in enumerate(metodos):
            vals = [df[(df.caso == c) & (df.metodo == m)][col].values[0] for c in casos]
            ax.bar(np.arange(len(casos)) + (i - 1) * ancho, vals, ancho,
                   color=colores[m], label=m)
        ax.set_xticks(np.arange(len(casos)))
        ax.set_xticklabels(casos)
        ax.set_ylabel(yl)
        if log:
            ax.set_yscale("log")
    axs[0].legend(loc="upper right")
    _guardar(fig, ruta)
    return fig
