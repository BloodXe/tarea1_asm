"""
senales.py
----------
Generación reproducible de las señales de prueba


Cada caso usa su propia semilla (numpy.random.default_rng(seed)) y una SNR objetivo. El
ruido se escala para que SNR = 10*log10( sum(s^2) / sum(n^2) ) coincida con el objetivo.
"""
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, chirp

# Parámetros comunes y configuración de los tres casos
COMUN = {
    "fs": 8000,                      # Hz
    "L": 16384,                      # muestras (2^14)  -> 2.048 s
    "tipo_senal": "multitono",
    "frecuencias_utiles_hz": [500.0, 1200.0, 2000.0],
    "amplitudes": [1.0, 0.7, 0.5],
    "fases_grados": [0.0, 35.0, -50.0],
    "banda_util_hz": [300.0, 2300.0],
    "N_bloque": 512,                 # bloque de procesamiento (potencia de 2)
    "pico_maximo": 0.80,             # amplitud digital final
    "fade_ms": 20,                   # entrada/salida suave 
    "num_impulsos": 10,              # solo se usa con ruido impulsivo/mixto
}

CASOS = {
    "caso_1": {
        "descripcion": "Multitono + interferencia sinusoidal separada (3100 Hz)",
        "seed": 101,
        "tipo_ruido": "tonal",
        "f_interferencia_hz": 3100.0,
        "snr_objetivo_db": 5.0,
    },
    "caso_2": {
        "descripcion": "Multitono + ruido blanco gaussiano de banda completa",
        "seed": 202,
        "tipo_ruido": "blanco",
        "snr_objetivo_db": 5.0,
    },
    "caso_3": {
        "descripcion": "Multitono + ruido de banda limitada (1800-2800 Hz) "
                       "que se superpone con la componente de 2000 Hz",
        "seed": 303,
        "tipo_ruido": "banda",
        "banda_ruido_hz": [1800.0, 2800.0],
        "snr_objetivo_db": 5.0,
    },
}



# Funciones del notebook base
def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def snr_db(clean, observed):
    """SNR en dB. Aquí 'observed' es la señal observada (limpia + ruido): el error es
    observed - clean. (El notebook del profesor la define así.)"""
    error = observed - clean
    return 10*np.log10((np.sum(clean**2)+1e-15)/(np.sum(error**2)+1e-15))


def fade(x, fs, ms=20):
    n = min(int(fs*ms/1000), len(x)//2)
    if n > 0:
        ramp = np.sin(np.linspace(0, np.pi/2, n))**2
        x = x.copy(); x[:n] *= ramp; x[-n:] *= ramp[::-1]
    return x


def normalizar_conjunto(*signals, peak=0.8):
    maximum = max(np.max(np.abs(x)) for x in signals)
    gain = peak/maximum if maximum > 0 else 1.0
    return [x*gain for x in signals], gain


def generar_limpia(kind, t, freqs, amps, phases_deg, rng):
    if kind == 'tono':
        return amps[0]*np.sin(2*np.pi*freqs[0]*t + np.deg2rad(phases_deg[0]))
    if kind == 'multitono':
        return sum(a*np.sin(2*np.pi*f*t + np.deg2rad(p)) for f,a,p in zip(freqs,amps,phases_deg))
    if kind == 'chirp':
        return chirp(t, f0=freqs[0], f1=freqs[-1], t1=t[-1], method='linear')
    if kind == 'armonicos':
        f0 = freqs[0]
        return sum((1/k)*np.sin(2*np.pi*k*f0*t + rng.uniform(-np.pi,np.pi)) for k in range(1,6))
    if kind == 'transitoria':
        env = np.exp(-5*(t % 1.0))*(np.sin(2*np.pi*2*t)>0)
        return env*np.sin(2*np.pi*freqs[0]*t)
    raise ValueError(f'Tipo de señal no reconocido: {kind}')


def ruido_banda(n, fs, band, rng):
    low, high = band
    if not (0 < low < high < fs/2):
        raise ValueError(f'BANDA_RUIDO debe cumplir 0 < low < high < {fs/2}')
    sos = butter(6, [low, high], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, rng.standard_normal(n))


def generar_ruido(kind, t, fs, f_int, band, impulses, rng):
    n = len(t)
    tonal = np.sin(2*np.pi*f_int*t + rng.uniform(-np.pi,np.pi))
    red = sum((1/k)*np.sin(2*np.pi*60*k*t + rng.uniform(-np.pi,np.pi)) for k in range(1,5))
    white = rng.standard_normal(n)
    band_noise = ruido_banda(n, fs, band, rng)
    impulse = np.zeros(n)
    locations = rng.choice(np.arange(int(.1*fs), n-int(.1*fs)), size=min(impulses,max(1,n//100)), replace=False)
    for loc in locations:
        width = max(2, int(rng.uniform(.0005,.004)*fs))
        stop = min(n, loc+width)
        impulse[loc:stop] += rng.choice([-1,1])*np.hanning(2*(stop-loc))[:stop-loc]
    options = {'tonal':tonal, 'red_60hz':red, 'blanco':white, 'banda':band_noise, 'impulsivo':impulse}
    if kind == 'mixto':
        return 0.55*tonal + 0.30*band_noise + 0.15*impulse
    if kind not in options: raise ValueError(f'Tipo de ruido no reconocido: {kind}')
    return options[kind]


def escalar_ruido_para_snr(clean, noise, target_db):
    target_noise_rms = rms(clean)/(10**(target_db/20))
    return noise*(target_noise_rms/(rms(noise)+1e-15))



# Generación de un caso completo como en la guia

def generar_caso(cfg_caso, comun=COMUN):
    """
    Devuelve dict con 'limpia', 'ruido', 'contaminada' (float64), 'factor_escala' y
    'fases_rad'. Las tres señales se multiplican por una misma ganancia para que el pico
    sea comun['pico_maximo']; así la SNR no cambia.
    """
    fs, L = comun["fs"], comun["L"]
    rng = np.random.default_rng(cfg_caso["seed"])
    t = np.arange(L) / fs

    clean = generar_limpia(comun["tipo_senal"], t, comun["frecuencias_utiles_hz"],
                           comun["amplitudes"], comun["fases_grados"], rng)
    clean = fade(clean, fs, comun["fade_ms"])
    banda = cfg_caso.get("banda_ruido_hz", [2500.0, 3800.0])     # solo se usa si el ruido es 'banda'
    noise_raw = generar_ruido(cfg_caso["tipo_ruido"], t, fs,
                              cfg_caso.get("f_interferencia_hz", 1750.0), tuple(banda),
                              comun["num_impulsos"], rng)
    noise = escalar_ruido_para_snr(clean, noise_raw, cfg_caso["snr_objetivo_db"])
    noisy = clean + noise
    (clean, noise, noisy), g = normalizar_conjunto(clean, noise, noisy, peak=comun["pico_maximo"])
    return {
        "limpia": clean,
        "ruido": noise,
        "contaminada": noisy,
        "fases_rad": np.deg2rad(comun["fases_grados"]),
        "factor_escala": g,
    }



# Entrada / salida (WAV PCM de 16 bits, JSON)
def guardar_wav(ruta, x, fs):
    """Guarda x (|x|<=1) como WAV PCM int16."""
    x16 = np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)
    wavfile.write(ruta, int(fs), x16)


def cargar_wav(ruta):
    """Lee un WAV int16 y lo devuelve como float en [-1, 1]."""
    fs, x16 = wavfile.read(ruta)
    return fs, x16.astype(float) / 32767.0


def guardar_json(ruta, obj):
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
