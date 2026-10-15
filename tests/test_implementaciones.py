"""
Pruebas automáticas de la Tarea 1 (ejecutar con:  python -m pytest tests -q
desde la carpeta raíz).  Comparan las implementaciones propias contra
numpy/scipy y verifican las especificaciones declaradas en el informe.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fft_propia as fp
import senales as sg
import metricas as mt
import filtros_temporales as ft
import filtrado_espectral as fe

FS = sg.COMUN["fs"]
N = sg.COMUN["N_bloque"]
E = N


@pytest.fixture(scope="module")
def casos():
    return {k: sg.generar_caso(v) for k, v in sg.CASOS.items()}


@pytest.fixture(scope="module")
def fir():
    b, _ = ft.disenar_fir_kaiser(FS, 2200.0, 2800.0, 62.0)
    return b


# FFT / DFT
@pytest.mark.parametrize("n", [8, 64, 512, 1024])
def test_fft_propia_vs_numpy(n):
    x = np.random.default_rng(n).standard_normal(n)
    assert np.max(np.abs(fp.fft(x) - np.fft.fft(x))) < 1e-10


def test_dft_directa_vs_fft():
    x = np.random.default_rng(3).standard_normal(256)
    assert np.max(np.abs(fp.dft_directa(x) - fp.fft(x))) < 1e-9


def test_ifft_inversa():
    x = np.random.default_rng(4).standard_normal(512)
    assert np.max(np.abs(fp.ifft(fp.fft(x)).real - x)) < 1e-12


def test_fft_exige_potencia_de_dos():
    with pytest.raises(ValueError):
        fp.fft(np.zeros(100))


def test_simetria_conjugada():
    x = np.random.default_rng(5).standard_normal(64)
    X = fp.fft(x)
    k = np.arange(1, 64)
    assert np.max(np.abs(X[64 - k] - np.conj(X[k]))) < 1e-12


# ---------------------------------------------------------------- señales
def test_reproducibilidad_por_semilla():
    a = sg.generar_caso(sg.CASOS["caso_2"])
    b = sg.generar_caso(sg.CASOS["caso_2"])
    assert np.array_equal(a["contaminada"], b["contaminada"])


def test_semillas_distintas_dan_ruidos_distintos(casos):
    assert not np.allclose(casos["caso_2"]["ruido"], casos["caso_3"]["ruido"])


@pytest.mark.parametrize("nombre", ["caso_1", "caso_2", "caso_3"])
def test_snr_igual_al_objetivo(casos, nombre):
    d = casos[nombre]
    assert abs(sg.snr_db(d["limpia"], d["contaminada"]) - sg.CASOS[nombre]["snr_objetivo_db"]) < 1e-9
    assert np.max(np.abs(d["contaminada"] - d["limpia"] - d["ruido"])) < 1e-12
    assert np.abs(d["contaminada"]).max() <= sg.COMUN["pico_maximo"] + 1e-12


def test_wav_ida_y_vuelta(tmp_path, casos):
    x = casos["caso_1"]["contaminada"]
    sg.guardar_wav(tmp_path / "a.wav", x, FS)
    fs2, y = sg.cargar_wav(tmp_path / "a.wav")
    assert fs2 == FS and np.max(np.abs(x - y)) < 1.0 / 32767


# ---------------------------------------------------------------- métricas
def test_metricas_valores_conocidos():
    s = np.ones(100)
    sh = np.ones(100) * 0.9
    assert mt.mse(s, sh) == pytest.approx(0.01)
    assert mt.snr_db_ref(s, sh) == pytest.approx(20.0)
    assert mt.energia_util_conservada(s, sh) == pytest.approx(90.0)
    assert mt.energia(np.array([3.0, 4.0])) == pytest.approx(25.0)


# ---------------------------------------------------------------- FIR
def test_fir_cumple_especificaciones(fir):
    esp = ft.especificaciones_medidas(fir, FS, 2200.0, 2800.0)
    assert esp["atenuacion_rechazo_dB"] >= 60.0
    assert esp["rizado_paso_dB"] <= 0.02


def test_fir_fase_lineal_y_estable(fir):
    assert np.allclose(fir, fir[::-1], atol=1e-15)           # simetría -> fase lineal
    ceros, polos = ft.polos_ceros(fir, [1.0])
    assert ft.es_estable(polos)
    assert len(polos) == len(fir) - 1 and np.all(polos == 0)   # M polos en z = 0


@pytest.mark.parametrize("nombre", ["caso_1", "caso_2", "caso_3"])
def test_ecuacion_diferencias_igual_a_lfilter(casos, fir, nombre):
    x = casos[nombre]["contaminada"][:4096]
    assert np.max(np.abs(ft.ecuacion_diferencias(fir, [1.0], x) - signal.lfilter(fir, [1.0], x))) < 1e-12


def test_ecuacion_diferencias_iir():
    b, a = signal.butter(4, 0.3)
    x = np.random.default_rng(6).standard_normal(500)
    assert np.max(np.abs(ft.ecuacion_diferencias(b, a, x) - signal.lfilter(b, a, x))) < 1e-10


# ---------------------------------------------------------------- espectral
def test_espectral_reconstruccion_perfecta_con_ganancia_1(casos):
    x = casos["caso_2"]["contaminada"]
    y, _ = fe.filtrar_espectral(x, FS, {"alpha": 0.0, "tau": 0.0, "ext": 0, "prom": 1e18})   # conserva todos los bins
    assert np.max(np.abs((y - x)[E:-E])) < 1e-10


def test_espectral_fft_propia_igual_a_numpy(casos):
    x = casos["caso_3"]["contaminada"]
    y1, _ = fe.filtrar_espectral(x, FS)
    y2, _ = fe.filtrar_espectral(x, FS, fft=np.fft.fft, ifft=np.fft.ifft)
    assert np.max(np.abs(y1 - y2)) < 1e-9


@pytest.mark.parametrize("nombre", ["caso_1", "caso_2", "caso_3"])
def test_espectral_conserva_simetria_conjugada(casos, nombre):
    _, info = fe.filtrar_espectral(casos[nombre]["contaminada"], FS)
    assert info["max_imag"].max() < 1e-12


def test_espectral_estima_frecuencia_interferente(casos):
    _, info = fe.filtrar_espectral(casos["caso_1"]["contaminada"], FS)
    f = info["f_int"]
    assert np.all(~np.isnan(f))
    assert abs(np.mean(f) - 3100.0) < FS / N          # la resolución es un bin (15.6 Hz)


def test_espectral_exige_bloque_potencia_de_dos(casos):
    with pytest.raises(ValueError):
        fe.filtrar_espectral(casos["caso_1"]["contaminada"], FS, {"N": 500})


@pytest.mark.parametrize("nombre,minimo", [("caso_1", 30.0), ("caso_2", 8.0), ("caso_3", 5.0)])
def test_espectral_mejora_snr(casos, nombre, minimo):
    d = casos[nombre]
    y, _ = fe.filtrar_espectral(d["contaminada"], FS)
    s_, x_, y_ = mt.recortar(d["limpia"], d["contaminada"], y, E=E)
    assert mt.snr_db_ref(s_, y_) - mt.snr_db_ref(s_, x_) > minimo
    assert mt.energia_util_conservada(s_, y_) > 95.0


@pytest.mark.parametrize("nombre,minimo", [("caso_1", 60.0), ("caso_2", 1.5), ("caso_3", 1.5)])
def test_fir_mejora_snr(casos, fir, nombre, minimo):
    d = casos[nombre]
    y = ft.compensar_retardo(signal.lfilter(fir, [1.0], d["contaminada"]), (len(fir) - 1) // 2)
    s_, x_, y_ = mt.recortar(d["limpia"], d["contaminada"], y, E=E)
    assert mt.snr_db_ref(s_, y_) - mt.snr_db_ref(s_, x_) > minimo


def test_clasificador_de_ruido(casos):
    esperado = {"caso_1": "interferencia tonal", "caso_2": "ruido de banda ancha",
                "caso_3": "ruido de banda limitada"}
    for nombre, tipo in esperado.items():
        assert fe.clasificar_ruido(casos[nombre]["contaminada"], FS)[0] == tipo


def test_sobreeliminacion_degrada_la_senal_util(casos):
    """Un umbral demasiado alto elimina tonos útiles: baja la energía útil."""
    d = casos["caso_2"]
    y, _ = fe.filtrar_espectral(d["contaminada"], FS, {"alpha": 24.0})
    s_, y_ = mt.recortar(d["limpia"], y, E=E)
    assert mt.energia_util_conservada(s_, y_) < 50.0
