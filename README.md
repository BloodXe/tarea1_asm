# CE 1110 - Tarea 1: caracterización espectral y prototipado de filtros para reducción de ruido

https://github.com/BloodXe/tarea1_asm.git

https://github.com/BloodXe/tarea1_asm/releases/tag/tarea1_prototipo


## Contenido

```
tarea1_grupoXX/
|-- informe.pdf                  Informe (respuestas, fundamentos, resultados, análisis)
|-- notebook.ipynb               Notebook ejecutable: regenera señales, figuras y métricas
|-- src/
|   |-- fft_propia.py            DFT directa y FFT radix-2 propias
|   |-- senales.py               Generación reproducible de señales (semillas), WAV, JSON
|   |-- filtros_temporales.py    Diseño FIR (Kaiser), ecuación de diferencias, polos/ceros, cuantización
|   |-- filtrado_espectral.py    Filtrado por bloques FFT/IFFT, diagnóstico y clasificador de ruido
|   |-- metricas.py              MSE, SNR, delta SNR, energía útil, atenuación
|   `-- graficas.py              Figuras del notebook
|-- tests/test_implementaciones.py   Pruebas automáticas (propia vs numpy/scipy, especificaciones)
|-- pruebas/caso_1|2|3/          01_referencia_limpia.wav, 02_ruido.wav, 03_entrada_contaminada.wav y
|                                metadatos_prueba.json de cada caso (mismos nombres del notebook del profesor)
|-- resultados/
|   |-- metricas.csv             Tabla comparativa (3 casos x {entrada, FIR, espectral})
|   |-- barrido_compromiso.csv   Barridos de parámetros del método espectral
|   `-- figuras/                 Todas las figuras del informe
|-- metadatos/configuraciones.json   Semillas, parámetros, coeficientes y especificaciones
```

## Cómo reproducir

Requisitos: Python 3.10+ con `numpy`, `scipy`, `matplotlib`, `pandas`, `nbformat`, `nbclient`,
`ipykernel`, `nbconvert`, `pytest`.

```bash
pip install numpy scipy matplotlib pandas nbformat nbclient nbconvert ipykernel pytest
# 1) Pruebas automáticas (36 pruebas)
python -m pytest tests -q
# 2) Regenerar todo (señales, WAV, figuras, métricas, JSON)
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace
```

El notebook debe ejecutarse desde la carpeta raíz (la que contiene `src/`). No depende de ningún
archivo externo: las señales se generan desde las semillas 101, 202 y 303.


## Casos de prueba

| Caso | Ruido | SNR objetivo | Semilla |
|------|-------|--------------|---------|
| 1 | Interferencia senoidal de 3100 Hz (separada de la banda útil) | 5 dB | 101 |
| 2 | Ruido blanco gaussiano | 5 dB | 202 |
| 3 | Ruido de banda limitada 1800-2800 Hz (se superpone con el tono de 2000 Hz) | 5 dB | 303 |

Señal útil común: tonos de 500, 1200 y 2000 Hz (amplitudes 1, 0.7, 0.5; fases 0, 35 y -50 grados), fs = 8 kHz,
16384 muestras, pico máximo 0.8.

## Supuestos de esta entrega

* Se usó el generador del notebook de la guia (`CE1110_generador_tonos_defensa_Kaggle.ipynb`):
  sus funciones están en `src/senales.py` y `notebook.ipynb` las llama con su misma cadena y parámetros
  (`TIPO_SENAL`, `FRECUENCIAS_UTILES`, `SNR_OBJETIVO_DB`, `PICO_MAXIMO`, `FADE_MS`...). Solo se cambió la
  configuración: `FS`=8 kHz, 16384 muestras, semillas 101/202/303 y `TIPO_RUIDO` = `tonal`/`blanco`/`banda`.
* Para las secciones de memoria, tiempo y bibliotecas se tomó como **plataforma de referencia un STM32F407** (Cortex-M4F, 168 MHz, ADC/DAC de 12 bits,
  CMSIS-DSP). Los tiempos del microcontrolador son **estimaciones por conteo de operaciones**, no
  mediciones; se deben validar con el contador de ciclos del hardware real.


