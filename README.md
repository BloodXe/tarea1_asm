# CE 1110 - Tarea 1: caracterización espectral y prototipado de filtros para reducción de ruido
Enlance al repo:
https://github.com/BloodXe/tarea1_asm.git


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

El notebook debe ejecutarse desde la carpeta raíz (la que contiene `src/`).


