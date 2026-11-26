"""Predicciones anidadas de un modelo del zoo para usarlas como entrada de otro (stacking sin fuga).

Para cada fold externo k: las filas de entrenamiento (folds j ≠ k) reciben la predicción de un ajuste
sin {j, k}; las de prueba (fold k), la de un ajuste sin k. Guarda preds/anid_<modelo>.npy con forma
(5, n), en residuo de la regla / 100.

Uso: python anidado.py cat_huber
"""
import itertools, pickle, sys, time
import numpy as np
from comun import AQUI, cargar
from zoo import MODELOS, variables
nombre = sys.argv[1]
hacer = MODELOS[nombre][0]
c = cargar(); e = c["e"]; S = pickle.load(open(AQUI / "seq.pkl", "rb"))
X = variables(c, S); y = (e["z1"] - c["regla"]).to_numpy()
dev, fold = ~c["caja"], c["fold"]
t0 = time.time()
A = np.zeros((5, len(e)), np.float32)
for k in range(5):
    tr, te = dev & (fold != k), dev & (fold == k)
    A[k, te] = hacer().fit(X[tr], y[tr]).predict(X[te]) / 100
for j, k in itertools.combinations(range(5), 2):
    tr = dev & (fold != j) & (fold != k)
    m = hacer().fit(X[tr], y[tr])
    A[k, fold == j] = m.predict(X[fold == j]) / 100
    A[j, fold == k] = m.predict(X[fold == k]) / 100
A[:, ~dev] = 0
np.save(AQUI / "preds" / f"anid_{nombre}.npy", A)
print(f"listo ({time.time() - t0:.0f} s)")
