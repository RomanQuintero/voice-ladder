"""Para cada tramo largo, elige la ventana de 10 s más representativa del
lector: la que tiene el embedding ECAPA más cercano a la media del tramo.

Así Chatterbox y F5-TTS, que solo usan ~10 s de referencia, pueden
aprovechar tener más audio disponible.

    python scripts/seleccionar_referencias.py     (entorno .venv-metricas)
"""

import csv

import numpy as np
import soundfile as sf

import comun
from metricas import SR, Hablante, cargar_audio


def main():
    hablante = Hablante()
    filas = []
    for spk in comun.lectores():
        for tramo in comun.TRAMOS_SEL:
            x = cargar_audio(comun.referencia(spk, tramo, "recorte"))
            n, paso = comun.VENTANA_SEL * SR, SR
            inicios = list(range(0, len(x) - n + 1, paso))
            emb = hablante.embeddings([x[i:i + n] for i in inicios])
            centro = emb.mean(axis=0)
            centro /= np.linalg.norm(centro)
            sims = emb @ centro
            mejor = int(np.argmax(sims))
            sf.write(comun.referencia(spk, tramo, "seleccion"),
                     x[inicios[mejor]:inicios[mejor] + n], SR, subtype="PCM_16")
            filas.append([spk, tramo, inicios[mejor] / SR, f"{sims[mejor]:.4f}",
                          f"{sims[0]:.4f}", len(inicios)])
            print(f"{spk} {tramo:>3}s: ventana en {inicios[mejor] / SR:.0f}s "
                  f"(sim {sims[mejor]:.3f}; los primeros 10 s: {sims[0]:.3f})")

    with (comun.REFS / "seleccion.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["lector", "tramo", "inicio_s", "sim_elegida",
                    "sim_primeros_10s", "ventanas"])
        w.writerows(filas)


if __name__ == "__main__":
    main()
