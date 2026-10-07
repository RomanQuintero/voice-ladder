"""Mide todos los audios generados y las grabaciones reales.

    python scripts/medir.py          (entorno .venv-metricas)

Es reanudable: guarda cada fila en resultados/metricas.tsv y salta lo ya
medido. Genera también resultados/base_real.tsv con las referencias de
techo (el lector real frente a sí mismo) y suelo (frente a otros lectores).
"""

import csv
from collections import defaultdict

import numpy as np

import comun
from metricas import Hablante, Transcriptor, cargar_audio, dnsmos, wer

COLS = ["modelo", "variante", "lector", "tramo", "id", "conjunto",
        "sim_real", "sim_lector", "wer", "wer_real", "dnsmos_p808",
        "dnsmos_ovrl", "transcripcion"]


def generados(incluir_prueba):
    for path in sorted(comun.SALIDAS.glob("*/*/*/*/*.wav")):
        modelo, variante, lector, tramo = path.parts[-5:-1]
        if modelo == "prueba" and not incluir_prueba:
            continue
        yield modelo, variante, lector, int(tramo.rstrip("s")), path.stem, path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--incluir-prueba", action="store_true",
                    help="medir también salidas/prueba (modelo simulado)")
    args = ap.parse_args()

    comun.RESULTADOS.mkdir(exist_ok=True)
    hablante = Hablante()
    asr = Transcriptor()

    # 1. Grabaciones reales: embeddings, WER y DNSMOS de cada frase de prueba.
    reales = comun.frases("prueba")
    textos = {(f["lector"], f["id"]): f["texto"] for f in comun.frases("ambas")}
    print(f"Midiendo {len(reales)} grabaciones reales...")
    emb_real = dict(zip([(f["lector"], f["id"]) for f in reales],
                        hablante.embeddings([cargar_audio(f["real"]) for f in reales])))
    por_lector = defaultdict(list)
    for (spk, fid), e in emb_real.items():
        por_lector[spk].append((fid, e))

    def centroide(spk, sin=None):
        v = np.mean([e for fid, e in por_lector[spk] if fid != sin], axis=0)
        return v / np.linalg.norm(v)

    base_path = comun.RESULTADOS / "base_real.tsv"
    wer_real = {}
    if base_path.exists():
        for r in comun.leer_tsv(base_path):
            wer_real[(r["lector"], r["id"])] = float(r["wer_real"])
    else:
        with base_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["lector", "id", "sim_mismo_lector", "sim_otro_lector_max",
                        "wer_real", "dnsmos_p808", "transcripcion"])
            for fr in reales:
                k = (fr["lector"], fr["id"])
                x = cargar_audio(fr["real"])
                texto = asr(x)
                wer_real[k] = wer(fr["texto"], texto)
                otros = [float(emb_real[k] @ centroide(s)) for s in por_lector
                         if s != fr["lector"]]
                w.writerow([*k, f"{emb_real[k] @ centroide(fr['lector'], fr['id']):.4f}",
                            f"{max(otros):.4f}" if otros else "", f"{wer_real[k]:.4f}",
                            f"{dnsmos(x)[0]:.3f}", texto])

    # 2. Audios generados.
    out_path = comun.RESULTADOS / "metricas.tsv"
    hechos = set()
    if out_path.exists():
        hechos = {(r["modelo"], r["variante"], r["lector"], r["tramo"], r["id"])
                  for r in comun.leer_tsv(out_path)}
    pendientes = [g for g in generados(args.incluir_prueba)
                  if tuple(map(str, g[:5])) not in hechos]
    print(f"Midiendo {len(pendientes)} audios generados...")
    nuevo = not out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if nuevo:
            w.writerow(COLS)
        for i, (modelo, variante, spk, tramo, fid, path) in enumerate(pendientes, 1):
            x = cargar_audio(path)
            e = hablante.embeddings([x])[0]
            k = (spk, fid)
            con_real = k in emb_real
            texto = asr(x)
            p808, ovrl = dnsmos(x)
            w.writerow([
                modelo, variante, spk, tramo, fid,
                "prueba" if con_real else "extra",
                f"{e @ emb_real[k]:.4f}" if con_real else "",
                f"{e @ centroide(spk, fid if con_real else None):.4f}",
                f"{wer(textos[k], texto):.4f}",
                f"{wer_real[k]:.4f}" if con_real else "",
                f"{p808:.3f}", f"{ovrl:.3f}", texto])
            f.flush()
            if i % 50 == 0 or i == len(pendientes):
                print(f"  {i}/{len(pendientes)}")


if __name__ == "__main__":
    main()
