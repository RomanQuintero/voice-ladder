"""Descarga el split de test de Multilingual LibriSpeech (español) y prepara
la escalera de referencias y el set de prueba para el #19.

Uso (desde la carpeta 19):
    pip install datasets soundfile numpy
    python scripts/preparar_dataset.py

Genera:
    datos/mls_raw/<lector>/<id>.flac     audios originales (16 kHz)
    datos/mls_raw/metadata.tsv           id, lector, capítulo, duración, texto
    datos/referencias/<lector>/ref_XXXs.wav + .txt
    datos/prueba/<lector>/<id>.wav       grabación real (verdad de referencia)
    datos/prueba.tsv                     lector, id, duración, texto
    datos/lectores.tsv                   lectores elegidos (sexo a rellenar)
"""

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
LADDER_S = [3, 10, 30, 60, 300]
FADE_S = 0.03


def descargar(raw_dir: Path, split: str) -> list[dict]:
    from datasets import Audio, load_dataset

    meta_path = raw_dir / "metadata.tsv"
    if meta_path.exists():
        print(f"Usando descarga previa en {raw_dir}")
        with meta_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for r in rows:
            r["duracion"] = float(r["duracion"])
        return rows

    ds = load_dataset("facebook/multilingual_librispeech", "spanish",
                      split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    for i, ex in enumerate(ds):
        spk = str(ex["speaker_id"])
        out = raw_dir / spk / f"{ex['id']}.flac"
        out.parent.mkdir(parents=True, exist_ok=True)
        data, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            raise SystemExit(f"Frecuencia inesperada {sr} Hz en {ex['id']}")
        sf.write(out, data, SR, format="FLAC")
        rows.append({
            "id": ex["id"], "lector": spk, "capitulo": str(ex["chapter_id"]),
            "duracion": len(data) / SR, "texto": ex["transcript"].strip(),
        })
        if i % 200 == 0:
            print(f"  {i} frases descargadas...")

    with meta_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["id", "lector", "capitulo", "duracion", "texto"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Descargadas {len(rows)} frases en {raw_dir}")
    return rows


def leer(raw_dir: Path, r: dict) -> np.ndarray:
    data, _ = sf.read(raw_dir / r["lector"] / f"{r['id']}.flac", dtype="float32")
    return data


def recortar(audio: np.ndarray, segundos: float) -> np.ndarray:
    out = audio[: int(segundos * SR)].copy()
    n = int(FADE_S * SR)
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def preparar_lector(raw_dir: Path, out_dir: Path, spk: str, frases: list[dict],
                    n_prueba: int) -> list[dict]:
    # Prueba y referencia salen de capítulos distintos siempre que se pueda,
    # para que el modelo no vea el mismo contexto que va a imitar.
    por_cap = defaultdict(list)
    for r in sorted(frases, key=lambda r: r["id"]):
        por_cap[r["capitulo"]].append(r)
    caps = sorted(por_cap, key=lambda c: -sum(r["duracion"] for r in por_cap[c]))

    if len(caps) > 1:
        # Prueba desde los capítulos más cortos; referencia del resto.
        prueba = []
        for c in reversed(caps):
            prueba.extend(por_cap[c][: n_prueba - len(prueba)])
            if len(prueba) >= n_prueba:
                break
        ids = {r["id"] for r in prueba}
        referencia = [r for c in caps for r in por_cap[c] if r["id"] not in ids]
    else:
        frases_cap = por_cap[caps[0]]
        prueba, referencia = frases_cap[-n_prueba:], frases_cap[:-n_prueba]

    # Referencia: frases consecutivas concatenadas con 0,3 s de silencio.
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)
    piezas, textos, total = [], [], 0.0
    for r in referencia:
        piezas += [leer(raw_dir, r), gap]
        textos.append(r["texto"])
        total += r["duracion"] + 0.3
        if total >= max(LADDER_S) + 5:
            break
    if total < max(LADDER_S):
        raise SystemExit(f"Lector {spk}: solo {total:.0f} s de referencia")
    continuo = np.concatenate(piezas)

    ref_dir = out_dir / "referencias" / spk
    ref_dir.mkdir(parents=True, exist_ok=True)
    for s in LADDER_S:
        sf.write(ref_dir / f"ref_{s:03d}s.wav", recortar(continuo, s), SR,
                 subtype="PCM_16")
    # Texto completo de la referencia; los modelos que necesitan la
    # transcripción exacta de un tramo recortado la sacan con Whisper.
    (ref_dir / "referencia_completa.txt").write_text(" ".join(textos),
                                                      encoding="utf-8")

    p_dir = out_dir / "prueba" / spk
    p_dir.mkdir(parents=True, exist_ok=True)
    for r in prueba:
        sf.write(p_dir / f"{r['id']}.wav", leer(raw_dir, r), SR, subtype="PCM_16")
    return prueba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="datos")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lectores", type=int, default=4)
    ap.add_argument("--frases-prueba", type=int, default=20)
    ap.add_argument("--min-minutos", type=float, default=10)
    ap.add_argument("--excluir", nargs="*", default=[],
                    help="ids de lector a descartar (p. ej. para equilibrar sexos)")
    args = ap.parse_args()

    out_dir = Path(args.datos)
    raw_dir = out_dir / "mls_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = descargar(raw_dir, args.split)

    por_lector = defaultdict(list)
    for r in rows:
        por_lector[r["lector"]].append(r)
    candidatos = sorted(
        (s for s, fr in por_lector.items()
         if s not in args.excluir
         and sum(r["duracion"] for r in fr) >= args.min_minutos * 60),
        key=lambda s: -sum(r["duracion"] for r in por_lector[s]))
    print(f"{len(candidatos)} lectores con al menos {args.min_minutos:g} min")
    if len(candidatos) < args.lectores:
        raise SystemExit("No hay suficientes lectores; baja --min-minutos")
    elegidos = candidatos[: args.lectores]

    todas = []
    for spk in elegidos:
        for r in preparar_lector(raw_dir, out_dir, spk, por_lector[spk],
                                 args.frases_prueba):
            todas.append(r)

    with (out_dir / "prueba.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["lector", "id", "duracion", "texto"])
        for r in todas:
            w.writerow([r["lector"], r["id"], f"{r['duracion']:.2f}", r["texto"]])

    with (out_dir / "lectores.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["lector", "minutos", "sexo"])
        for spk in elegidos:
            mins = sum(r["duracion"] for r in por_lector[spk]) / 60
            w.writerow([spk, f"{mins:.1f}", ""])

    print("Listo. Lectores elegidos:", ", ".join(elegidos))
    print("MLS no trae el sexo del lector en Hugging Face: escucha un audio de "
          "cada uno y rellena la columna 'sexo' de datos/lectores.tsv.")


if __name__ == "__main__":
    main()
