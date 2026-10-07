"""Rutas y utilidades compartidas por los scripts del #19."""

import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
REFS = DATOS / "referencias"
PRUEBA = DATOS / "prueba"
SALIDAS = RAIZ / "salidas"
RESULTADOS = RAIZ / "resultados"

TRAMOS = [3, 10, 30, 60, 300]
# Chatterbox usa como mucho 10 s de referencia y F5-TTS recorta a 12 s, así
# que para tramos más largos se prueba también una ventana de 10 s elegida.
VENTANA_SEL = 10
TRAMOS_SEL = [t for t in TRAMOS if t > VENTANA_SEL]


def leer_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def lectores() -> list[str]:
    return [r["lector"] for r in leer_tsv(DATOS / "lectores.tsv")]


def frases(conjunto: str = "prueba") -> list[dict]:
    """Devuelve dicts con lector, id, texto y real (ruta o None)."""
    out = []
    if conjunto in ("prueba", "ambas"):
        for r in leer_tsv(DATOS / "prueba.tsv"):
            out.append({"lector": r["lector"], "id": r["id"], "texto": r["texto"],
                        "real": PRUEBA / r["lector"] / f"{r['id']}.wav"})
    if conjunto in ("extra", "ambas"):
        extra = leer_tsv(RAIZ / "evaluacion" / "frases_prueba.tsv")
        for spk in lectores():
            for r in extra:
                out.append({"lector": spk, "id": r["id"], "texto": r["texto"],
                            "real": None})
    return out


def referencia(lector: str, tramo: int, variante: str) -> Path:
    if variante == "seleccion":
        return REFS / lector / f"sel{VENTANA_SEL:02d}_de_{tramo:03d}s.wav"
    return REFS / lector / f"ref_{tramo:03d}s.wav"


def variantes(modelo: str) -> list[tuple[str, int]]:
    """Pares (variante, tramo) que se generan para cada modelo."""
    if modelo == "xtts":
        # XTTS trocea y promedia toda la referencia: no necesita selección.
        return [("completa", t) for t in TRAMOS]
    pares = [("recorte", t) for t in TRAMOS]
    pares += [("seleccion", t) for t in TRAMOS_SEL]
    return pares


def salida(modelo: str, variante: str, lector: str, tramo: int, fid: str) -> Path:
    return SALIDAS / modelo / variante / lector / f"{tramo:03d}s" / f"{fid}.wav"
