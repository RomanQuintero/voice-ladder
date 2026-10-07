"""Agrega las métricas y dibuja las curvas de la escalera.

    python scripts/graficar.py       (entorno .venv-metricas)

Lee resultados/metricas.tsv, resultados/base_real.tsv y
salidas/<modelo>/tiempos.tsv. Escribe resultados/resumen.tsv,
resultados/resumen.md y las gráficas PNG.
"""

from collections import defaultdict

import numpy as np

import comun

# Umbral de la rodilla: primer tramo cuya similitud queda a menos de esto
# del mejor tramo de esa serie.
UMBRAL_RODILLA = 0.01

COLOR = {"xtts": "#2a78d6", "chatterbox": "#eb6834", "f5": "#1baf7a",
         "prueba": "#4a3aa7"}
NOMBRE = {"xtts": "XTTS-v2", "chatterbox": "Chatterbox", "f5": "F5-TTS es",
          "prueba": "Simulado"}
ETIQ_TRAMO = {3: "3 s", 10: "10 s", 30: "30 s", 60: "1 min", 300: "5 min"}
TINTA, TINTA_2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def num(v):
    return float(v) if v not in ("", None) else np.nan


def etiqueta(serie):
    modelo, variante = serie
    extra = {"completa": "", "recorte": "", "seleccion": " (selección 10 s)"}
    return NOMBRE.get(modelo, modelo) + extra.get(variante, f" ({variante})")


def agregar():
    filas = comun.leer_tsv(comun.RESULTADOS / "metricas.tsv")
    grupos = defaultdict(list)
    for r in filas:
        grupos[(r["modelo"], r["variante"], int(r["tramo"]))].append(r)

    tiempos = defaultdict(list)
    for t in comun.SALIDAS.glob("*/tiempos.tsv"):
        for r in comun.leer_tsv(t):
            tiempos[(r["modelo"], r["variante"], int(r["tramo"]))].append(r)

    resumen = []
    for (modelo, variante, tramo), rs in sorted(grupos.items()):
        prueba = [r for r in rs if r["conjunto"] == "prueba"]
        sim_l = np.array([num(r["sim_lector"]) for r in rs])
        # Consistencia: desviación entre frases dentro de cada lector, promediada.
        por_lector = defaultdict(list)
        for r in rs:
            por_lector[r["lector"]].append(num(r["sim_lector"]))
        tt = tiempos.get((modelo, variante, tramo), [])
        resumen.append({
            "modelo": modelo, "variante": variante, "tramo": tramo, "n": len(rs),
            "sim_lector": np.nanmean(sim_l),
            "sim_real": np.nanmean([num(r["sim_real"]) for r in prueba]) if prueba else np.nan,
            "consistencia_sd": np.mean([np.nanstd(v) for v in por_lector.values()]),
            "wer": np.nanmean([num(r["wer"]) for r in rs]),
            "delta_wer": np.nanmean([num(r["wer"]) - num(r["wer_real"]) for r in prueba])
            if prueba else np.nan,
            "dnsmos": np.nanmean([num(r["dnsmos_p808"]) for r in rs]),
            "rtf": np.nanmean([num(r["rtf"]) for r in tt]) if tt else np.nan,
            "vram_gb": np.nanmax([num(r["vram_gb"]) for r in tt]) if tt else np.nan,
        })
    return resumen


def base_real():
    b = comun.leer_tsv(comun.RESULTADOS / "base_real.tsv")
    return {
        "techo": np.mean([num(r["sim_mismo_lector"]) for r in b]),
        "techo_sd": np.std([num(r["sim_mismo_lector"]) for r in b]),
        "suelo": np.nanmean([num(r["sim_otro_lector_max"]) for r in b]),
        "wer_real": np.mean([num(r["wer_real"]) for r in b]),
        "dnsmos": np.mean([num(r["dnsmos_p808"]) for r in b]),
    }


def rodilla(puntos):
    """puntos: [(tramo, sim)] ordenados. Devuelve el tramo de la rodilla."""
    mejor = max(s for _, s in puntos)
    return next(t for t, s in puntos if s >= mejor - UMBRAL_RODILLA)


def series(resumen):
    out = defaultdict(list)
    for r in resumen:
        out[(r["modelo"], r["variante"])].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["tramo"])
    return dict(sorted(out.items()))


def estilo_ejes(ax, titulo, ylabel):
    import matplotlib.ticker as mt
    ax.set_xscale("log")
    ax.set_xticks(comun.TRAMOS)
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: ETIQ_TRAMO.get(int(round(v)), "")))
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_xlabel("Audio de referencia disponible", color=TINTA_2)
    ax.set_ylabel(ylabel, color=TINTA_2)
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=13, pad=12)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2)


def curva(ax, ss, clave):
    for serie, rs in ss.items():
        x = [r["tramo"] for r in rs]
        y = [r[clave] for r in rs]
        ax.plot(x, y, color=COLOR.get(serie[0], TINTA_2), linewidth=2,
                linestyle="--" if serie[1] == "seleccion" else "-",
                marker="o", markersize=6, markeredgecolor=FONDO,
                markeredgewidth=1.5, label=etiqueta(serie))


def graficas(resumen, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ss = series(resumen)
    out = comun.RESULTADOS
    defs = [
        ("similitud.png", "sim_lector", "Parecido con el lector real",
         "Similitud ECAPA con el lector (coseno)"),
        ("consistencia.png", "consistencia_sd", "Variación del parecido entre frases",
         "Desviación típica de la similitud (menos es mejor)"),
        ("wer.png", "delta_wer", "Inteligibilidad frente a la grabación real",
         "WER del clon − WER del original (menos es mejor)"),
        ("dnsmos.png", "dnsmos", "Calidad de audio estimada", "DNSMOS P.808"),
    ]
    for fichero, clave, titulo, ylabel in defs:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=FONDO)
        ax.set_facecolor(FONDO)
        if clave == "sim_lector":
            ax.axhspan(base["techo"] - base["techo_sd"], base["techo"] + base["techo_sd"],
                       color=REJILLA, alpha=0.6, linewidth=0)
            ax.axhline(base["techo"], color=TINTA_2, linewidth=1)
            ax.text(comun.TRAMOS[0], base["techo"], " Real frente a sí mismo",
                    va="bottom", color=TINTA_2, fontsize=9)
            ax.axhline(base["suelo"], color=TINTA_2, linewidth=1, linestyle=":")
            ax.text(comun.TRAMOS[0], base["suelo"], " Real frente a otro lector",
                    va="bottom", color=TINTA_2, fontsize=9)
        if clave == "delta_wer":
            ax.axhline(0, color=TINTA_2, linewidth=1)
        if clave == "dnsmos":
            ax.axhline(base["dnsmos"], color=TINTA_2, linewidth=1)
            ax.text(comun.TRAMOS[0], base["dnsmos"], " Grabación real",
                    va="bottom", color=TINTA_2, fontsize=9)
        curva(ax, ss, clave)
        estilo_ejes(ax, titulo, ylabel)
        ax.legend(frameon=False, labelcolor=TINTA_2, fontsize=9, handlelength=3.5,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        fig.savefig(out / fichero, dpi=160, facecolor=FONDO)
        plt.close(fig)

    # Coste: RTF medio por serie (no depende del tramo).
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.5 * len(ss)), facecolor=FONDO)
    ax.set_facecolor(FONDO)
    nombres = [etiqueta(s) for s in ss]
    rtf = [np.nanmean([r["rtf"] for r in rs]) for rs in ss.values()]
    ax.barh(nombres, rtf, color=[COLOR.get(s[0], TINTA_2) for s in ss], height=0.5)
    for i, v in enumerate(rtf):
        ax.text(v, i, f"  {v:.2f}", va="center", color=TINTA, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Tiempo de generación por segundo de audio (RTF, RTX 3090)",
                 loc="left", color=TINTA, fontsize=13, pad=12)
    ax.grid(axis="x", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=TINTA_2)
    fig.tight_layout()
    fig.savefig(out / "coste.png", dpi=160, facecolor=FONDO)
    plt.close(fig)


def escribir(resumen, base):
    cols = ["modelo", "variante", "tramo", "n", "sim_lector", "sim_real",
            "consistencia_sd", "wer", "delta_wer", "dnsmos", "rtf", "vram_gb"]
    with (comun.RESULTADOS / "resumen.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in resumen:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")

    ss = series(resumen)
    lineas = ["# Resultados #19", "",
              f"Techo (real frente a sí mismo): {base['techo']:.3f} · "
              f"suelo (frente a otro lector): {base['suelo']:.3f} · "
              f"WER de las grabaciones reales: {base['wer_real']:.3f}", "",
              "## Rodilla por serie", "",
              f"Primer tramo a menos de {UMBRAL_RODILLA} de la mejor similitud.", "",
              "| Serie | Rodilla | Similitud en la rodilla | Mejor similitud |",
              "|---|---|---|---|"]
    for serie, rs in ss.items():
        pts = [(r["tramo"], r["sim_lector"]) for r in rs]
        t = rodilla(pts)
        lineas.append(f"| {etiqueta(serie)} | {ETIQ_TRAMO[t]} | "
                      f"{dict(pts)[t]:.3f} | {max(s for _, s in pts):.3f} |")
    lineas += ["", "## Detalle", "",
               "| Serie | Tramo | Sim. lector | Sim. frase | SD | ΔWER | DNSMOS | RTF | VRAM GB |",
               "|---|---|---|---|---|---|---|---|---|"]
    for serie, rs in ss.items():
        for r in rs:
            lineas.append(
                f"| {etiqueta(serie)} | {ETIQ_TRAMO[r['tramo']]} | {r['sim_lector']:.3f} | "
                f"{r['sim_real']:.3f} | {r['consistencia_sd']:.3f} | {r['delta_wer']:+.3f} | "
                f"{r['dnsmos']:.2f} | {r['rtf']:.2f} | {r['vram_gb']:.1f} |")
    (comun.RESULTADOS / "resumen.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


def main():
    resumen = agregar()
    base = base_real()
    escribir(resumen, base)
    graficas(resumen, base)
    print(f"\nGráficas en {comun.RESULTADOS}")


if __name__ == "__main__":
    main()
