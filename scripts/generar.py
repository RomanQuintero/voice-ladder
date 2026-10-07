"""Genera las frases clonadas de un modelo para todos los lectores y tramos.

Cada modelo vive en su propio entorno virtual (sus dependencias chocan):
    python scripts/generar.py --modelo xtts
    python scripts/generar.py --modelo chatterbox
    python scripts/generar.py --modelo f5

Es reanudable: salta los audios que ya existen. Los tiempos y la VRAM se
añaden a salidas/<modelo>/tiempos.tsv.

--modelo prueba no carga ningún modelo: copia la referencia con ruido para
comprobar rutas y el resto de la cadena sin GPU.
"""

import argparse
import csv
import re
import time
import zlib

import numpy as np
import soundfile as sf

import comun


class Prueba:
    sr = 16000

    def generar(self, texto, ref, tramo):
        x, _ = sf.read(ref, dtype="float32")
        rng = np.random.default_rng(len(texto))
        n = min(len(x), int(2 + len(texto) * 0.06) * self.sr)
        return x[:n] + rng.normal(0, 0.01, n).astype(np.float32)


class XTTS:
    def __init__(self):
        from TTS.api import TTS
        self.modelo = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        self.modelo = self.modelo.synthesizer.tts_model
        self.sr = 24000
        self._cache = {}

    def generar(self, texto, ref, tramo):
        if ref not in self._cache:
            # gpt_cond_len=-1: usa toda la referencia en trozos de 6 s y los
            # promedia, así que los tramos largos sí aportan información.
            self._cache.clear()
            self._cache[ref] = self.modelo.get_conditioning_latents(
                audio_path=[str(ref)], max_ref_length=tramo + 1,
                gpt_cond_len=-1, gpt_cond_chunk_len=6)
        latentes, embedding = self._cache[ref]
        out = self.modelo.inference(texto, "es", latentes, embedding,
                                    enable_text_splitting=True)
        return np.asarray(out["wav"], dtype=np.float32)


class Chatterbox:
    def __init__(self):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        self.modelo = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        self.sr = self.modelo.sr
        self._ref = None

    def generar(self, texto, ref, tramo):
        if ref != self._ref:
            self.modelo.prepare_conditionals(str(ref))
            self._ref = ref
        wav = self.modelo.generate(texto, language_id="es")
        return wav.squeeze(0).cpu().numpy().astype(np.float32)


class F5:
    REPO = "jpgallegoar/F5-Spanish"

    def __init__(self, config):
        from f5_tts.api import F5TTS
        from huggingface_hub import hf_hub_download, list_repo_files

        ficheros = list_repo_files(self.REPO)
        ckpts = [f for f in ficheros if f.endswith((".safetensors", ".pt"))]
        if not ckpts or "vocab.txt" not in [f.split("/")[-1] for f in ficheros]:
            raise SystemExit(f"No encuentro checkpoint y vocab.txt en {self.REPO}: "
                             f"{ficheros}")
        paso = lambda f: int((re.findall(r"\d+", f.split("/")[-1]) or ["0"])[-1])
        ckpt = max(ckpts, key=lambda f: (f.endswith(".safetensors"), paso(f)))
        vocab = next(f for f in ficheros if f.split("/")[-1] == "vocab.txt")
        print(f"F5-Spanish: {ckpt} + {vocab} (config {config})")
        self.modelo = F5TTS(model=config,
                            ckpt_file=hf_hub_download(self.REPO, ckpt),
                            vocab_file=hf_hub_download(self.REPO, vocab))
        self.sr = self.modelo.target_sample_rate

        # f5_tts transcribe la referencia pasándole a Whisper la ruta del
        # fichero, y transformers necesita ffmpeg para abrirla. Le pasamos el
        # audio ya leído para no depender de ffmpeg en Windows.
        import f5_tts.infer.utils_infer as ui
        original = ui.transcribe

        def transcribir(ref_audio, language=None):
            x, sr = sf.read(ref_audio, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            return original({"raw": x, "sampling_rate": sr}, language or "es")

        ui.transcribe = transcribir

        # Velocidad de habla de cada lector (caracteres por segundo) sacada de
        # sus grabaciones de MLS, para fijar la duración de lo generado.
        tot = {}
        for r in comun.leer_tsv(comun.DATOS / "mls_raw" / "metadata.tsv"):
            c, d = tot.get(r["lector"], (0, 0.0))
            tot[r["lector"]] = (c + len(r["texto"]), d + float(r["duracion"]))
        self.cps = {k: c / d for k, (c, d) in tot.items()}
        self.ui = ui
        self._refs = {}

    def generar(self, texto, ref, tramo):
        # F5 recorta la referencia a 12 s y la transcribe con Whisper. Si la
        # transcripción sale corta, su estimación de duración (proporcional a
        # caracteres de referencia) se dispara y revienta el modelo. Por eso
        # la duración se fija con la velocidad real del lector.
        if ref not in self._refs:
            ref_proc, ref_text = self.ui.preprocess_ref_audio_text(
                str(ref), "", show_info=lambda *a: None)
            self._refs[ref] = (ref_proc, ref_text, sf.info(ref_proc).duration)
            with (comun.SALIDAS / "f5" / "ref_textos.tsv").open("a", encoding="utf-8") as f:
                f.write(f"{ref}\t{self._refs[ref][2]:.2f}\t{ref_text}\n")
        ref_proc, ref_text, ref_s = self._refs[ref]
        gen_s = len(texto) / self.cps[ref.parent.name]
        wav, _, _ = self.modelo.infer(ref_proc, ref_text, texto,
                                      show_info=lambda *a: None, progress=None,
                                      fix_duration=ref_s + gen_s, seed=0)
        return np.asarray(wav, dtype=np.float32)


def cargar(nombre, args):
    if nombre == "prueba":
        return Prueba()
    if nombre == "xtts":
        return XTTS()
    if nombre == "chatterbox":
        return Chatterbox()
    if nombre == "f5":
        return F5(args.f5_config)
    raise SystemExit(f"Modelo desconocido: {nombre}")


def vram_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return True
    except ImportError:
        pass
    return False


def vram_pico():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 2**30
    except ImportError:
        pass
    return float("nan")


def semilla(n):
    np.random.seed(n)
    try:
        import torch
        torch.manual_seed(n)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True,
                    choices=["xtts", "chatterbox", "f5", "prueba"])
    ap.add_argument("--frases", default="prueba", choices=["prueba", "extra", "ambas"])
    ap.add_argument("--lectores", nargs="*", help="solo estos lectores")
    ap.add_argument("--limite", type=int, help="máx. frases por lector (para probar)")
    ap.add_argument("--f5-config", default="F5TTS_Base")
    args = ap.parse_args()

    lectores = args.lectores or comun.lectores()
    todas = [f for f in comun.frases(args.frases) if f["lector"] in lectores]
    if args.limite:
        cuenta, recortadas = {}, []
        for f in todas:
            cuenta[f["lector"]] = cuenta.get(f["lector"], 0) + 1
            if cuenta[f["lector"]] <= args.limite:
                recortadas.append(f)
        todas = recortadas

    trabajos = []
    for variante, tramo in comun.variantes(args.modelo):
        for spk in lectores:
            ref = comun.referencia(spk, tramo, variante)
            if not ref.exists():
                if variante == "seleccion":
                    raise SystemExit(f"Falta {ref}: ejecuta antes "
                                     "scripts/seleccionar_referencias.py")
                raise SystemExit(f"Falta {ref}")
            for f in todas:
                if f["lector"] != spk:
                    continue
                out = comun.salida(args.modelo, variante, spk, tramo, f["id"])
                if not out.exists():
                    trabajos.append((variante, tramo, spk, ref, f, out))

    print(f"{args.modelo}: {len(trabajos)} audios pendientes")
    if not trabajos:
        return
    modelo = cargar(args.modelo, args)

    tiempos = comun.SALIDAS / args.modelo / "tiempos.tsv"
    tiempos.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not tiempos.exists()
    with tiempos.open("a", encoding="utf-8", newline="") as ft:
        w = csv.writer(ft, delimiter="\t")
        if nuevo:
            w.writerow(["modelo", "variante", "lector", "tramo", "id",
                        "segundos_gen", "segundos_audio", "rtf", "vram_gb"])
        inicio = time.time()
        for i, (variante, tramo, spk, ref, f, out) in enumerate(trabajos, 1):
            semilla(zlib.crc32(f"{spk}/{f['id']}".encode()))
            vram_reset()
            t0 = time.perf_counter()
            wav = modelo.generar(f["texto"], ref, tramo)
            dt = time.perf_counter() - t0
            dur = len(wav) / modelo.sr
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.clip(wav, -1, 1), modelo.sr, subtype="PCM_16")
            w.writerow([args.modelo, variante, spk, tramo, f["id"], f"{dt:.3f}",
                        f"{dur:.3f}", f"{dt / max(dur, 1e-6):.3f}",
                        f"{vram_pico():.2f}"])
            ft.flush()
            if i % 20 == 0 or i == len(trabajos):
                eta = (time.time() - inicio) / i * (len(trabajos) - i)
                print(f"  {i}/{len(trabajos)}  quedan ~{eta / 60:.0f} min")


if __name__ == "__main__":
    main()
