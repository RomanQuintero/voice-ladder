"""Modelos de medida (entorno .venv-metricas): ECAPA, Whisper y DNSMOS."""

import re
import unicodedata

import numpy as np

SR = 16000


def cargar_audio(path) -> np.ndarray:
    import librosa
    x, _ = librosa.load(str(path), sr=SR, mono=True)
    return x.astype(np.float32)


def _dispositivo():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Hablante:
    """Embeddings de hablante ECAPA-TDNN (speechbrain, VoxCeleb)."""

    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        self.dev = _dispositivo()
        # COPY evita los enlaces simbólicos, que en Windows piden permisos.
        self.modelo = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="modelos/ecapa", run_opts={"device": self.dev},
            local_strategy=LocalStrategy.COPY)

    def embeddings(self, audios: list[np.ndarray]) -> np.ndarray:
        import torch
        out = []
        for x in audios:
            with torch.no_grad():
                e = self.modelo.encode_batch(torch.from_numpy(x)[None].to(self.dev))
            out.append(e.squeeze().cpu().numpy())
        out = np.stack(out)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class Transcriptor:
    """Whisper large-v3-turbo en español vía transformers."""

    def __init__(self, modelo="openai/whisper-large-v3-turbo"):
        import torch
        from transformers import pipeline
        dev = _dispositivo()
        tipo = torch.float16 if dev == "cuda" else torch.float32
        try:
            self.pipe = pipeline("automatic-speech-recognition", model=modelo,
                                 dtype=tipo, device=dev)
        except TypeError:  # transformers < 4.56
            self.pipe = pipeline("automatic-speech-recognition", model=modelo,
                                 torch_dtype=tipo, device=dev)

    def __call__(self, x: np.ndarray) -> str:
        r = self.pipe({"raw": x, "sampling_rate": SR}, chunk_length_s=30,
                      generate_kwargs={"language": "spanish", "task": "transcribe"})
        return r["text"].strip()


def dnsmos(x: np.ndarray) -> tuple[float, float]:
    """(P.808 MOS, OVRL). DNSMOS está pensado para ruido y distorsión, no para
    prosodia: sirve para detectar audio roto, no para juzgar naturalidad fina."""
    from speechmos import dnsmos as d
    x = np.clip(x, -1, 1)
    r = d.run(x, SR)
    return float(r["p808_mos"]), float(r["ovrl_mos"])


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKC", texto).lower()
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", t)
    t = t.replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def wer(referencia: str, hipotesis: str) -> float:
    import jiwer
    ref, hip = normalizar(referencia), normalizar(hipotesis)
    if not ref:
        return float("nan")
    return float(jiwer.wer(ref, hip or "<vacio>"))
