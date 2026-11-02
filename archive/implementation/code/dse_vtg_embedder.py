"""Load Qwen3-VL-Embedding and score query↔frame similarity."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

EMBED_MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"


class DSEEmbedder:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._backend = None
        self._model = None
        self._processor = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBED_MODEL_ID, device=self.device, trust_remote_code=True)
            self._backend = "sentence_transformers"
            return
        except Exception as exc:
            st_err = exc

        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(EMBED_MODEL_ID, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                EMBED_MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self._model.eval()
            self._backend = "transformers"
            return
        except Exception as exc:
            raise RuntimeError(f"could not load {EMBED_MODEL_ID}: ST={st_err}; transformers={exc}") from exc

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if self._backend == "sentence_transformers":
            vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [vectors[i] for i in range(len(texts))]
        import torch

        out = []
        with torch.inference_mode():
            for text in texts:
                inputs = self._processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                vec = self._model(**inputs)
                if hasattr(vec, "pooler_output") and vec.pooler_output is not None:
                    arr = vec.pooler_output[0].float().cpu().numpy()
                elif hasattr(vec, "last_hidden_state"):
                    arr = vec.last_hidden_state[0].mean(dim=0).float().cpu().numpy()
                else:
                    raise RuntimeError("unexpected embedding model output")
                out.append(arr)
        return out

    def embed_frames(self, frames: np.ndarray, *, batch_size: int = 4) -> list[np.ndarray]:
        """frames: (T, H, W, C) uint8 RGB."""

        images = [Image.fromarray(frame) for frame in frames]
        if self._backend == "sentence_transformers":
            vectors = self._model.encode(images, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
            return [vectors[i] for i in range(len(images))]
        import torch

        out: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                vec = self._model(**inputs)
                if hasattr(vec, "pooler_output") and vec.pooler_output is not None:
                    batch_vecs = vec.pooler_output.float().cpu().numpy()
                elif hasattr(vec, "last_hidden_state"):
                    batch_vecs = vec.last_hidden_state.mean(dim=1).float().cpu().numpy()
                else:
                    raise RuntimeError("unexpected embedding model output")
                for row in batch_vecs:
                    out.append(row)
        return out

    def query_frame_similarities(self, query: str, frames: np.ndarray) -> list[float]:
        query_vec = self.embed_texts([query])[0]
        frame_vecs = self.embed_frames(frames)
        return [self._cosine(query_vec, fv) for fv in frame_vecs]
