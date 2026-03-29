"""CLIP-based frame embedding for similarity, clustering, and search."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from fighter_iq.models import FrameEmbedding


class CLIPEmbedder:
    """Produces CLIP embeddings from PIL images.

    Implements the EmbeddingModel protocol. Uses open_clip for inference.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def load(self) -> None:
        """Lazy-load the CLIP model and preprocessing pipeline."""
        if self._model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(self._model_name, pretrained=self._pretrained)
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._model_name)

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    @property
    def embedding_dim(self) -> int:
        return 512

    def embed_frame(self, image: Image.Image) -> np.ndarray:
        """Return a normalized embedding vector for a single image."""
        self.load()
        import torch

        tensor = self._preprocess(image).unsqueeze(0)  # type: ignore[union-attr]
        with torch.no_grad():
            features = self._model.encode_image(tensor)  # type: ignore[union-attr]
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def embed_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Return normalized embedding vectors for a batch of images."""
        self.load()
        import torch

        tensors = torch.stack([self._preprocess(img) for img in images])  # type: ignore[union-attr]
        with torch.no_grad():
            features = self._model.encode_image(tensors)  # type: ignore[union-attr]
            features /= features.norm(dim=-1, keepdim=True)
        return [features[i].cpu().numpy() for i in range(len(images))]

    def make_embedding(self, timestamp: float, image: Image.Image, vector: np.ndarray) -> FrameEmbedding:
        """Wrap a raw vector into a FrameEmbedding with metadata."""
        img_hash = hashlib.md5(image.tobytes()[:4096]).hexdigest()[:12]
        return FrameEmbedding(timestamp=timestamp, embedding=vector, image_hash=img_hash)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
