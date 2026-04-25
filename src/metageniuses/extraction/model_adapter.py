from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .schemas import ModelDescription


@dataclass(frozen=True)
class BatchExtraction:
    token_ids: list[list[int]]
    hidden_states_by_layer: dict[int, list[list[list[float]]]]


class ModelAdapter(ABC):
    @abstractmethod
    def describe(self) -> ModelDescription:
        raise NotImplementedError

    @abstractmethod
    def extract_batch(
        self,
        sequences: list[str],
        transformer_layers: list[int],
        max_length: int,
    ) -> BatchExtraction:
        raise NotImplementedError


class FakeModelAdapter(ModelAdapter):
    """Deterministic adapter for local tests without external model downloads."""

    def __init__(
        self,
        model_id: str = "fake/metagene-tiny",
        num_transformer_layers: int = 8,
        d_model: int = 16,
    ) -> None:
        self._desc = ModelDescription(
            model_id=model_id,
            tokenizer_id=f"{model_id}-tokenizer",
            revision=None,
            num_transformer_layers=num_transformer_layers,
            d_model=d_model,
        )
        self._tok = {
            "A": 10,
            "C": 11,
            "G": 12,
            "T": 13,
            "U": 14,
            "N": 15,
        }
        self._bos = 1
        self._unk = 2

    def describe(self) -> ModelDescription:
        return self._desc

    def _tokenize(self, seq: str, max_length: int) -> list[int]:
        token_ids = [self._bos]
        for ch in seq:
            token_ids.append(self._tok.get(ch, self._unk))
            if len(token_ids) >= max_length:
                break
        return token_ids

    def _vector(self, token_id: int, layer: int, token_index: int) -> list[float]:
        d_model = self._desc.d_model
        base = token_id + (layer * 17) + token_index
        return [((base + dim) % 257) / 257.0 for dim in range(d_model)]

    def extract_batch(
        self,
        sequences: list[str],
        transformer_layers: list[int],
        max_length: int,
    ) -> BatchExtraction:
        token_ids_batch = [self._tokenize(seq, max_length=max_length) for seq in sequences]
        hidden_states_by_layer: dict[int, list[list[list[float]]]] = {}
        for layer in transformer_layers:
            per_sequence: list[list[list[float]]] = []
            for seq_tokens in token_ids_batch:
                seq_vectors: list[list[float]] = []
                for token_index, token_id in enumerate(seq_tokens):
                    seq_vectors.append(self._vector(token_id, layer=layer, token_index=token_index))
                per_sequence.append(seq_vectors)
            hidden_states_by_layer[layer] = per_sequence
        return BatchExtraction(token_ids=token_ids_batch, hidden_states_by_layer=hidden_states_by_layer)


class TransformersModelAdapter(ModelAdapter):
    """Hugging Face adapter. Imports torch/transformers lazily."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._torch, self._transformers = self._import_dependencies()
        self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(
            cfg.tokenizer_id or cfg.model_id,
            revision=cfg.revision,
            local_files_only=cfg.local_files_only,
            trust_remote_code=cfg.trust_remote_code,
        )
        model_dtype = self._resolve_dtype(cfg.dtype)
        self._model = self._transformers.AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            local_files_only=cfg.local_files_only,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=model_dtype,
        )
        self._device = self._resolve_device(cfg.device)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._desc = ModelDescription(
            model_id=cfg.model_id,
            tokenizer_id=cfg.tokenizer_id or cfg.model_id,
            revision=cfg.revision,
            num_transformer_layers=int(getattr(self._model.config, "num_hidden_layers")),
            d_model=int(getattr(self._model.config, "hidden_size")),
        )

    def _import_dependencies(self) -> tuple[Any, Any]:
        try:
            import torch
            import transformers
        except Exception as exc:  # pragma: no cover - environment-specific.
            raise RuntimeError(
                "TransformersModelAdapter requires torch and transformers. "
                "Install them first or use FakeModelAdapter for tests."
            ) from exc
        return torch, transformers

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_dtype(self, requested: str):
        if requested == "auto":
            if self._torch.cuda.is_available():
                return self._torch.bfloat16
            return self._torch.float32
        requested = requested.lower()
        if requested in {"bf16", "bfloat16"}:
            return self._torch.bfloat16
        if requested in {"fp16", "float16"}:
            return self._torch.float16
        if requested in {"fp32", "float32"}:
            return self._torch.float32
        raise ValueError(f"Unsupported dtype value: {requested}")

    def describe(self) -> ModelDescription:
        return self._desc

    def extract_batch(
        self,
        sequences: list[str],
        transformer_layers: list[int],
        max_length: int,
    ) -> BatchExtraction:
        encoded = self._tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with self._torch.no_grad():
            outputs = self._model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        input_ids = encoded["input_ids"].detach().cpu()
        attention_mask = encoded["attention_mask"].detach().cpu()

        token_ids_batch: list[list[int]] = []
        for batch_idx in range(input_ids.shape[0]):
            valid_tokens = int(attention_mask[batch_idx].sum().item())
            token_ids_batch.append(input_ids[batch_idx, :valid_tokens].tolist())

        hidden_states_by_layer: dict[int, list[list[list[float]]]] = {}
        for layer in transformer_layers:
            layer_tensor = outputs.hidden_states[layer].detach().cpu()
            per_sequence: list[list[list[float]]] = []
            for batch_idx in range(layer_tensor.shape[0]):
                valid_tokens = len(token_ids_batch[batch_idx])
                per_sequence.append(layer_tensor[batch_idx, :valid_tokens, :].tolist())
            hidden_states_by_layer[layer] = per_sequence

        return BatchExtraction(
            token_ids=token_ids_batch,
            hidden_states_by_layer=hidden_states_by_layer,
        )

