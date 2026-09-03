from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, n_embd = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores.masked_fill(
            self.causal_mask[:, :, :sequence_length, :sequence_length] == 0,
            float("-inf"),
        )
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(
            batch_size, sequence_length, n_embd
        )
        return self.dropout(self.projection(output))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x))
        return x + self.mlp(self.layer_norm_2(x))


class GPT(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.block_size = config["block_size"]
        self.token_embedding = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.position_embedding = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    config["n_embd"], config["n_head"], config["block_size"], config["dropout"]
                )
                for _ in range(config["n_layer"])
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config["n_embd"])
        self.language_model_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.language_model_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = input_ids.size(1)
        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.final_layer_norm(self.blocks(x))
        return self.language_model_head(x)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[1] / path).resolve()


class LocalModel:
    def __init__(self) -> None:
        self.checkpoint_path = _resolve_path(
            os.getenv("MODEL_PATH", "../model/final_model/step_10500.pt")
        )
        self.tokenizer_path = _resolve_path(os.getenv("TOKENIZER_PATH", "../model/tokenizer"))
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._lock = Lock()
        self._checkpoint: dict[str, Any] | None = None
        self.model: GPT | None = None
        self.tokenizer: ByteLevelBPETokenizer | None = None
        self.loaded = False
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            return
        vocab_file = self.tokenizer_path / "vocab.json"
        merges_file = self.tokenizer_path / "merges.txt"
        if not vocab_file.exists() or not merges_file.exists():
            return
        self._checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model = GPT(self._checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(self._checkpoint["model_state_dict"])
        self.model.eval()
        self.tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
        self.loaded = True

    def status(self) -> dict[str, Any]:
        config = self._checkpoint.get("model_config", {}) if self._checkpoint else {}
        return {
            "loaded": self.loaded,
            "checkpoint": str(self.checkpoint_path),
            "step": self._checkpoint.get("step") if self._checkpoint else None,
            "device": str(self.device),
            "vocab_size": config.get("vocab_size", 0),
            "block_size": config.get("block_size", 0),
            "inference_ready": self.loaded,
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        if not self.loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("The step 10500 checkpoint or tokenizer could not be loaded")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        encoded = self.tokenizer.encode(prompt)
        prompt_ids = encoded.ids[-self.model.block_size :]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        with self._lock, torch.inference_mode():
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.model.block_size :]
                logits = self.model(context)[:, -1, :] / temperature
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)

        generated = input_ids[0, len(prompt_ids) :].tolist()
        return self.tokenizer.decode(generated).strip()
