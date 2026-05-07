"""
Input / embedding perturbation methods (Somepalli et al., Wen et al.).
All operate purely on text tokens or embeddings — no model internals needed.
"""
import random
import torch
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Token-level (prompt-string) perturbations
# ---------------------------------------------------------------------------

def perturb_random_tokens(prompt: str, tokenizer, num_tokens: int = 4) -> str:
    """Insert random vocabulary tokens at random positions (RT)."""
    for _ in range(num_tokens):
        tok = tokenizer.decode([random.randint(1000, 40000)])
        pos = random.randint(0, len(prompt))
        prompt = prompt[:pos] + f" {tok} " + prompt[pos:]
    return prompt


def perturb_word_repetition(prompt: str, num_repeats: int = 10) -> str:
    """Repeat random words within the prompt (CWR)."""
    words = prompt.split()
    if not words:
        return prompt
    for _ in range(num_repeats):
        word = random.choice(words)
        words.insert(random.randint(0, len(words)), word)
    return " ".join(words)


def perturb_random_numbers(prompt: str, num_numbers: int = 10) -> str:
    """Insert random integers into the prompt (RNA)."""
    for _ in range(num_numbers):
        pos = random.randint(0, len(prompt))
        prompt = prompt[:pos] + f" {random.randint(0, 1_000_000)} " + prompt[pos:]
    return prompt


def perturb_tokenwise(prompt: str, tokenizer, token_idx: int) -> str:
    """Replace the k-th meaningful token with a random one."""
    enc = tokenizer(prompt, truncation=True, max_length=77,
                    padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].squeeze()
    mask = enc["attention_mask"].squeeze()
    meaningful = (mask == 1).nonzero(as_tuple=True)[0][1:-1]  # skip BOS/EOS
    if token_idx >= len(meaningful):
        return prompt
    ids[meaningful[token_idx]] = random.randint(1000, 40000)
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Embedding-level perturbations
# ---------------------------------------------------------------------------

def add_gaussian_noise(embedding: torch.Tensor, std: float = 0.5) -> torch.Tensor:
    """Add isotropic Gaussian noise to a text embedding (GNI)."""
    return embedding + torch.randn_like(embedding) * std


def optimize_embedding_wen(embedding: torch.Tensor, model_adapter,
                            steps: int = 20, lr: float = 0.01) -> torch.Tensor:
    """
    Metric-aware embedding perturbation (Wen et al.).
    Minimises the noise-difference norm via gradient descent on the embedding.
    Requires model_adapter.apply_model and model_adapter.make_conditioning.
    """
    emb = embedding.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=lr)
    device = model_adapter.device

    for _ in range(steps):
        opt.zero_grad()
        c = model_adapter.make_conditioning(emb)
        uc = model_adapter.make_conditioning(
            model_adapter.null_embedding(emb.shape[0]))
        x = model_adapter.randn_latent(emb.shape[0])
        t = torch.tensor([999] * emb.shape[0], device=device)
        s_cond = model_adapter.apply_model(x, t, c)
        s_uncond = model_adapter.apply_model(x, t, uc)
        loss = (s_cond - s_uncond).norm(p=2)
        loss.backward()
        opt.step()

    return emb.detach()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_token_perturb_fn(method: str, tokenizer=None) -> Optional[Callable]:
    """Return a (prompt, seed) -> prompt function for the given method name."""
    if method == "tokenwise":
        assert tokenizer is not None
        return lambda p, seed: perturb_tokenwise(p, tokenizer, seed)
    if method == "rt":
        assert tokenizer is not None
        return lambda p, seed: perturb_random_tokens(p, tokenizer)
    if method == "cwr":
        return lambda p, seed: perturb_word_repetition(p)
    if method == "rna":
        return lambda p, seed: perturb_random_numbers(p)
    return None


def get_embed_perturb_fn(method: str, model_adapter=None) -> Optional[Callable]:
    """Return an (embedding, **kw) -> embedding function."""
    if method == "gni":
        return lambda emb, **kw: add_gaussian_noise(emb)
    if method == "wen":
        assert model_adapter is not None
        return lambda emb, **kw: optimize_embedding_wen(emb, model_adapter)
    return None
