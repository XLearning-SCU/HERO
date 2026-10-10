"""megatron.training.tokenizer shim.

Newer Megatron-LM (26.04+) removed ``megatron.training.tokenizer`` entirely
(Megatron's own tokenizer was deprecated in favour of HF/tiktoken backends).
Slime v0.3.0 still imports ``_vocab_size_with_padding`` from it for vocab
padding computation, so this shim restores the minimum surface slime uses.
This is a verbatim re-host of the historical Megatron implementation — no new
behaviour — to keep slime imports working against a current Megatron checkout.
"""
from .tokenizer import _vocab_size_with_padding  # noqa: F401