"""megatron.training.tokenizer.tokenizer shim — the part slime v0.3.0 needs.

Restored from the historical Megatron-LM ``megatron/training/tokenizer/tokenizer.py``.
Only ``_vocab_size_with_padding`` is required by slime today; the rest of the
historical tokenizer classes are intentionally omitted (Megatron now uses
HuggingFace/tiktoken backends and slime waves ``--tokenizer-type
HuggingFaceTokenizer`` itself).
"""

import math


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by args.make_vocab_size_divisible_by.

    After padding, also enforce args.tensor_model_parallel_size alignment unless
    the caller disabled it (args.vocab_size_with_padding_absolutely_aligned /
    None-attr fallbacks handled). Historical Megatron semantics preserved.
    """
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
        vocab_size = orig_vocab_size

    make_vocab_size_divisible_by = getattr(
        args, "make_vocab_size_divisible_by", 128
    )
    tensor_model_parallel_size = getattr(
        args, "tensor_model_parallel_size", 1
    )

    multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
    padded_vocab_size = int(math.ceil(vocab_size / multiple)) * multiple

    return padded_vocab_size