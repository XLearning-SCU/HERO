# Trajectory Builders

Builders convert a captured `CompletionSession` into a `Trajectory` of trainable
`Trace`s — the first reconstruction step, before evaluation.

## Main files

- `base.py`: the builder contract (`async build(session) -> Trajectory`).
- `per_request.py`: one trace per completion.
- `prefix_merging.py`: stitch an append-only agent chain into one token-level trace.
- `record_utils.py`: helpers to pull messages, token ids, logprobs, and metadata
  out of a completion record.

## `per_request`

The simplest strategy: each completion becomes its own trace. Use it when you
want every request preserved independently rather than merged into longer
multi-turn examples.

## `prefix_merging`

A multi-turn agent resends the growing conversation on every step, so
consecutive requests share a common prefix. `prefix_merging` detects this and
merges the chain into a single trace with one prompt and the concatenated turns.

The join test is a **strict token prefix**: a new request joins the chain only
when its `prompt_ids` start with the previous completion's prompt (append-only).
A message-level key gates candidates first, and it deliberately ignores
tool-result and empty assistant messages so ordinary tool loops still merge. When
the prefix relationship breaks — e.g. after context compaction rewrites earlier
turns — a new trace is started (and a partially merged chain can be truncated at
the break).

## Loss mask and logprobs

Builders set `loss_mask = 1` for the sampled assistant tokens that should train
the policy and `loss_mask = 0` for interstitial/copied tokens. Sampled tokens
keep their real logprobs; `prefix_merging` fills interstitial positions with
`0.0` placeholders so the arrays stay aligned. The turn boundary is found via an
end-of-turn token (auto-detected, or set explicitly with the builder's
`end_of_turn_token_id` config). Training bridges expect trainable tokens to have
matching logprob data.
