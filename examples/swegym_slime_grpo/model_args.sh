# shellcheck shell=bash
# One model preset entry; preserve each architecture and dispatcher configuration.
case "${TRAINING_MODEL_SIZE:-4b}" in
4b|9b)
if [[ "${TRAINING_MODEL_SIZE:-4b}" == "9b" ]]; then
    _hidden_size=4096
    _ffn_hidden_size=12288
    _embedding_args=(--untie-embeddings-and-output-weights)
else
    _hidden_size=2560
    _ffn_hidden_size=9216
    _embedding_args=()
fi
# Shared Dense attention and normalization; architecture dimensions and
# embedding tying follow the selected Hugging Face checkpoint.
MODEL_ARGS=(
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --kv-channels 256
    --num-layers 32
    --hidden-size "${_hidden_size}"
    --ffn-hidden-size "${_ffn_hidden_size}"
    --use-gated-attention
    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    "${_embedding_args[@]}"
    --vocab-size 248320
    --rotary-base 10000000
    --attention-output-gate
)
unset _hidden_size _ffn_hidden_size _embedding_args
;;
35b)
# shellcheck shell=bash
# Qwen3.5-35B-A3B Megatron model args.
source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

# Prefer the official flex/DeepEP dispatcher when the optional extension is
# installed. The shared ProRL venv does not guarantee DeepEP, so auto mode
# keeps the model preset's alltoall dispatcher as a functional fallback.
MOE_DISPATCHER_BACKEND="${MOE_DISPATCHER_BACKEND:-auto}"
case "${MOE_DISPATCHER_BACKEND}" in
    auto)
        if "${PYTHON_BIN}" -c 'import deep_ep' >/dev/null 2>&1; then
            MODEL_ARGS+=(--moe-token-dispatcher-type flex --moe-enable-deepep)
            echo "[moe] dispatcher=flex backend=deepep"
        else
            echo "[moe] DeepEP not installed; dispatcher=alltoall"
        fi
        ;;
    deepep)
        "${PYTHON_BIN}" -c 'import deep_ep' >/dev/null 2>&1 || {
            echo "MOE_DISPATCHER_BACKEND=deepep requires the deep_ep Python extension" >&2
            return 1
        }
        MODEL_ARGS+=(--moe-token-dispatcher-type flex --moe-enable-deepep)
        ;;
    alltoall)
        ;;
    *)
        echo "MOE_DISPATCHER_BACKEND must be auto, deepep, or alltoall" >&2
        return 2
        ;;
esac

;;
*) echo "TRAINING_MODEL_SIZE must be 4b, 9b or 35b" >&2; return 2 ;;
esac
