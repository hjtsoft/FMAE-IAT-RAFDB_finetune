#!/usr/bin/env bash
set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

# ============================================================
# RAF-DB 多 Seed 验证脚本
#
# 目标：验证 TextGuidedSaliencyMask（单流抑制）在 seed=0/1/2 下
#       的平均精度，与官方 FMAE README 报告的 93.45% 对比
#
# 运行方式：
#   SEEDS="0 1 2" bash scripts/run_rafdb_ablation.sh
#
# 超参与官方 FMAE：
#   blr=0.001, epochs=60, warmup=6, smoothing=0.15
#   batch_size=32, layer_decay=0.65（完全一致）
# ============================================================

SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BLR="${BLR:-0.001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-6}"
SMOOTHING="${SMOOTHING:-0.15}"
MODEL="${MODEL:-vit_large_patch16}"
FINETUNE="${FINETUNE:-/Data/hjt/NLA/pretrain_models/FMAE_ViT_large.pth}"
TRAIN_PATH="${TRAIN_PATH:-dummy}"
TEST_PATH="${TEST_PATH:-dummy}"
BASE_DIR="${BASE_DIR:-./multiseed_runs}"

mkdir -p "${BASE_DIR}"

common_args=(
  --blr "${BLR}"
  --nb_classes 7
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --smoothing "${SMOOTHING}"
  --model "${MODEL}"
  --finetune "${FINETUNE}"
  --train_path "${TRAIN_PATH}"
  --test_path "${TEST_PATH}"
  --prior_mask_dir ""
  --text_attn_lr_scale 50.0
)

run_one() {
  local tag="$1"
  local seed="$2"
  shift 2

  local out_dir="${BASE_DIR}/${tag}_seed${seed}"
  mkdir -p "${out_dir}"

  echo "============================================"
  echo "[RUN] ${tag}  seed=${seed}  → ${out_dir}"
  echo "============================================"
  python RAFDB_finetune.py \
    --seed "${seed}" \
    "${common_args[@]}" \
    --output_dir "${out_dir}" \
    --log_dir    "${out_dir}" \
    "$@" 2>&1 | tee "${out_dir}/train.log"
}

# ── 3 个 seed 串行训练 ──────────────────────────────────────────────────────
for s in ${SEEDS}; do
  run_one A0_multiseed "${s}"
done

# ── 结果汇总（用独立的 Python 脚本计算，不使用会产生临时文件的 Here-Document）─────────────────
summary_file="${BASE_DIR}/summary.txt"

# 获取脚本所在的目录 (也就是 scripts/ 目录)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 "${SCRIPT_DIR}/summary.py" "${BASE_DIR}" | tee "${summary_file}"

echo ""
echo "[DONE] 结果汇总: ${summary_file}"
