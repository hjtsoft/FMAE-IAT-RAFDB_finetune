#!/usr/bin/env bash
set -eu
# 在某些环境中脚本可能被 sh 间接执行，pipefail 不是所有 shell 都支持
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

# 一键跑 RAF-DB 消融：
# A0: 无先验掩码
# A1: 有先验掩码（默认）
# A2: 有先验掩码 + text_attn 低学习率倍率

SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BLR="${BLR:-0.001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-6}"
SMOOTHING="${SMOOTHING:-0.15}"
MODEL="${MODEL:-vit_large_patch16}"
FINETUNE="${FINETUNE:-/Data/hjt/NLA/pretrain_models/FMAE_ViT_large.pth}"
TRAIN_PATH="${TRAIN_PATH:-dummy}"
TEST_PATH="${TEST_PATH:-dummy}"
BASE_DIR="${BASE_DIR:-./ablation_runs}"

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
)

run_one() {
  local tag="$1"
  local seed="$2"
  shift 2

  local out_dir="${BASE_DIR}/${tag}_seed${seed}"
  mkdir -p "${out_dir}"

  echo "[RUN] ${tag} seed=${seed}"
  python RAFDB_finetune.py \
    --seed "${seed}" \
    "${common_args[@]}" \
    --output_dir "${out_dir}" \
    --log_dir "${out_dir}" \
    "$@" | tee "${out_dir}/train.log"
}

for s in ${SEEDS}; do
  # A0: 关闭先验掩码
  run_one A0_nomask "${s}" --prior_mask_dir ""

  # A1: 开启先验掩码（使用默认 prior_mask_dir）
  run_one A1_mask "${s}"

  # A2: 开启先验掩码 + 更温和的 text_attn lr scale
  run_one A2_mask_lr5 "${s}" --text_attn_lr_scale 5.0
done

summary_file="${BASE_DIR}/summary.txt"
{
  echo "=== RAF-DB Ablation Summary ==="
  for d in "${BASE_DIR}"/*_seed*; do
    [ -d "${d}" ] || continue
    if [ -f "${d}/train.log" ]; then
      best_line=$(grep -E "Max accuracy:" "${d}/train.log" | tail -n 1 || true)
      tta_line=$(grep -E "TTA 最终精度:" "${d}/train.log" | tail -n 1 || true)
      echo "$(basename "${d}") | ${best_line:-Max accuracy: N/A} | ${tta_line:-TTA 最终精度: N/A}"
    fi
  done
} | tee "${summary_file}"

echo "[DONE] 结果汇总: ${summary_file}"
