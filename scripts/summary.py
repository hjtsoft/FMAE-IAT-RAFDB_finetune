import sys, os, re, glob

def main():
    if len(sys.argv) < 2:
        print("Usage: python summary.py <base_dir>")
        sys.exit(1)
        
    base_dir = sys.argv[1]
    dirs = sorted(glob.glob(os.path.join(base_dir, "A0_multiseed_seed*")))

    rows = []
    for d in dirs:
        log = os.path.join(d, "train.log")
        if not os.path.isfile(log):
            continue
        with open(log, encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        # 单次最高精度
        m = re.findall(r"Max accuracy:\s*([\d.]+)%", txt)
        best_acc = float(m[-1]) if m else None

        # 单次最高精度对应的 epoch
        me = re.findall(r"Max accuracy:\s*[\d.]+%\s*at epoch\s*(\d+)", txt)
        best_ep = int(me[-1]) if me else None

        # TTA 精度
        mt = re.findall(r"TTA 最终精度(?:\(baseline\))?:\s*([\d.]+)%", txt)
        tta_acc = float(mt[-1]) if mt else None

        rows.append((os.path.basename(d), best_acc, best_ep, tta_acc))

    SEP  = "=" * 72
    SEP2 = "-" * 72

    print(SEP)
    print("  RAF-DB Multi-Seed 结果汇总")
    print(SEP)
    print(f"  {'实验目录':<30} {'单次最高':>10} {'最佳Epoch':>9} {'TTA':>10}")
    print(SEP2)

    best_accs, tta_accs = [], []
    for name, ba, ep, ta in rows:
        ba_s = f"{ba:.3f}%" if ba is not None else "N/A"
        ep_s = str(ep)         if ep is not None else "N/A"
        ta_s = f"{ta:.3f}%"   if ta is not None else "N/A"
        print(f"  {name:<30} {ba_s:>10} {ep_s:>9} {ta_s:>10}")
        if ba is not None: best_accs.append(ba)
        if ta is not None: tta_accs.append(ta)

    if best_accs or tta_accs:
        print(SEP2)

    def stats(vals, label):
        if not vals:
            return
        n    = len(vals)
        mu   = sum(vals) / n
        std  = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        best = max(vals)
        print(f"  {label}:")
        print(f"    seeds     = {n}")
        print(f"    均值      = {mu:.3f}%")
        print(f"    标准差    = ±{std:.3f}%")
        print(f"    最高      = {best:.3f}%")

    stats(best_accs, "单次最高精度（各 seed top-1）")
    if tta_accs:
        print()
        stats(tta_accs,  "TTA 精度")

    print(SEP)
    print("  对比目标：FMAE 论文 93.09% / GitHub 93.45%")
    if best_accs:
        mu = sum(best_accs) / len(best_accs)
        verdict = "✅ 超过论文" if mu > 93.09 else "❌ 未达论文"
        print(f"  单次均值 {mu:.3f}%  →  {verdict}")
    if tta_accs:
        mu = sum(tta_accs) / len(tta_accs)
        verdict = "✅ 超过 GitHub" if mu > 93.45 else ("✅ 超过论文" if mu > 93.09 else "❌ 未达论文")
        print(f"  TTA  均值 {mu:.3f}%  →  {verdict}")
    print(SEP)

if __name__ == '__main__':
    main()
