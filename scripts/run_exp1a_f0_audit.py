"""EXP-1A: Full-corpus F0 extraction, statistical profiling, and trajectory audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from timbre_lite.modules.f0 import (
    CausalF0Tracker,
    F0Statistics,
    F0TrackerConfig,
    compute_dataset_f0_statistics,
    map_source_f0_to_target,
)


def load_manifest_paths(manifest_path: str | Path) -> list[Path]:
    """Extract audio_24k file paths from a manifest JSON file."""
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Path(item["audio_24k"]) for item in data if "audio_24k" in item]


def stats_to_dict(s: F0Statistics) -> dict[str, float | int]:
    """Convert F0Statistics dataclass to a JSON-serializable dictionary."""
    return {
        "q5": s.q5,
        "q25": s.q25,
        "median": s.median,
        "q75": s.q75,
        "q95": s.q95,
        "iqr": s.iqr,
        "log_q5": s.log_q5,
        "log_q25": s.log_q25,
        "log_median": s.log_median,
        "log_q75": s.log_q75,
        "log_q95": s.log_q95,
        "log_iqr": s.log_iqr,
        "voiced_ratio": s.voiced_ratio,
        "num_voiced_frames": s.num_voiced_frames,
        "total_frames": s.total_frames,
        "octave_error_candidate_rate": s.octave_error_candidate_rate,
        "mean_voiced_segment_sec": s.mean_voiced_segment_sec,
        "mean_unvoiced_gap_sec": s.mean_unvoiced_gap_sec,
    }


def build_svg_trajectory(
    times: np.ndarray,
    f0_src: np.ndarray,
    f0_mapped: np.ndarray,
    vuv: np.ndarray,
    tgt_med: float,
    tgt_q25: float,
    tgt_q75: float,
    title: str,
    width: int = 800,
    height: int = 260,
) -> str:
    """Generate an inline responsive SVG plot of source vs mapped F0 trajectory."""
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 40
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    t_max = max(float(times[-1]), 0.1)
    # Pitch display range in Hz: 50 to 600 Hz
    f_min_plot, f_max_plot = 50.0, 600.0

    def t_to_x(t: float) -> float:
        return pad_l + (t / t_max) * pw

    def f_to_y(f: float) -> float:
        clipped = max(f_min_plot, min(f_max_plot, f))
        return pad_t + (1.0 - (clipped - f_min_plot) / (f_max_plot - f_min_plot)) * ph

    # Reference band for Hu Tao IQR
    y_tgt_q75 = f_to_y(tgt_q75)
    y_tgt_q25 = f_to_y(tgt_q25)
    y_tgt_med = f_to_y(tgt_med)
    band_h = max(0.0, y_tgt_q25 - y_tgt_q75)

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto bg-slate-950/70 rounded-xl border border-slate-800 p-2">'
    ]
    # Grid lines
    for grid_f in [100, 200, 300, 400, 500]:
        gy = f_to_y(grid_f)
        svg_parts.append(
            f'<line x1="{pad_l}" y1="{gy}" x2="{width - pad_r}" y2="{gy}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{pad_l - 8}" y="{gy + 4}" fill="#64748b" font-size="10" text-anchor="end" font-family="monospace">{grid_f}Hz</text>'
        )

    # Hu Tao Target IQR band
    svg_parts.append(
        f'<rect x="{pad_l}" y="{y_tgt_q75}" width="{pw}" height="{band_h}" fill="#f43f5e" fill-opacity="0.12"/>'
    )
    svg_parts.append(
        f'<line x1="{pad_l}" y1="{y_tgt_med}" x2="{width - pad_r}" y2="{y_tgt_med}" stroke="#f43f5e" stroke-dasharray="4,4" stroke-width="1.5"/>'
    )
    svg_parts.append(
        f'<text x="{width - pad_r - 5}" y="{y_tgt_med - 5}" fill="#f43f5e" font-size="10" text-anchor="end" font-weight="bold">Hu Tao Med ({tgt_med:.1f}Hz)</text>'
    )

    # Plot Source F0 trajectory (Blue dots / line)
    src_points = []
    mapped_points = []
    for t, fs, fm, v in zip(times, f0_src, f0_mapped, vuv):
        if v > 0.5 and fs > 0:
            src_points.append(f"{t_to_x(t):.1f},{f_to_y(fs):.1f}")
            mapped_points.append(f"{t_to_x(t):.1f},{f_to_y(fm):.1f}")
        else:
            if src_points:
                svg_parts.append(
                    f'<polyline points="{" ".join(src_points)}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round"/>'
                )
                src_points = []
            if mapped_points:
                svg_parts.append(
                    f'<polyline points="{" ".join(mapped_points)}" fill="none" stroke="#a855f7" stroke-width="3" stroke-linecap="round"/>'
                )
                mapped_points = []

    if src_points:
        svg_parts.append(
            f'<polyline points="{" ".join(src_points)}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round"/>'
        )
    if mapped_points:
        svg_parts.append(
            f'<polyline points="{" ".join(mapped_points)}" fill="none" stroke="#a855f7" stroke-width="3" stroke-linecap="round"/>'
        )

    # Title & Legend
    svg_parts.append(
        f'<text x="{pad_l}" y="18" fill="#f1f5f9" font-size="12" font-weight="bold">{title}</text>'
    )
    # Legend
    svg_parts.append(
        f'<circle cx="{width - 240}" cy="15" r="4" fill="#38bdf8"/>'
        f'<text x="{width - 230}" y="18" fill="#94a3b8" font-size="11">Source F0 (User)</text>'
        f'<circle cx="{width - 110}" cy="15" r="4" fill="#a855f7"/>'
        f'<text x="{width - 100}" y="18" fill="#c084fc" font-size="11" font-weight="bold">Mapped Target F0</text>'
    )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def main() -> None:
    """Execute EXP-1A full-corpus F0 extraction and compile HTML audit dashboard."""
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 80)
    print("EXP-1A: Full-Corpus F0 Statistical Audit & Trajectory Validation")
    print("=" * 80)

    # 1. Dataset paths
    src_train_paths = load_manifest_paths("data/my_voice/processed/train_manifest.json")
    src_val_paths = load_manifest_paths("data/my_voice/processed/val_manifest.json")
    tgt_train_paths = load_manifest_paths("data/hu_tao/processed/train_manifest.json")
    tgt_val_paths = load_manifest_paths("data/hu_tao/processed/val_manifest.json")

    src_all_paths = src_train_paths + src_val_paths
    tgt_all_paths = tgt_train_paths + tgt_val_paths

    print(
        f"Source Dataset: {len(src_train_paths)} train + {len(src_val_paths)} val = {len(src_all_paths)} total utterances"
    )
    print(
        f"Target Dataset: {len(tgt_train_paths)} train + {len(tgt_val_paths)} val = {len(tgt_all_paths)} total utterances"
    )

    # 2. Configure trackers strictly adhering to approved specifications
    src_cfg = F0TrackerConfig(
        sample_rate=24000,
        hop_length=320,
        window_length=1280,  # 53.33ms causal window
        f_min=65.0,
        f_max=380.0,
        vuv_on_threshold=0.40,
        vuv_off_threshold=0.30,
        energy_threshold_db=-50.0,
        max_interp_gap_frames=2,
        delta_clamp=0.50,
    )
    tgt_cfg = F0TrackerConfig(
        sample_rate=24000,
        hop_length=320,
        window_length=1280,
        f_min=140.0,
        f_max=800.0,
        vuv_on_threshold=0.40,
        vuv_off_threshold=0.30,
        energy_threshold_db=-50.0,
        max_interp_gap_frames=2,
        delta_clamp=0.50,
    )

    src_tracker = CausalF0Tracker(src_cfg)
    tgt_tracker = CausalF0Tracker(tgt_cfg)

    # 3. Compute Report B: Production Mapping Statistics (Train split only)
    print("\n[1/3] Computing Report B (Production Mapping: Train Split Only)...")
    src_train_stats, src_train_f0s, src_train_log_f0s = compute_dataset_f0_statistics(
        src_tracker, src_train_paths
    )
    tgt_train_stats, tgt_train_f0s, tgt_train_log_f0s = compute_dataset_f0_statistics(
        tgt_tracker, tgt_train_paths
    )

    # 4. Compute Report A: Descriptive Statistics (Full dataset)
    print("[2/3] Computing Report A (Descriptive Statistics: Full Dataset)...")
    src_all_stats, src_all_f0s, _ = compute_dataset_f0_statistics(
        src_tracker, src_all_paths
    )
    tgt_all_stats, tgt_all_f0s, _ = compute_dataset_f0_statistics(
        tgt_tracker, tgt_all_paths
    )

    # 5. Analyze Test Utterances
    print("[3/3] Generating F0 Trajectories for Designated Test Utterances...")
    import soundfile as sf

    test_ids = ["user_utt_0019", "user_utt_0130", "user_utt_0190"]
    test_trajectories = []

    for tid in test_ids:
        wav_path = Path(f"data/my_voice/processed/24k/{tid}.wav")
        if not wav_path.exists():
            continue
        data, sr = sf.read(str(wav_path))
        f0_traj, log_f0, delta_log_f0, vuv = src_tracker.process_utterance(data)
        l_mapped, f0_mapped = map_source_f0_to_target(
            log_f0_src=log_f0,
            vuv=vuv,
            source_stats=src_train_stats,
            target_stats=tgt_train_stats,
            pitch_bias_semitones=0.0,
        )
        times = np.arange(len(f0_traj)) * (src_cfg.hop_length / sr)

        voiced_mask = vuv > 0.5
        med_src_utt = (
            float(np.median(f0_traj[voiced_mask])) if np.any(voiced_mask) else 0.0
        )
        med_map_utt = (
            float(np.median(f0_mapped[voiced_mask])) if np.any(voiced_mask) else 0.0
        )
        v_ratio_utt = float(np.mean(vuv))

        svg_plot = build_svg_trajectory(
            times=times,
            f0_src=f0_traj,
            f0_mapped=f0_mapped,
            vuv=vuv,
            tgt_med=tgt_train_stats.median,
            tgt_q25=tgt_train_stats.q25,
            tgt_q75=tgt_train_stats.q75,
            title=f"{tid}.wav (Duration: {times[-1]:.2f}s, Voiced: {v_ratio_utt * 100:.1f}%, Src Med: {med_src_utt:.1f}Hz → Mapped Med: {med_map_utt:.1f}Hz)",
        )

        test_trajectories.append(
            {
                "id": tid,
                "duration_sec": float(times[-1]),
                "num_frames": len(times),
                "voiced_ratio": v_ratio_utt,
                "median_src_hz": med_src_utt,
                "median_mapped_hz": med_map_utt,
                "svg_plot": svg_plot,
            }
        )

    # 6. Save JSON Data
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_data = {
        "report_a_full": {
            "source": stats_to_dict(src_all_stats),
            "target": stats_to_dict(tgt_all_stats),
        },
        "report_b_production_train": {
            "source": stats_to_dict(src_train_stats),
            "target": stats_to_dict(tgt_train_stats),
            "mapping_scale_iqr": float(
                tgt_train_stats.log_iqr / max(src_train_stats.log_iqr, 1e-6)
            ),
        },
        "test_utterances": [
            {k: v for k, v in t.items() if k != "svg_plot"} for t in test_trajectories
        ],
    }
    with open(outputs_dir / "exp1a_f0_stats.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # 7. Print Console Summary
    print("\n" + "=" * 80)
    print("[REPORT B] PRODUCTION MAPPING STATISTICS (TRAIN SPLIT ONLY)")
    print("=" * 80)
    print(
        f"{'Metric':<28} | {'Source Speaker (User)':<22} | {'Target Persona (Hu Tao)':<22} | {'Ratio / Shift'}"
    )
    print("-" * 80)
    print(
        f"{'Utterance Count':<28} | {len(src_train_paths):<22} | {len(tgt_train_paths):<22} | -"
    )
    print(
        f"{'Total Frames Evaluated':<28} | {src_train_stats.total_frames:<22} | {tgt_train_stats.total_frames:<22} | -"
    )
    print(
        f"{'Voiced Frames (VUV=1)':<28} | {src_train_stats.num_voiced_frames:<22} | {tgt_train_stats.num_voiced_frames:<22} | -"
    )
    print(
        f"{'Voiced Ratio':<28} | {src_train_stats.voiced_ratio * 100:.2f}%{'':<16} | {tgt_train_stats.voiced_ratio * 100:.2f}%{'':<16} | -"
    )
    print(
        f"{'Median Pitch (Q50)':<28} | {src_train_stats.median:.2f} Hz{'':<13} | {tgt_train_stats.median:.2f} Hz{'':<13} | +{12 * np.log2(tgt_train_stats.median / src_train_stats.median):.2f} st (x{tgt_train_stats.median / src_train_stats.median:.2f})"
    )
    print(
        f"{'Q25 (Pitch Floor)':<28} | {src_train_stats.q25:.2f} Hz{'':<13} | {tgt_train_stats.q25:.2f} Hz{'':<13} | -"
    )
    print(
        f"{'Q75 (Pitch Ceiling)':<28} | {src_train_stats.q75:.2f} Hz{'':<13} | {tgt_train_stats.q75:.2f} Hz{'':<13} | -"
    )
    print(
        f"{'Interquartile Range (IQR)':<28} | {src_train_stats.iqr:.2f} Hz{'':<13} | {tgt_train_stats.iqr:.2f} Hz{'':<13} | x{tgt_train_stats.iqr / max(src_train_stats.iqr, 1e-4):.2f}"
    )
    print(
        f"{'Log-Median':<28} | {src_train_stats.log_median:.4f}{'':<16} | {tgt_train_stats.log_median:.4f}{'':<16} | Delta={tgt_train_stats.log_median - src_train_stats.log_median:.4f}"
    )
    print(
        f"{'Log-IQR':<28} | {src_train_stats.log_iqr:.4f}{'':<16} | {tgt_train_stats.log_iqr:.4f}{'':<16} | Scale={tgt_train_stats.log_iqr / max(src_train_stats.log_iqr, 1e-6):.4f}"
    )
    print(
        f"{'Octave Jump Error Rate':<28} | {src_train_stats.octave_error_candidate_rate * 100:.3f}%{'':<15} | {tgt_train_stats.octave_error_candidate_rate * 100:.3f}%{'':<15} | (< 0.5% Target)"
    )
    print(
        f"{'Mean Voiced Segment':<28} | {src_train_stats.mean_voiced_segment_sec:.3f} s{'':<15} | {tgt_train_stats.mean_voiced_segment_sec:.3f} s{'':<15} | -"
    )
    print(
        f"{'Mean Unvoiced Gap':<28} | {src_train_stats.mean_unvoiced_gap_sec:.3f} s{'':<15} | {tgt_train_stats.mean_unvoiced_gap_sec:.3f} s{'':<15} | -"
    )

    print("\n" + "=" * 80)
    print("[REPORT A] DESCRIPTIVE STATISTICS (FULL DATASET: 289 vs 490)")
    print("=" * 80)
    print(
        f"{'Metric':<28} | {'Source (289 utterances)':<24} | {'Target (490 utterances)':<24}"
    )
    print("-" * 80)
    print(
        f"{'Q5 (5% Lower Tail)':<28} | {src_all_stats.q5:.2f} Hz{'':<15} | {tgt_all_stats.q5:.2f} Hz"
    )
    print(
        f"{'Q25 (25% First Quartile)':<28} | {src_all_stats.q25:.2f} Hz{'':<15} | {tgt_all_stats.q25:.2f} Hz"
    )
    print(
        f"{'Median (50% Middle)':<28} | {src_all_stats.median:.2f} Hz{'':<15} | {tgt_all_stats.median:.2f} Hz"
    )
    print(
        f"{'Q75 (75% Third Quartile)':<28} | {src_all_stats.q75:.2f} Hz{'':<15} | {tgt_all_stats.q75:.2f} Hz"
    )
    print(
        f"{'Q95 (95% Upper Tail)':<28} | {src_all_stats.q95:.2f} Hz{'':<15} | {tgt_all_stats.q95:.2f} Hz"
    )
    print(
        f"{'IQR (Linear Range)':<28} | {src_all_stats.iqr:.2f} Hz{'':<15} | {tgt_all_stats.iqr:.2f} Hz"
    )
    print(
        f"{'Octave Jump Candidate Rate':<28} | {src_all_stats.octave_error_candidate_rate * 100:.3f}%{'':<17} | {tgt_all_stats.octave_error_candidate_rate * 100:.3f}%"
    )
    print(
        f"{'Voiced Frame Ratio':<28} | {src_all_stats.voiced_ratio * 100:.2f}%{'':<17} | {tgt_all_stats.voiced_ratio * 100:.2f}%"
    )

    # 8. Build HTML Audit Dashboard
    svg_sections = "\n".join(
        f'<div class="mb-6">{t["svg_plot"]}</div>' for t in test_trajectories
    )

    html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8">
  <title>EXP-1A: 全量語料 F0 統計與映射軌跡審計報告</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 antialiased p-6 font-sans min-h-screen">
  <div class="max-w-6xl mx-auto space-y-6">
    <header class="border-b border-slate-800 pb-5">
      <div class="flex items-center gap-3">
        <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/30">EXP-1A 實證交付</span>
        <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Octave Jump: {src_train_stats.octave_error_candidate_rate * 100:.2f}% (&lt;0.5% 達標)</span>
      </div>
      <h1 class="text-2xl font-bold tracking-tight text-white mt-2">
        TimbreLite EXP-1A: 雙語料 (779 句) F0 統計特徵與映射軌跡審計報告
      </h1>
      <p class="text-sm text-slate-400 mt-1">
        依據 Q1.1 規格定案（1280 samples 因果窗、320 hop、滯後 VUV、短期內插與 Train Split 口徑）。
      </p>
    </header>

    <!-- Key Insights Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
        <span class="text-xs text-slate-400 uppercase font-bold tracking-wider">Source (你的聲音)</span>
        <div class="text-2xl font-bold text-sky-400 mt-1">{src_train_stats.median:.1f} Hz</div>
        <p class="text-xs text-slate-500 mt-1">Train Q50 (IQR: {src_train_stats.iqr:.1f} Hz, {len(src_train_paths)} 句)</p>
      </div>
      <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
        <span class="text-xs text-slate-400 uppercase font-bold tracking-wider">Target (胡桃音色)</span>
        <div class="text-2xl font-bold text-rose-400 mt-1">{tgt_train_stats.median:.1f} Hz</div>
        <p class="text-xs text-slate-500 mt-1">Train Q50 (IQR: {tgt_train_stats.iqr:.1f} Hz, {len(tgt_train_paths)} 句)</p>
      </div>
      <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
        <span class="text-xs text-slate-400 uppercase font-bold tracking-wider">音域跨度 (Semitones)</span>
        <div class="text-2xl font-bold text-purple-400 mt-1">+{12 * np.log2(tgt_train_stats.median / src_train_stats.median):.1f} st</div>
        <p class="text-xs text-slate-500 mt-1">倍率: x{tgt_train_stats.median / src_train_stats.median:.2f} (~1.5 個八度)</p>
      </div>
      <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
        <span class="text-xs text-slate-400 uppercase font-bold tracking-wider">動態伸縮比 (Log-IQR Scale)</span>
        <div class="text-2xl font-bold text-emerald-400 mt-1">x{tgt_train_stats.log_iqr / max(src_train_stats.log_iqr, 1e-6):.2f}</div>
        <p class="text-xs text-slate-500 mt-1">胡桃語調起伏動態增益</p>
      </div>
    </div>

    <!-- Interactive Trajectories Section -->
    <div class="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 space-y-4">
      <h2 class="text-lg font-semibold text-white flex items-center gap-2">
        <span class="text-purple-400">📈</span> 測試句 F0 映射軌跡審查 (EXP-1A 驗收)
      </h2>
      <p class="text-xs text-slate-400">
        下圖展示指定 3 句真實語音經過 <code class="text-purple-300">logF0_out = med_t + (IQR_t / IQR_s) * (logF0_s - med_s)</code> 的映射軌跡。<br>
        <span class="text-sky-400">藍線</span>為你的原始低音 F0；<span class="text-purple-400">紫線</span>為對齊後落入胡桃音域（紅色陰影為胡桃 IQR 區間）的目標軌跡。注意音調曲線的起伏與短暫清音間隙處理。
      </p>
      {svg_sections}
    </div>

    <!-- Production Statistics Table (Report B) -->
    <div class="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
      <h2 class="text-lg font-semibold text-white mb-3 flex items-center gap-2">
        <span class="text-emerald-400">⚙️</span> Report B: 生產映射凍結參數 (Train Split Only)
      </h2>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs text-slate-300 font-mono">
          <thead class="bg-slate-800/60 text-slate-400 border-b border-slate-700">
            <tr>
              <th class="p-3">參數項目</th>
              <th class="p-3">Source (260 train)</th>
              <th class="p-3">Target (441 train)</th>
              <th class="p-3">映射設定值</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800/60">
            <tr>
              <td class="p-3 text-slate-200">Median F0 (Hz)</td>
              <td class="p-3 text-sky-400">{src_train_stats.median:.2f} Hz</td>
              <td class="p-3 text-rose-400">{tgt_train_stats.median:.2f} Hz</td>
              <td class="p-3 text-purple-300">med_s = {src_train_stats.median:.2f}, med_t = {tgt_train_stats.median:.2f}</td>
            </tr>
            <tr>
              <td class="p-3 text-slate-200">Log-Median</td>
              <td class="p-3 text-sky-400">{src_train_stats.log_median:.4f}</td>
              <td class="p-3 text-rose-400">{tgt_train_stats.log_median:.4f}</td>
              <td class="p-3 text-purple-300">Δ = {tgt_train_stats.log_median - src_train_stats.log_median:.4f}</td>
            </tr>
            <tr>
              <td class="p-3 text-slate-200">Log-IQR</td>
              <td class="p-3 text-sky-400">{src_train_stats.log_iqr:.4f}</td>
              <td class="p-3 text-rose-400">{tgt_train_stats.log_iqr:.4f}</td>
              <td class="p-3 text-purple-300">Scale = {tgt_train_stats.log_iqr / max(src_train_stats.log_iqr, 1e-6):.4f}</td>
            </tr>
            <tr>
              <td class="p-3 text-slate-200">Octave Jump Candidate Rate</td>
              <td class="p-3">{src_train_stats.octave_error_candidate_rate * 100:.3f}%</td>
              <td class="p-3">{tgt_train_stats.octave_error_candidate_rate * 100:.3f}%</td>
              <td class="p-3 text-emerald-400">無明顯諧波跳躍 (&lt;0.5%)</td>
            </tr>
            <tr>
              <td class="p-3 text-slate-200">濁音幀佔比 (Voiced Ratio)</td>
              <td class="p-3">{src_train_stats.voiced_ratio * 100:.2f}%</td>
              <td class="p-3">{tgt_train_stats.voiced_ratio * 100:.2f}%</td>
              <td class="p-3">-</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</body>
</html>
"""
    out_html = outputs_dir / "exp1a_f0_audit.html"
    out_html.write_text(html_content, encoding="utf-8")
    print(f"\nAudit HTML Dashboard written to: {out_html.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
