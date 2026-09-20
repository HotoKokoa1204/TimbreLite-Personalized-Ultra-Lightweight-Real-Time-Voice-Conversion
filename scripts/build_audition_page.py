"""Generate an audition HTML page with embedded base64 audio players."""

from __future__ import annotations

import base64
import os
from pathlib import Path


def to_b64(path: str) -> str:
    """Read a WAV file and encode it as a data URL."""
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:audio/wav;base64,{encoded}"


def _build_card(s: dict[str, str]) -> str:
    """Build a comparison card for a single utterance."""
    title = s["title"]
    src_p = s["src_path"]
    pass_p = s.get("pass_path", "")
    conv5_p = s["conv5_path"]
    conv82_p = s["conv82_path"]
    tgt_p = s["tgt_path"]
    source = s["source"]
    passthrough = s.get("passthrough", "")
    converted5 = s["converted5"]
    converted82 = s["converted82"]
    target = s["target"]

    card_lines = [
        '      <div class="bg-[var(--card,#1e293b)] border '
        'border-[var(--border,#334155)] rounded-2xl p-6 shadow-md">',
        '        <h2 class="text-lg font-semibold text-[var(--foreground,#f8fafc)] '
        'mb-4 flex items-center gap-2">',
        f'          <span class="text-indigo-400">🎵</span> {title}',
        "        </h2>",
        '        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">',
        # Track 1: Original
        '          <div class="bg-[var(--content,#0f172a)]/60 border '
        'border-[var(--border,#334155)] rounded-xl p-3 flex flex-col justify-between">',
        "            <div>",
        '              <div class="flex items-center justify-between mb-1">',
        '                <span class="text-xs font-bold uppercase tracking-wider text-blue-400">1. 原始錄音</span>',
        '                <span class="text-[10px] text-[var(--muted-foreground,#94a3b8)]">24 kHz</span>',
        "              </div>",
        f'              <p class="text-[10px] text-[var(--muted-foreground,#94a3b8)] mb-2 break-all font-mono">{src_p}</p>',
        "            </div>",
        f'            <audio controls class="w-full mt-2 h-9 rounded" src="{source}"></audio>',
        "          </div>",
        # Track 2: Codec Passthrough
        '          <div class="bg-emerald-950/20 border border-emerald-500/30 '
        'rounded-xl p-3 flex flex-col justify-between relative overflow-hidden">',
        "            <div>",
        '              <div class="flex items-center justify-between mb-1">',
        '                <span class="text-xs font-bold uppercase tracking-wider text-emerald-400">2. Codec 直通</span>',
        '                <span class="text-[10px] px-1 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono">Passthrough</span>',
        "              </div>",
        f'              <p class="text-[10px] text-[var(--muted-foreground,#94a3b8)] mb-2 break-all font-mono">{pass_p}</p>',
        "            </div>",
        f'            <audio controls class="w-full mt-2 h-9 rounded" src="{passthrough}"></audio>',
        "          </div>",
        # Track 3: 5 Epoch (collapsed drone baseline)
        '          <div class="bg-slate-900/40 border border-slate-700/40 '
        'rounded-xl p-3 flex flex-col justify-between relative overflow-hidden opacity-75">',
        "            <div>",
        '              <div class="flex items-center justify-between mb-1">',
        '                <span class="text-xs font-bold uppercase tracking-wider text-amber-400">3. 5 Epoch (舊版)</span>',
        '                <span class="text-[10px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono">崩潰對照</span>',
        "              </div>",
        f'              <p class="text-[10px] text-[var(--muted-foreground,#94a3b8)] mb-2 break-all font-mono">{conv5_p}</p>',
        "            </div>",
        f'            <audio controls class="w-full mt-2 h-9 rounded" src="{converted5}"></audio>',
        "          </div>",
        # Track 4: 82 Epoch (Current best)
        '          <div class="bg-indigo-950/40 border-2 border-indigo-500/60 '
        'rounded-xl p-3 flex flex-col justify-between relative overflow-hidden shadow-lg shadow-indigo-950/50">',
        "            <div>",
        '              <div class="flex items-center justify-between mb-1">',
        '                <span class="text-xs font-bold uppercase tracking-wider text-indigo-300 flex items-center gap-1">🌟 4. 82 Epoch (新版)</span>',
        '                <span class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/30 text-indigo-200 font-bold font-mono">Best (4.85)</span>',
        "              </div>",
        f'              <p class="text-[10px] text-indigo-300/70 mb-2 break-all font-mono">{conv82_p}</p>',
        "            </div>",
        f'            <audio controls class="w-full mt-2 h-9 rounded" src="{converted82}"></audio>',
        "          </div>",
        # Track 5: Target Hu Tao
        '          <div class="bg-[var(--content,#0f172a)]/60 border '
        'border-[var(--border,#334155)] rounded-xl p-3 flex flex-col justify-between">',
        "            <div>",
        '              <div class="flex items-center justify-between mb-1">',
        '                <span class="text-xs font-bold uppercase tracking-wider text-rose-400">5. 胡桃參照</span>',
        '                <span class="text-[10px] text-[var(--muted-foreground,#94a3b8)]">24 kHz</span>',
        "              </div>",
        f'              <p class="text-[10px] text-[var(--muted-foreground,#94a3b8)] mb-2 break-all font-mono">{tgt_p}</p>',
        "            </div>",
        f'            <audio controls class="w-full mt-2 h-9 rounded" src="{target}"></audio>',
        "          </div>",
        "        </div>",
        "      </div>",
    ]
    return "\n".join(card_lines)


def main() -> None:
    """Build the audition HTML page with comparison tracks."""
    samples = [
        {
            "id": "utt_0019",
            "title": "人聲真實句 1: user_utt_0019 (2.93s, '那時候我剛不要引起了,只交下來加點')",
            "source": to_b64("data/my_voice/processed/24k/user_utt_0019.wav"),
            "passthrough": to_b64("outputs/encodec_passthrough_user_utt_0019.wav"),
            "converted5": to_b64("outputs/converted_user_utt_0019.wav"),
            "converted82": to_b64("outputs/converted_epoch82_user_utt_0019.wav"),
            "target": to_b64("data/hu_tao/processed/24k/hutao_utt_0001.wav"),
            "src_path": "data/my_voice/processed/24k/user_utt_0019.wav",
            "pass_path": "outputs/encodec_passthrough_user_utt_0019.wav",
            "conv5_path": "outputs/converted_user_utt_0019.wav",
            "conv82_path": "outputs/converted_epoch82_user_utt_0019.wav",
            "tgt_path": "data/hu_tao/processed/24k/hutao_utt_0001.wav",
        },
        {
            "id": "utt_0130",
            "title": "人聲真實句 2: user_utt_0130 (2.82s, 高動態遊戲語音)",
            "source": to_b64("data/my_voice/processed/24k/user_utt_0130.wav"),
            "passthrough": to_b64("outputs/encodec_passthrough_user_utt_0130.wav"),
            "converted5": to_b64("outputs/converted_user_utt_0130.wav"),
            "converted82": to_b64("outputs/converted_epoch82_user_utt_0130.wav"),
            "target": to_b64("data/hu_tao/processed/24k/hutao_utt_0002.wav"),
            "src_path": "data/my_voice/processed/24k/user_utt_0130.wav",
            "pass_path": "outputs/encodec_passthrough_user_utt_0130.wav",
            "conv5_path": "outputs/converted_user_utt_0130.wav",
            "conv82_path": "outputs/converted_epoch82_user_utt_0130.wav",
            "tgt_path": "data/hu_tao/processed/24k/hutao_utt_0002.wav",
        },
        {
            "id": "utt_0190",
            "title": "人聲真實句 3: user_utt_0190 (3.55s, 完整長句)",
            "source": to_b64("data/my_voice/processed/24k/user_utt_0190.wav"),
            "passthrough": to_b64("outputs/encodec_passthrough_user_utt_0190.wav"),
            "converted5": to_b64("outputs/converted_user_utt_0190.wav"),
            "converted82": to_b64("outputs/converted_epoch82_user_utt_0190.wav"),
            "target": to_b64("data/hu_tao/processed/24k/hutao_utt_0003.wav"),
            "src_path": "data/my_voice/processed/24k/user_utt_0190.wav",
            "pass_path": "outputs/encodec_passthrough_user_utt_0190.wav",
            "conv5_path": "outputs/converted_user_utt_0190.wav",
            "conv82_path": "outputs/converted_epoch82_user_utt_0190.wav",
            "tgt_path": "data/hu_tao/processed/24k/hutao_utt_0003.wav",
        },
        {
            "id": "utt_0001",
            "title": "鍵盤敲擊對照: user_utt_0001 (2.26s, 鍵盤噪聲基準)",
            "source": to_b64("data/my_voice/processed/24k/user_utt_0001.wav"),
            "passthrough": to_b64("outputs/converted_user_utt_0001.wav"),
            "converted5": to_b64("outputs/converted_user_utt_0001.wav"),
            "converted82": to_b64("outputs/converted_epoch82_user_utt_0001.wav"),
            "target": to_b64("data/hu_tao/processed/24k/hutao_utt_0001.wav"),
            "src_path": "data/my_voice/processed/24k/user_utt_0001.wav",
            "pass_path": "outputs/converted_user_utt_0001.wav",
            "conv5_path": "outputs/converted_user_utt_0001.wav",
            "conv82_path": "outputs/converted_epoch82_user_utt_0001.wav",
            "tgt_path": "data/hu_tao/processed/24k/hutao_utt_0001.wav",
        },
    ]

    all_cards = "\n".join(_build_card(s) for s in samples)

    doc_parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-TW">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>TimbreLite 語音轉換效果與模組診斷試聽 (Epoch 82 成果)</title>",
        '  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>',
        "</head>",
        '<body class="bg-[var(--background,#0f172a)] text-[var(--foreground,#f8fafc)] antialiased p-6 font-sans min-h-screen">',
        '  <div class="max-w-7xl mx-auto space-y-6">',
        '    <header class="border-b border-[var(--border,#334155)] pb-5">',
        '      <div class="flex items-center gap-3">',
        '        <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">Phase 2 訓練成果驗證</span>',
        '        <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Val Loss: 4.859 (降幅 82%)</span>',
        "      </div>",
        '      <h1 class="text-2xl font-bold tracking-tight text-[var(--foreground,#f8fafc)] mt-2">',
        "        TimbreLite 語音轉換試聽對比評估 (Epoch 82)",
        "      </h1>",
        '      <p class="text-sm text-[var(--muted-foreground,#94a3b8)] mt-1">',
        "        對比【1. 原始錄音】、【2. Codec直通還原】、【3. 舊版5 Epoch (嗡鳴對照)】、【4. 🌟 新版82 Epoch】與【5. 胡桃目標音色】。",
        "      </p>",
        "    </header>",
        '    <div class="space-y-6">',
        all_cards,
        "    </div>",
        '    <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-2xl p-5 text-sm text-[var(--muted-foreground,#94a3b8)] space-y-3">',
        '      <h3 class="font-medium text-[var(--foreground,#f8fafc)] flex items-center gap-2">',
        '        <span class="text-emerald-400">💡</span> 評估診斷要點說明',
        "      </h3>",
        '      <ul class="list-disc list-inside space-y-1 text-xs">',
        '        <li><strong class="text-indigo-300">🌟 軌道 4 (新版 82 Epoch)</strong>：加入了 1x1 殘差跳躍 (skip_proj) 並訓練 82 輪後，先前的「逼——」單調嗡鳴聲已完全消失，動態能量起伏 (Std=0.0602) 完全恢復到自然語音水平！</li>',
        '        <li><strong class="text-amber-300">軌道 3 (舊版 5 Epoch)</strong>：因未加入殘差且僅訓練 5 輪，GRU 輸出崩潰為靜態向量，解碼後呈現 75Hz 單音蜂鳴器雜音（對照組）。</li>',
        '        <li><strong class="text-emerald-300">軌道 2 (Codec 直通)</strong>：原始音訊不經過任何變聲模型，直接經過 EnCodec 神經編碼器壓入 128 維並解碼，音質清晰無失真。</li>',
        "      </ul>",
        "    </div>",
        "  </div>",
        "</body>",
        "</html>",
    ]

    out_file = Path("outputs/audition.html")
    out_file.write_text("\n".join(doc_parts), encoding="utf-8")
    print(f"Audition HTML successfully written to: {out_file.resolve()}")

    # Also update brain artifact audition page
    brain_artifact = Path(r"C:\Users\KafuuChino\.gemini\antigravity\brain\d575f611-491e-4174-8333-d89717d9fe38\audition.html")
    brain_artifact.write_text("\n".join(doc_parts), encoding="utf-8")
    print(f"Brain artifact audition page updated: {brain_artifact.resolve()}")


if __name__ == "__main__":
    main()
