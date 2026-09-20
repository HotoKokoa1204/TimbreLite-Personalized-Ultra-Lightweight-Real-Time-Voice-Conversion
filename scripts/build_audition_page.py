"""Generate an audition HTML page with embedded base64 audio players."""

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
    conv_p = s["conv_path"]
    tgt_p = s["tgt_path"]
    source = s["source"]
    converted = s["converted"]
    target = s["target"]

    card_lines = [
        '      <div class="bg-[var(--card,#1e293b)] border '
        'border-[var(--border,#334155)] rounded-2xl p-6 shadow-md">',
        '        <h2 class="text-lg font-semibold text-[var(--foreground,#f8fafc)] '
        'mb-4 flex items-center gap-2">',
        f'          <span class="text-indigo-400">🎵</span> {title}',
        "        </h2>",
        '        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">',
        '          <div class="bg-[var(--content,#0f172a)]/60 border '
        'border-[var(--border,#334155)] rounded-xl p-4 flex flex-col '
        'justify-between">',
        "            <div>",
        '              <div class="flex items-center justify-between mb-2">',
        '                <span class="text-xs font-bold uppercase '
        'tracking-wider text-blue-400">1. 原始錄音</span>',
        '                <span class="text-[10px] '
        'text-[var(--muted-foreground,#94a3b8)]">24 kHz</span>',
        "              </div>",
        '              <p class="text-xs text-[var(--muted-foreground,#94a3b8)] '
        f'mb-3 break-all font-mono">{src_p}</p>',
        "            </div>",
        f'            <audio controls class="w-full mt-2 h-9 rounded" src="{source}">'
        "</audio>",
        "          </div>",
        '          <div class="bg-indigo-950/30 border border-indigo-500/40 '
        "rounded-xl p-4 flex flex-col justify-between relative overflow-hidden\">",
        "            <div>",
        '              <div class="flex items-center justify-between mb-2">',
        '                <span class="text-xs font-bold uppercase tracking-wider '
        'text-indigo-400">⚡ 2. 轉換輸出</span>',
        '                <span class="text-[10px] px-1.5 py-0.5 rounded '
        'bg-indigo-500/20 text-indigo-300 font-mono">13.33ms</span>',
        "              </div>",
        '              <p class="text-xs text-[var(--muted-foreground,#94a3b8)] '
        f'mb-3 break-all font-mono">{conv_p}</p>',
        "            </div>",
        '            <audio controls class="w-full mt-2 h-9 rounded" '
        f'src="{converted}"></audio>',
        "          </div>",
        '          <div class="bg-[var(--content,#0f172a)]/60 border '
        'border-[var(--border,#334155)] rounded-xl p-4 flex flex-col '
        'justify-between">',
        "            <div>",
        '              <div class="flex items-center justify-between mb-2">',
        '                <span class="text-xs font-bold uppercase '
        'tracking-wider text-rose-400">3. 胡桃參照</span>',
        '                <span class="text-[10px] '
        'text-[var(--muted-foreground,#94a3b8)]">24 kHz</span>',
        "              </div>",
        '              <p class="text-xs text-[var(--muted-foreground,#94a3b8)] '
        f'mb-3 break-all font-mono">{tgt_p}</p>',
        "            </div>",
        f'            <audio controls class="w-full mt-2 h-9 rounded" src="{target}">'
        "</audio>",
        "          </div>",
        "        </div>",
        "      </div>",
    ]
    return "\n".join(card_lines)


def main() -> None:
    """Build the audition HTML page with comparison tracks."""
    samples = [
        {
            "id": "utt_0001",
            "title": "Sample 1: user_utt_0001 (2.26s)",
            "source": to_b64("data/my_voice/processed/24k/user_utt_0001.wav"),
            "converted": to_b64("outputs/converted_user_utt_0001.wav"),
            "target": to_b64("data/hu_tao/processed/24k/hutao_utt_0001.wav"),
            "src_path": "data/my_voice/processed/24k/user_utt_0001.wav",
            "conv_path": "outputs/converted_user_utt_0001.wav",
            "tgt_path": "data/hu_tao/processed/24k/hutao_utt_0001.wav",
        },
        {
            "id": "utt_0002",
            "title": "Sample 2: user_utt_0002 (4.02s)",
            "source": to_b64("data/my_voice/processed/24k/user_utt_0002.wav"),
            "converted": to_b64("outputs/converted_user_utt_0002.wav"),
            "target": to_b64("data/hu_tao/processed/24k/hutao_utt_0002.wav"),
            "src_path": "data/my_voice/processed/24k/user_utt_0002.wav",
            "conv_path": "outputs/converted_user_utt_0002.wav",
            "tgt_path": "data/hu_tao/processed/24k/hutao_utt_0002.wav",
        },
        {
            "id": "utt_0003",
            "title": "Sample 3: user_utt_0003 (2.62s)",
            "source": to_b64("data/my_voice/processed/24k/user_utt_0003.wav"),
            "converted": to_b64("outputs/converted_user_utt_0003.wav"),
            "target": to_b64("data/hu_tao/processed/24k/hutao_utt_0003.wav"),
            "src_path": "data/my_voice/processed/24k/user_utt_0003.wav",
            "conv_path": "outputs/converted_user_utt_0003.wav",
            "tgt_path": "data/hu_tao/processed/24k/hutao_utt_0003.wav",
        },
    ]

    all_cards = "\n".join(_build_card(s) for s in samples)

    doc_parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-TW">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>TimbreLite 語音轉換效果試聽 (Hu Tao Target)</title>",
        '  <script src="https://www.gstatic.com/antigravity/web/dev/'
        'tailwindcss.min.js"></script>',
        "</head>",
        '<body class="bg-[var(--background,#0f172a)] '
        'text-[var(--foreground,#f8fafc)] antialiased p-6 font-sans min-h-screen">',
        '  <div class="max-w-4xl mx-auto space-y-6">',
        '    <div class="bg-[var(--card,#1e293b)] border '
        'border-[var(--border,#334155)] rounded-2xl p-6 shadow-lg">',
        '      <div class="flex items-center justify-between">',
        "        <div>",
        '          <h1 class="text-2xl font-bold tracking-tight '
        'text-[var(--foreground,#f8fafc)] flex items-center gap-2">',
        "            <span>🎙️</span> TimbreLite 語音轉換效果試聽 (Hu Tao Target)",
        "          </h1>",
        '          <p class="text-sm text-[var(--muted-foreground,#94a3b8)] mt-1">',
        "            Phase 6（Stage 1 蒸餾 + Stage 2 音色重構）｜ "
        "320-sample (13.33ms) 因果串流推論",
        "          </p>",
        "        </div>",
        '        <span class="px-3 py-1 text-xs font-semibold rounded-full '
        'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">',
        "          RTF: 0.83 (10.98ms)",
        "        </span>",
        "      </div>",
        "    </div>",
        '    <div class="space-y-6">',
        all_cards,
        "    </div>",
        '    <div class="bg-[var(--card,#1e293b)] border '
        'border-[var(--border,#334155)] rounded-2xl p-5 text-sm '
        'text-[var(--muted-foreground,#94a3b8)] space-y-2">',
        '      <h3 class="font-medium text-[var(--foreground,#f8fafc)] '
        'flex items-center gap-2">',
        "        <span>💡</span> 本地播放與檔案位置",
        "      </h3>",
        '      <ul class="list-disc list-inside space-y-1 text-xs">',
        "        <li><strong>瀏覽器直接開啟：</strong> 本頁面已生成於專案的 "
        "<code>outputs/audition.html</code>，直接雙擊即可用任何瀏覽器播放。</li>",
        "        <li><strong>WAV 檔案本機路徑：</strong>",
        '          <ul class="list-disc list-inside pl-4 mt-1 space-y-0.5 '
        'font-mono text-[11px]">',
        "            <li>原始錄音：data/my_voice/processed/24k/user_utt_0001.wav</li>",
        "            <li>轉換輸出：outputs/converted_user_utt_0001.wav</li>",
        "            <li>胡桃參照：data/hu_tao/processed/24k/hutao_utt_0001.wav</li>",
        "          </ul>",
        "        </li>",
        "      </ul>",
        "    </div>",
        "  </div>",
        "</body>",
        "</html>",
    ]
    html_content = "\n".join(doc_parts)

    os.makedirs("outputs", exist_ok=True)
    out_file = Path("outputs/audition.html")
    out_file.write_text(html_content, encoding="utf-8")

    brain_dir = Path(
        r"C:/Users/KafuuChino/.gemini/antigravity/brain/d575f611-491e-4174-8333-d89717d9fe38"
    )
    if brain_dir.exists():
        (brain_dir / "audition.html").write_text(html_content, encoding="utf-8")

    print(f"Generated {out_file} successfully! Size: {len(html_content):,} bytes")


if __name__ == "__main__":
    main()
