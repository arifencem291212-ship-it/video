#!/usr/bin/env python3
"""
Generator video motion graphics (9:16, TikTok / Reels / Shorts)
- Suara dilajukan sikit (pitch kekal, tak jadi suara tikus)
- Hook visual 3 saat pertama: flash, glitch RGB, teks pop-in
- Visual bereaksi dengan audio (bass pulse, bar spectrum, partikel)
- Caption pop-in (pilihan, dari fail .srt)
- Butang FOLLOW tengah video + CTA besar di hujung

Keperluan:  pip install numpy pillow     (+ ffmpeg dalam PATH)
Guna:
    python buat_video.py suara.mp3
    python buat_video.py suara.mp3 --srt sarikata.srt --handle @akaun_kau
    python buat_video.py suara.mp3 --scale 0.5 --t0 0 --t1 6      # preview laju
"""
import argparse, math, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

# ============================ TETAPAN (EDIT SINI) ============================
SPEED = 1.12                    # 1.0 = asal, 1.12 = laju sikit, 1.2 = agak laju
HANDLE = "@akaun_kau"           # nama akaun kau
FPS = 30
W, H = 1080, 1920

# Teks hook: (teks, masa_mula_saat, warna)   -- ubah ikut topik video kau
HOOK = [
    ("TUNGGU DULU!", 0.10, (255, 255, 255)),
    ("JANGAN SKIP", 0.90, (255, 214, 10)),
    ("SAMPAI HABIS", 1.70, (255, 59, 120)),
]
HOOK_END = 3.4                  # saat hook habis

CTA_LINES = ["SUKA VIDEO NI?", "FOLLOW SEKARANG!"]
CTA_SECONDS = 6.0               # berapa saat terakhir untuk CTA
MID_FOLLOW_AT = 0.38            # butang follow muncul pada 38% video
MID_FOLLOW_DUR = 3.2

# Palet warna (neon)
C1 = np.array([255, 59, 120])   # pink
C2 = np.array([0, 229, 255])    # cyan
C3 = np.array([255, 214, 10])   # kuning
BG_DARK = np.array([8, 6, 22])

FONT_PATHS = [
    os.environ.get("FONT_PATH", ""),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
# ============================================================================


# ------------------------------- utiliti ------------------------------------
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lerp(a, b, t): return a + (b - a) * t
def ease_out_cubic(t): t = clamp(t); return 1 - (1 - t) ** 3
def ease_in_cubic(t): t = clamp(t); return t ** 3
def ease_out_back(t, s=2.2):
    t = clamp(t) - 1
    return 1 + t * t * ((s + 1) * t + s)
def ease_out_elastic(t):
    t = clamp(t)
    if t in (0, 1): return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3) + 1


_font_cache = {}
def font(size):
    size = max(8, int(round(size / 4.0) * 4))
    if size not in _font_cache:
        for p in FONT_PATHS:
            if p and os.path.exists(p):
                _font_cache[size] = ImageFont.truetype(p, size); break
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def run(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


# ------------------------------- audio --------------------------------------
def load_audio(path, speed):
    """Decode audio yang dah dilajukan (atempo = pitch kekal)."""
    sr = 22050
    af = build_atempo(speed)
    raw = run(["ffmpeg", "-v", "error", "-i", path, "-af", af, "-ac", "1",
               "-ar", str(sr), "-f", "f32le", "-"])
    return np.frombuffer(raw, dtype=np.float32), sr


def build_atempo(speed):
    parts, s = [], speed
    while s > 2.0: parts.append("atempo=2.0"); s /= 2.0
    while s < 0.5: parts.append("atempo=0.5"); s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


def analyse(samples, sr, n_frames, n_bars=48):
    """Kira kekuatan bar spectrum, bass & beat untuk setiap frame."""
    win = 2048
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    edges = np.geomspace(50, 9000, n_bars + 1)
    idx = [(np.searchsorted(freqs, edges[i]), max(np.searchsorted(freqs, edges[i + 1]), np.searchsorted(freqs, edges[i]) + 1)) for i in range(n_bars)]
    bars = np.zeros((n_frames, n_bars), dtype=np.float32)
    padded = np.pad(samples, (win, win))
    for f in range(n_frames):
        c = int(f / FPS * sr) + win
        seg = padded[c - win // 2: c + win // 2] * window
        mag = np.abs(np.fft.rfft(seg))
        for b, (lo, hi) in enumerate(idx):
            bars[f, b] = mag[lo:hi].mean()
    bars = np.log1p(bars * 8)
    bars /= (np.percentile(bars, 97) + 1e-6)
    bars = np.clip(bars, 0, 1.2)
    # smoothing: attack laju, decay perlahan -> nampak "hidup"
    sm = np.zeros_like(bars)
    for f in range(n_frames):
        prev = sm[f - 1] if f else 0
        sm[f] = np.where(bars[f] > prev, lerp(prev, bars[f], 0.7), lerp(prev, bars[f], 0.18))
    bass = sm[:, :8].mean(axis=1)
    kick = np.zeros(n_frames, dtype=np.float32)
    for f in range(1, n_frames):
        jump = bass[f] - bass[f - 1]
        kick[f] = max(kick[f - 1] * 0.82, 1.0 if jump > 0.09 else 0.0)
    return sm, bass, kick


# ------------------------------- SRT ----------------------------------------
def parse_srt(path, speed):
    txt = open(path, encoding="utf-8-sig").read().replace("\r", "")
    out = []
    for block in re.split(r"\n\s*\n", txt.strip()):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", block)
        if not m: continue
        g = list(map(int, m.groups()))
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        lines = [l for l in block.split("\n") if "-->" not in l and not l.strip().isdigit()]
        text = " ".join(lines).strip()
        if text: out.append((s / speed, e / speed, text.upper()))  # masa ikut audio laju
    return out


# ------------------------------- lukisan ------------------------------------
def text_size(d, txt, fnt, stroke=0):
    b = d.textbbox((0, 0), txt, font=fnt, stroke_width=stroke)
    return b[2] - b[0], b[3] - b[1]


def draw_text(d, cx, cy, txt, size, fill, stroke=8, stroke_fill=(10, 8, 30), alpha=255):
    f = font(size)
    w, h = text_size(d, txt, f, stroke)
    d.text((cx - w / 2, cy - h / 2), txt, font=f, fill=fill + (alpha,),
           stroke_width=stroke, stroke_fill=stroke_fill + (alpha,))
    return w, h


def glitch_text(base, glow, cx, cy, txt, size, fill, intensity, rng, alpha=255):
    """Teks dengan pecahan RGB (chromatic aberration) + gegaran."""
    off = int(intensity * 26)
    jx = int(rng.integers(-off, off + 1)) if off else 0
    jy = int(rng.integers(-off // 3, off // 3 + 1)) if off else 0
    d = ImageDraw.Draw(base)
    if off > 2:
        f = font(size)
        w, h = text_size(d, txt, f, 0)
        for dx, col in ((-off, (255, 0, 80)), (off, (0, 240, 255))):
            d.text((cx - w / 2 + dx + jx, cy - h / 2 + jy), txt, font=f, fill=col + (int(alpha * 0.8),))
    draw_text(d, cx + jx, cy + jy, txt, size, fill, alpha=alpha)
    draw_text(ImageDraw.Draw(glow), cx + jx, cy + jy, txt, size, fill, stroke=14, stroke_fill=fill, alpha=int(alpha * 0.7))


def make_background(t, bass, S):
    """Gradien bergerak, dikira saiz kecil lalu di-upscale (laju & licin)."""
    w, h = int(W * S / 4), int(H * S / 4)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x /= w; y /= h
    a = 0.5 + 0.5 * np.sin(x * 3.0 + t * 0.7 + y * 2.0)
    b = 0.5 + 0.5 * np.sin(y * 4.0 - t * 0.9 + x * 1.5)
    glow_c = 0.55 * a[..., None] * C1 + 0.55 * b[..., None] * C2
    amount = (0.16 + 0.22 * bass) * (0.6 + 0.4 * np.sin(y * 3.14))[..., None]
    img = BG_DARK + glow_c * amount
    # vignette
    vig = 1 - 0.55 * (((x - 0.5) ** 2 + (y - 0.5) ** 2) * 2.2)[..., None]
    img = np.clip(img * vig, 0, 255).astype(np.uint8)
    return Image.fromarray(img).resize((int(W * S), int(H * S)), Image.BILINEAR).convert("RGBA")


def draw_particles(d, t, bass, S, rng_state):
    px, py, sp, sz, ph = rng_state
    for i in range(len(px)):
        y = (py[i] - t * sp[i]) % (H + 80) - 40
        x = px[i] + math.sin(t * 0.8 + ph[i]) * 30
        r = sz[i] * (1 + 0.8 * bass) * (0.6 + 0.4 * math.sin(t * 3 + ph[i]))
        col = tuple(int(v) for v in (C2 if i % 3 else C1)) + (int(120 + 100 * bass),)
        d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=col)


def draw_visualizer(d, glow_d, cx, cy, bars, bass, kick, t, S, appear=1.0):
    """Bulatan audio-reactive: bar jejari + gelang pulse."""
    n = len(bars)
    base_r = (250 + 38 * bass + 22 * kick) * appear
    for i in range(n):
        ang = (i / n) * 2 * math.pi + t * 0.35 - math.pi / 2
        mirror = abs(i - n / 2) / (n / 2)      # simetri kiri-kanan
        v = bars[min(n - 1, int((1 - mirror) * (n - 1)))] if False else bars[i]
        ln = (24 + 210 * v) * appear
        x0, y0 = cx + math.cos(ang) * base_r, cy + math.sin(ang) * base_r
        x1, y1 = cx + math.cos(ang) * (base_r + ln), cy + math.sin(ang) * (base_r + ln)
        k = i / n
        col = tuple(int(lerp(C1[j], C2[j], 0.5 + 0.5 * math.sin(k * 2 * math.pi))) for j in range(3))
        for dr in (d, glow_d):
            dr.line([x0 * S, y0 * S, x1 * S, y1 * S], fill=col + (255,), width=max(2, int(14 * S)))
    for r, wd, col in ((base_r - 18, 6, C3), (base_r - 52, 3, C2)):
        for dr in (d, glow_d):
            dr.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                       outline=tuple(int(v) for v in col) + (230,), width=max(1, int(wd * S)))
    # gelombang shock pada setiap kick
    if kick > 0.05:
        r = base_r + (1 - kick) * 320
        glow_d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S],
                       outline=tuple(int(v) for v in C1) + (int(200 * kick),), width=max(2, int(10 * S)))


def draw_bottom_bars(d, glow_d, bars, S, y_base=1560):
    n = len(bars)
    bw = (W - 120) / n
    for i, v in enumerate(bars):
        h = 14 + 190 * v
        x = 60 + i * bw
        col = tuple(int(lerp(C2[j], C1[j], i / n)) for j in range(3))
        for dr in (d, glow_d):
            dr.rounded_rectangle([x * S, (y_base - h) * S, (x + bw * 0.62) * S, y_base * S],
                                 radius=int(8 * S), fill=col + (255,))


def draw_follow_button(d, glow_d, cx, cy, scale, t, S, label="+ FOLLOW"):
    pulse = 1 + 0.05 * math.sin(t * 9)
    w, h = 520 * scale * pulse, 150 * scale * pulse
    box = [(cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S]
    for dr, ex in ((glow_d, 14), (d, 0)):
        dr.rounded_rectangle([box[0] - ex * S, box[1] - ex * S, box[2] + ex * S, box[3] + ex * S],
                             radius=int(h / 2 * S), fill=(254, 44, 85, 255))
    f = font(70 * scale * pulse * S)
    tw, th = text_size(d, label, f)
    d.text((cx * S - tw / 2, cy * S - th / 2 - 4 * S), label, font=f, fill=(255, 255, 255, 255))


def draw_cursor(d, x, y, S, press):
    """Kursor tangan klik butang."""
    s = 1 - 0.18 * press
    pts = [(0, 0), (0, 70), (18, 55), (32, 90), (48, 83), (34, 50), (58, 50)]
    pts = [((x + px * s) * S, (y + py * s) * S) for px, py in pts]
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(10, 8, 30, 255))


# ------------------------------- frame --------------------------------------
class Scene:
    def __init__(self, duration, bars, bass, kick, captions, handle, S):
        self.dur, self.bars, self.bass, self.kick = duration, bars, bass, kick
        self.captions, self.handle, self.S = captions, handle, S
        rng = np.random.default_rng(7)
        n = 70
        self.particles = (rng.uniform(0, W, n), rng.uniform(0, H + 80, n), rng.uniform(25, 120, n),
                          rng.uniform(3, 11, n), rng.uniform(0, 6.28, n))

    def frame(self, f):
        S, t = self.S, f / FPS
        bars, bass, kick = self.bars[f], float(self.bass[f]), float(self.kick[f])
        rng = np.random.default_rng(f)

        img = make_background(t, bass, S)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d, gd = ImageDraw.Draw(layer), ImageDraw.Draw(glow)
        draw_particles(d, t, bass, S, self.particles)

        in_hook = t < HOOK_END
        cta_start = self.dur - CTA_SECONDS
        in_cta = t >= cta_start

        # ---- visualizer tengah (masuk selepas hook) ----
        vis_in = ease_out_back((t - (HOOK_END - 0.3)) / 0.7) if t > HOOK_END - 0.3 else 0
        cta_p = ease_out_cubic((t - cta_start) / 0.6) if in_cta else 0
        vis_y = lerp(840, 520, cta_p)
        vis_scale = lerp(1.0, 0.72, cta_p)
        if vis_in > 0:
            draw_visualizer(d, gd, W / 2, vis_y, bars, bass, kick, t, S, appear=clamp(vis_in, 0, 1.15) * vis_scale)
            draw_bottom_bars(d, gd, bars, S, y_base=1620 if not in_cta else 1700)

        # ---- HOOK ----
        if in_hook:
            self.draw_hook(d, gd, layer, glow, t, kick, rng)

        # ---- handle atas ----
        if t > HOOK_END:
            a = int(255 * ease_out_cubic((t - HOOK_END) / 0.5))
            draw_text(d, W / 2 * S, 150 * S, self.handle, 54 * S, (255, 255, 255), stroke=int(6 * S), alpha=a)

        # ---- caption ----
        if not in_hook and not in_cta:
            self.draw_caption(d, gd, t)

        # ---- follow pop tengah video ----
        ms = self.dur * MID_FOLLOW_AT
        if ms <= t < ms + MID_FOLLOW_DUR and not in_cta:
            lt = t - ms
            slide = ease_out_back(lt / 0.5) * (1 - ease_in_cubic((lt - (MID_FOLLOW_DUR - 0.4)) / 0.4))
            draw_follow_button(d, gd, lerp(W + 300, W - 330, slide), 300, 0.6, t, S)

        # ---- CTA hujung ----
        if in_cta:
            self.draw_cta(d, gd, t - cta_start, t)

        # ---- progress bar ----
        p = clamp(t / self.dur)
        d.rectangle([0, 0, W * S, 14 * S], fill=(255, 255, 255, 40))
        d.rectangle([0, 0, W * p * S, 14 * S], fill=tuple(int(v) for v in C3) + (255,))

        # ---- gabung: glow blur + layer ----
        small = glow.resize((max(1, glow.width // 4), max(1, glow.height // 4)), Image.BILINEAR)
        blur = small.filter(ImageFilter.GaussianBlur(max(2, int(7 * S)))).resize(glow.size, Image.BILINEAR)
        out = Image.alpha_composite(img, blur)
        out = Image.alpha_composite(out, layer).convert("RGB")

        # flash putih pada frame pertama + kick halus
        flash = max(0.0, 1 - t / 0.18) * 0.9 + kick * 0.06
        if flash > 0.01:
            out = Image.blend(out, Image.new("RGB", out.size, (255, 255, 255)), min(flash, 1))
        return out

    def draw_hook(self, d, gd, layer, glow, t, kick, rng):
        S = self.S
        out_p = ease_in_cubic((t - (HOOK_END - 0.35)) / 0.35)      # keluar ke atas
        ys = [700, 930, 1160]
        for (txt, t0, col), y in zip(HOOK, ys):
            lt = t - t0
            if lt < 0: continue
            pop = ease_out_back(lt / 0.38, 2.8)
            # saiz asas dikira supaya teks muat lebar skrin (maks ~880px), pop overshoot masih selamat
            probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
            pw, _ = text_size(probe, txt, font(100), 8)
            size = min(150, 880 / pw * 100) * pop
            glitch = max(0.0, 1 - lt / 0.55) + (0.6 if rng.random() < 0.04 else 0) + kick * 0.25
            glitch_text(layer, glow, W / 2 * S, (y - out_p * 900) * S, txt, size * S, col, glitch * 0.9, rng,
                        alpha=int(255 * (1 - out_p)))
            # garis wipe bawah teks
            lw = ease_out_cubic(lt / 0.5) * 760
            d.rectangle([(W / 2 - lw / 2) * S, (y + 95 - out_p * 900) * S, (W / 2 + lw / 2) * S, (y + 108 - out_p * 900) * S],
                        fill=tuple(int(v) for v in col) + (255,))
        # slash diagonal laju di permulaan
        if t < 0.5:
            x = lerp(-300, W + 300, ease_out_cubic(t / 0.5))
            gd.polygon([((x - 60) * S, 0), ((x + 60) * S, 0), ((x - 160) * S, H * S), ((x - 280) * S, H * S)],
                       fill=tuple(int(v) for v in C2) + (200,))
        # bar atas/bawah gaya sinematik
        bh = lerp(260, 0, ease_out_cubic((t - 0.05) / 0.35))
        d.rectangle([0, 0, W * S, bh * S], fill=(0, 0, 0, 255))
        d.rectangle([0, (H - bh) * S, W * S, H * S], fill=(0, 0, 0, 255))

    def draw_caption(self, d, gd, t):
        S = self.S
        for s, e, txt in self.captions:
            if s <= t < e:
                lt = t - s
                pop = ease_out_back(lt / 0.22, 3.0)
                words, lines, cur = txt.split(), [], ""
                for w in words:                        # pecah baris (maks ~16 huruf)
                    if len(cur) + len(w) + 1 > 16 and cur: lines.append(cur); cur = w
                    else: cur = (cur + " " + w).strip()
                lines.append(cur)
                y0 = 1290 - (len(lines) - 1) * 50
                for i, line in enumerate(lines[:3]):
                    hi = (i == len(lines) - 1)
                    draw_text(d, W / 2 * S, (y0 + i * 100) * S, line, 84 * pop * S,
                              tuple(int(v) for v in (C3 if hi else (255, 255, 255))), stroke=int(10 * S))
                return

    def draw_cta(self, d, gd, lt, t):
        S = self.S
        # tajuk
        for i, line in enumerate(CTA_LINES):
            p = ease_out_back((lt - 0.15 - i * 0.18) / 0.45, 2.6)
            if p <= 0: continue
            col = (255, 255, 255) if i == 0 else tuple(int(v) for v in C3)
            draw_text(d, W / 2 * S, (1010 + i * 120) * S, line, (92 if i == 0 else 100) * p * S, col, stroke=int(10 * S))
        # butang follow
        bp = ease_out_elastic((lt - 0.7) / 0.9)
        if bp > 0:
            draw_follow_button(d, gd, W / 2, 1340, bp * 1.25, t, S, "+ FOLLOW")
        # kursor klik
        ct = lt - 1.8
        if ct > 0:
            cp = ease_out_cubic(ct / 0.6)
            press = clamp(math.sin(ct * 6)) if ct > 0.7 else 0
            draw_cursor(d, lerp(W - 60, W / 2 + 120, cp), lerp(1700, 1360, cp), S, press)
            if press > 0.8:
                r = 80 + 120 * (ct * 6 % 3.14)
                gd.ellipse([(W / 2 + 120 - r) * S, (1360 - r) * S, (W / 2 + 120 + r) * S, (1360 + r) * S],
                           outline=(255, 255, 255, 180), width=int(6 * S))
        # hati terbang
        for i in range(8):
            ph = (lt * 0.55 + i / 8) % 1
            x = W / 2 + math.sin(i * 2.1 + lt * 2) * 330
            y = 1560 - ph * 700
            r = 22 + 14 * math.sin(i)
            col = tuple(int(v) for v in C1) + (int(255 * (1 - ph)),)
            d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=col)


# ------------------------------- utama --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--output", default="video_siap.mp4")
    ap.add_argument("--srt", help="fail sarikata .srt (masa ikut audio asal)")
    ap.add_argument("--handle", default=HANDLE)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--scale", type=float, default=1.0, help="0.5 = preview laju (540x960)")
    ap.add_argument("--t0", type=float, default=0.0, help="mula render (saat, utk preview)")
    ap.add_argument("--t1", type=float, default=None, help="tamat render (saat, utk preview)")
    a = ap.parse_args()

    samples, sr = load_audio(a.audio, a.speed)
    total = len(samples) / sr
    n_frames = int(total * FPS)
    print(f"Audio lepas dilajukan {a.speed}x: {total:.1f}s -> {n_frames} frame")

    bars, bass, kick = analyse(samples, sr, n_frames)
    caps = parse_srt(a.srt, a.speed) if a.srt else []
    scene = Scene(total, bars, bass, kick, caps, a.handle, a.scale)

    f0 = int(a.t0 * FPS)
    f1 = min(n_frames, int((a.t1 if a.t1 else total) * FPS))
    ow, oh = int(W * a.scale) // 2 * 2, int(H * a.scale) // 2 * 2
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-r", str(FPS), "-i", "-",
           "-ss", f"{f0 / FPS:.3f}", "-i", a.audio,
           "-af", build_atempo(a.speed), "-t", f"{(f1 - f0) / FPS:.3f}",
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", a.output]
    # NOTA: -ss pada input audio dikira ikut masa ASAL, jadi darab dengan speed
    cmd[cmd.index("-ss") + 1] = f"{(f0 / FPS) * a.speed:.3f}"
    cmd[cmd.index("-t") + 1] = f"{(f1 - f0) / FPS:.3f}"
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(f0, f1):
            fr = scene.frame(f)
            if fr.size != (ow, oh): fr = fr.resize((ow, oh))
            pipe.stdin.write(fr.tobytes())
            if (f - f0) % 30 == 0:
                print(f"\r  {f - f0}/{f1 - f0} frame", end="", flush=True)
    finally:
        pipe.stdin.close(); pipe.wait()
    print(f"\nSiap! -> {a.output}")


if __name__ == "__main__":
    main()
