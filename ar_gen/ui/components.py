"""SVG and HTML building blocks (gauges, radar, KPI tiles, pills, feed items). All text is escaped."""
from __future__ import annotations

import html
import math
import zlib

from .theme import CYAN, MUTED, STATUS_COLORS_HEX, TEXT


def _esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _uid(*parts) -> str:
    return "g" + format(zlib.crc32("|".join(map(str, parts)).encode()), "x")


def _arc(cx, cy, r, a0, a1) -> str:
    x0, y0 = cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0))
    x1, y1 = cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1))
    large = 1 if (a1 - a0) > 180 else 0
    return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}"


def gauge_svg(value, max_value: float, label: str, color: str, unit: str = "", decimals: int = 0,
              size: int = 190) -> str:
    """270-degree neon gauge; value is clamped to [0, max_value]; None shows '-'."""
    v = _num(value)
    shown = "-" if v is None else f"{min(max(v, 0.0), max_value):.{decimals}f}"
    frac = 0.0 if v is None else min(max(v / max_value, 0.0), 1.0)
    uid = _uid("gauge", label, value, color)
    a0, sweep = 135.0, 270.0
    track = _arc(100, 100, 78, a0, a0 + sweep)
    value_arc = _arc(100, 100, 78, a0, a0 + max(sweep * frac, 0.01)) if frac > 0 else ""
    ticks = "".join(
        f'<line x1="{100 + 92 * math.cos(math.radians(a0 + sweep * t / 10)):.1f}" '
        f'y1="{100 + 92 * math.sin(math.radians(a0 + sweep * t / 10)):.1f}" '
        f'x2="{100 + 98 * math.cos(math.radians(a0 + sweep * t / 10)):.1f}" '
        f'y2="{100 + 98 * math.sin(math.radians(a0 + sweep * t / 10)):.1f}" stroke="#2a3d5c" stroke-width="2"/>'
        for t in range(11))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="{size}" height="{size}">'
            f'<defs><filter id="{uid}" x="-30%" y="-30%" width="160%" height="160%">'
            f'<feGaussianBlur stdDeviation="3.5" result="b"/><feMerge><feMergeNode in="b"/>'
            f'<feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
            f'{ticks}<path d="{track}" fill="none" stroke="#13213a" stroke-width="14" stroke-linecap="round"/>'
            + (f'<path d="{value_arc}" fill="none" stroke="{_esc(color)}" stroke-width="14" stroke-linecap="round" '
               f'filter="url(#{uid})"/>' if value_arc else "")
            + f'<text x="100" y="108" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="38" '
              f'font-weight="600" fill="{_esc(color)}">{_esc(shown)}</text>'
              f'<text x="100" y="132" text-anchor="middle" font-family="Inter, sans-serif" font-size="12" '
              f'fill="{MUTED}">{_esc(unit)}</text>'
              f'<text x="100" y="178" text-anchor="middle" font-family="Orbitron, sans-serif" font-size="12" '
              f'letter-spacing="2" fill="{TEXT}">{_esc(label)}</text></svg>')


def radar_svg(scores: dict, color: str = CYAN, size: int = 320) -> str:
    """Radar of factor scores (0-100) - one axis per factor; missing values plot at 0."""
    names = list(scores)
    n = max(len(names), 3)
    cx = cy = 160
    r = 105

    def pt(i, frac):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * frac * math.cos(ang), cy + r * frac * math.sin(ang)

    rings = "".join(
        '<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, f) for i in range(n)))
        + f'" fill="none" stroke="#1d2f4d" stroke-width="1"/>' for f in (0.25, 0.5, 0.75, 1.0))
    axes = "".join(f'<line x1="{cx}" y1="{cy}" x2="{pt(i, 1)[0]:.1f}" y2="{pt(i, 1)[1]:.1f}" stroke="#1d2f4d"/>'
                   for i in range(n))
    vals = [min(max(_num(scores[k]) or 0.0, 0.0), 100.0) for k in names]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, v / 100) for i, v in enumerate(vals)))
    dots = "".join(f'<circle cx="{pt(i, v / 100)[0]:.1f}" cy="{pt(i, v / 100)[1]:.1f}" r="3.5" fill="{_esc(color)}"/>'
                   for i, v in enumerate(vals))
    labels = []
    for i, k in enumerate(names):
        x, y = pt(i, 1.22)
        anchor = "middle" if abs(x - cx) < 10 else ("start" if x > cx else "end")
        score = _num(scores[k])
        labels.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Orbitron, sans-serif" '
                      f'font-size="10.5" letter-spacing="1" fill="{TEXT}">{_esc(k)}</text>'
                      f'<text x="{x:.1f}" y="{y + 13:.1f}" text-anchor="{anchor}" font-family="JetBrains Mono, monospace" '
                      f'font-size="10" fill="{MUTED}">{"-" if score is None else f"{score:.0f}"}</text>')
    uid = _uid("radar", *names, *vals)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-50 -12 420 344" width="{size}" height="{size}">'
            f'<defs><filter id="{uid}"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge>'
            f'<feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
            f'{rings}{axes}<polygon points="{poly}" fill="{_esc(color)}" fill-opacity="0.18" stroke="{_esc(color)}" '
            f'stroke-width="2" filter="url(#{uid})"/>{dots}{"".join(labels)}</svg>')


def kpi_tile(label, value, color: str, sub: str = "", pulse: bool = False) -> str:
    cls = "ar-kpi ar-pulse" if pulse else "ar-kpi"
    return (f'<div class="{cls}" style="--c:{_esc(color)}"><div class="lbl">{_esc(label)}</div>'
            f'<div class="val">{_esc(value)}</div><div class="sub">{_esc(sub)}</div></div>')


def status_pill(status) -> str:
    color = STATUS_COLORS_HEX.get(str(status), MUTED)
    return f'<span class="ar-pill" style="--c:{color}">{_esc(status)}</span>'


def feed_item(ts_text, kind, asset, detail, color: str) -> str:
    return (f'<div class="ar-feed-item" style="--c:{_esc(color)}"><span class="k">{_esc(kind)}</span> '
            f'<b>{_esc(asset)}</b> <span class="t">{_esc(ts_text)}</span><br/>{_esc(detail)}</div>')


def branch_bar(name, avg_hi, n_assets: int, n_problem: int) -> str:
    v = _num(avg_hi)
    width = 0 if v is None else min(max(v, 0), 100)
    shown = "-" if v is None else f"{v:.1f}"
    return (f'<div class="ar-bar"><div class="row"><span>{_esc(name)}</span><span>HI {shown} · '
            f'{int(n_assets)} AR · {int(n_problem)} alert</span></div><div class="track">'
            f'<div class="fill" style="width:{width:.0f}%"></div></div></div>')


def legend_html() -> str:
    items = "".join(f'<span><i style="--c:{c}"></i>{_esc(s)}</span>' for s, c in STATUS_COLORS_HEX.items())
    return f'<div class="ar-legend">{items}<span>· column height = risk</span></div>'
