"""Dark "control room" theme for the dashboard: colours and CSS."""
from __future__ import annotations

from ..dashboard_data import STATUS_COLORS

STATUS_COLORS_HEX = {k: "#%02x%02x%02x" % tuple(v) for k, v in STATUS_COLORS.items()}
CYAN, VIOLET, TEXT, MUTED, PANEL, GRID = "#00e5ff", "#8c52ff", "#e6f1ff", "#8aa0c2", "#0b1426", "#1b2a44"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');
:root { --cyan:#00e5ff; --violet:#8c52ff; --text:#e6f1ff; --muted:#8aa0c2; --panel:rgba(11,20,38,.72);
        --border:rgba(0,229,255,.22); }
.stApp { background:
    radial-gradient(1100px 520px at 8% -8%, rgba(0,229,255,.10), transparent 60%),
    radial-gradient(900px 480px at 105% 0%, rgba(140,82,255,.10), transparent 60%),
    linear-gradient(180deg, #050a14 0%, #060d1c 100%); color: var(--text); font-family: 'Inter', sans-serif; }
.stApp::before { content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image: linear-gradient(rgba(0,229,255,.035) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(0,229,255,.035) 1px, transparent 1px);
    background-size: 44px 44px; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1700px; }
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; letter-spacing: .04em; }
section[data-testid="stSidebar"] { background: rgba(5,10,20,.92); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

.ar-header { display:flex; justify-content:space-between; align-items:center; gap:18px; padding:16px 22px;
    margin-bottom:14px; border:1px solid var(--border); border-radius:16px; background:var(--panel);
    backdrop-filter: blur(10px); box-shadow: 0 0 30px rgba(0,229,255,.08), inset 0 0 40px rgba(0,229,255,.03); }
.ar-title { font-family:'Orbitron',sans-serif; font-weight:900; font-size:21px; letter-spacing:.07em; white-space:nowrap;
    background: linear-gradient(90deg, #00e5ff, #8c52ff 60%, #ff2e93); -webkit-background-clip:text;
    background-clip:text; color:transparent; }
.ar-sub { color:var(--muted); font-size:13px; letter-spacing:.12em; text-transform:uppercase; margin-top:4px; }
.ar-meta { display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
.ar-chip { font-family:'JetBrains Mono',monospace; font-size:12px; padding:6px 12px; border-radius:999px;
    border:1px solid var(--border); background:rgba(0,229,255,.06); color:var(--text); white-space:nowrap; }
.ar-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; background:var(--c);
    box-shadow:0 0 10px var(--c); animation: arBlink 1.8s infinite; }
@keyframes arBlink { 0%,100% { opacity:1 } 50% { opacity:.35 } }

.ar-kpi { position:relative; padding:12px 14px; border-radius:14px; background:var(--panel); min-height:92px;
    border:1px solid rgba(255,255,255,.06); border-left:3px solid var(--c);
    box-shadow: 0 0 22px color-mix(in srgb, var(--c) 22%, transparent); backdrop-filter: blur(8px); }
.ar-kpi .lbl { font-family:'Orbitron',sans-serif; font-size:11px; letter-spacing:.14em; color:var(--muted); }
.ar-kpi .val { font-family:'JetBrains Mono',monospace; font-size:32px; font-weight:600; color:var(--c);
    text-shadow:0 0 14px color-mix(in srgb, var(--c) 60%, transparent); line-height:1.2; }
.ar-kpi .sub { font-size:11px; color:var(--muted); }
.ar-pulse { animation: arPulse 1.6s infinite; }
@keyframes arPulse { 0% { box-shadow:0 0 0 0 var(--c) } 70% { box-shadow:0 0 0 12px transparent }
    100% { box-shadow:0 0 0 0 transparent } }

.ar-panel { padding:14px 16px; border-radius:14px; background:var(--panel); border:1px solid var(--border);
    backdrop-filter: blur(8px); margin-bottom:12px; }
.ar-panel h4 { margin:0 0 10px 0; font-size:13px; letter-spacing:.14em; color:var(--cyan); }
.ar-feed { max-height:420px; overflow-y:auto; padding-right:4px; }
.ar-feed-item { padding:8px 10px; margin-bottom:6px; border-radius:10px; background:rgba(255,255,255,.025);
    border-left:3px solid var(--c); font-size:12.5px; }
.ar-feed-item .t { font-family:'JetBrains Mono',monospace; color:var(--muted); font-size:11px; }
.ar-feed-item .k { font-family:'Orbitron',sans-serif; font-size:10px; letter-spacing:.12em; color:var(--c); }
.ar-bar { margin:7px 0; font-size:12px; }
.ar-bar .row { display:flex; justify-content:space-between; color:var(--text); }
.ar-bar .track { height:7px; border-radius:6px; background:#13213a; overflow:hidden; margin-top:4px; }
.ar-bar .fill { height:100%; border-radius:6px; background:linear-gradient(90deg, var(--violet), var(--cyan));
    box-shadow:0 0 10px rgba(0,229,255,.6); }
.ar-pill { display:inline-block; padding:3px 12px; border-radius:999px; font-family:'Orbitron',sans-serif;
    font-size:11px; letter-spacing:.12em; color:var(--c); border:1px solid var(--c);
    background: color-mix(in srgb, var(--c) 14%, transparent); box-shadow:0 0 12px color-mix(in srgb, var(--c) 40%, transparent); }
.ar-legend span { margin-right:14px; font-size:12px; color:var(--muted); }
.ar-legend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:5px;
    background:var(--c); box-shadow:0 0 8px var(--c); vertical-align:middle; }

.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:rgba(11,20,38,.7); border:1px solid rgba(0,229,255,.16);
    border-radius:12px 12px 0 0; padding:8px 18px; font-family:'Orbitron',sans-serif; letter-spacing:.08em; }
.stTabs [aria-selected="true"] { border-color:var(--cyan) !important; box-shadow:0 0 16px rgba(0,229,255,.35);
    color:var(--cyan) !important; }
div[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:12px; }
div[data-testid="stForm"] { border:1px solid var(--border); border-radius:14px; background:var(--panel); }
</style>
"""


def altair_dark(chart):
    """Apply the dark theme to an Altair chart."""
    return (chart.configure(background="transparent")
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor="#2a3d5c",
                            tickColor="#2a3d5c", labelFont="JetBrains Mono", titleFont="Inter")
            .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFont="Inter")
            .configure_view(stroke=None)
            .configure_title(color=TEXT, font="Orbitron", fontSize=13))
