"""LECO · AR Health Command Centre: futuristic Streamlit dashboard.

Start with:  python main.py dashboard      (opens http://localhost:8501)
"""
from __future__ import annotations

import html
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root, however the app is started

import altair as alt                    # noqa: E402
import pandas as pd                     # noqa: E402
import streamlit as st                  # noqa: E402

from ar_gen import pipeline             # noqa: E402
from ar_gen.config import load_config, ml_enabled   # noqa: E402
from ar_gen.dashboard_data import (STATUS_MEANING, STATUS_ORDER, alert_feed, branch_summary,  # noqa: E402
                                   branches, daily_summary_frame, dl_history, dl_model_status, dl_snapshot,
                                   fleet_snapshot, hi_history, latest_variables, ml_history, model_status,
                                   recent_events, reset_log, with_dl)
from ar_gen.db.engine import create_all  # noqa: E402
from ar_gen.sim.timeutil import dt_to_ms, zone   # noqa: E402
from ar_gen.ui.components import (branch_bar, feed_item, gauge_svg, kpi_tile, legend_html,  # noqa: E402
                                  radar_svg, status_pill)
from ar_gen.ui.map3d import build_deck   # noqa: E402
from ar_gen.ui.theme import CSS, CYAN, MUTED, STATUS_COLORS_HEX, VIOLET, altair_dark   # noqa: E402
from ar_gen.variables import RESET_GROUPS        # noqa: E402

st.set_page_config(page_title="LECO AR Command Centre", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

CFG = load_config()
TZ = CFG["timezone"]
URL = CFG["database_url"]
ML_ON = ml_enabled(CFG)          # config.yaml ml.enabled: false hides every ML part of the page
PART_LABELS = {"ALL": "All parts", "BATTERY": "Battery", "SWITCHING": "OSM switch (mechanism / contacts)",
               "COMMS": "Communications (modem / antenna)", "CONTROLLER": "Controller (RC modules)"}
FACTORS = {"f_battery": "BATTERY", "f_faults": "FAULTS", "f_switching": "SWITCHING", "f_comms": "COMMS",
           "f_controlling": "CONTROL", "f_aging": "AGING", "f_environment": "ENVIRON"}
CLASSES = {k: float(v) for k, v in CFG["health_index"]["classes"].items()}   # lower HI limit of each class
HEALTHY_MIN = float(CFG["health_index"]["healthy_min"])
KPI_HINT = {"OFFLINE": "no data for 60 min", "FAULTY": "failed, awaiting repair", "LOCKOUT": "locked out after fault",
            "CRITICAL": f"HI below {CLASSES['POOR']:g}", "POOR": f"HI {CLASSES['POOR']:g} - {CLASSES['FAIR']:g}",
            "FAIR": f"HI {CLASSES['FAIR']:g} - {CLASSES['GOOD']:g}", "GOOD": f"HI {CLASSES['GOOD']:g} or more"}
FEED_COLORS = {"FAILURE": STATUS_COLORS_HEX["FAULTY"], "LOCKOUT": STATUS_COLORS_HEX["LOCKOUT"],
               "REPAIR": STATUS_COLORS_HEX["GOOD"], "MAINTENANCE": CYAN, "RESET": "#8c52ff"}
FACTOR_VARS = {
    "Battery": ["v01", "v02", "v03", "v04", "v05", "v06", "v07", "v08", "v09", "v10"],
    "Faults": ["v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20"],
    "Switching": ["v21", "v22", "v23", "v24", "v25", "v26", "v27"],
    "Comms": ["v28", "v29", "v30", "v31", "v32"],
    "Controlling": ["v33", "v34", "v35", "v36"], "Aging": ["v37", "v38", "v39"], "Environment": ["v40", "v41", "v42"],
}
ML_PINK = "#ff2e93"
MODEL_COLORS = {"LightGBM": ML_PINK, "Deep learning": VIOLET}


# ------------------------------------------------------------------ data (cached 30 s)
@st.cache_resource(show_spinner=False)
def _engine(url: str):
    eng = pipeline.get_engine(CFG)
    create_all(eng)                     # databases made before the dashboard get its new tables
    return eng


@st.cache_data(ttl=30, show_spinner=False)
def _snapshot(url: str):
    ts, snap = fleet_snapshot(_engine(url), CFG)
    if not ML_ON or not model_status(CFG).get("installed"):   # ML off / model unusable: old values are not current
        snap = snap.assign(ml_p=float("nan"), ml_class=None, ml_healthy=None, ml_component=None)
    return ts, snap


@st.cache_data(ttl=30, show_spinner=False)
def _history(url: str, asset: str):
    return hi_history(_engine(url), asset, days=30, tz=TZ)


@st.cache_data(ttl=30, show_spinner=False)
def _ml_history(url: str, asset: str):
    return ml_history(_engine(url), asset, days=30, tz=TZ)


@st.cache_data(ttl=30, show_spinner=False)
def _dl_snapshot(url: str):
    return dl_snapshot(_engine(url))


@st.cache_data(ttl=30, show_spinner=False)
def _dl_history(url: str, asset: str):
    return dl_history(_engine(url), asset, days=30, tz=TZ)


@st.cache_data(ttl=30, show_spinner=False)
def _events(url: str, asset: str):
    return recent_events(_engine(url), asset, days=30, tz=TZ)


@st.cache_data(ttl=30, show_spinner=False)
def _feed(url: str):
    return alert_feed(_engine(url), hours=48, tz=TZ)


@st.cache_data(ttl=30, show_spinner=False)
def _daily(url: str):
    return daily_summary_frame(_engine(url), days=30)


@st.cache_data(ttl=30, show_spinner=False)
def _variables(url: str, asset: str):
    return latest_variables(_engine(url), asset)


@st.cache_data(ttl=30, show_spinner=False)
def _resets(url: str):
    return reset_log(_engine(url), tz=TZ)


def _local(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert(TZ)


def _fmt(v, nd=1, unit=""):
    return "-" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{nd}f}{unit}"


def _sorted(snap: pd.DataFrame) -> pd.DataFrame:
    rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    return snap.assign(_r=snap["status"].map(rank), _hi=snap["health_index"].fillna(-1)) \
        .sort_values(["_r", "_hi", "asset_id"]).drop(columns=["_r", "_hi"])


def _filtered(snap: pd.DataFrame) -> pd.DataFrame:
    s = st.session_state
    out = snap[snap["area_name"].isin(s.get("f_branch", list(snap["area_name"].unique())))
               & snap["status"].isin(s.get("f_status", STATUS_ORDER))]
    term = (s.get("f_search") or "").strip()
    if term:
        out = out[out["asset_id"].str.contains(term, case=False, regex=False)]
    return out


# ------------------------------------------------------------------ header
def header(ts, snap: pd.DataFrame, model: dict, dl: dict) -> None:
    now = datetime.now(timezone.utc)
    if ts is not None:
        age = (now.replace(tzinfo=None) - pd.Timestamp(ts).to_pydatetime()).total_seconds() / 60
        beat = STATUS_COLORS_HEX["GOOD"] if age < 20 else (STATUS_COLORS_HEX["POOR"] if age < 60 else
                                                         STATUS_COLORS_HEX["CRITICAL"])
        data_txt = f"DATA {_local(ts):%H:%M} · {age:.0f} MIN AGO"
    else:
        beat, data_txt = STATUS_COLORS_HEX["OFFLINE"], "NO DATA"
    model_chips = ""
    if ML_ON:
        if model.get("installed"):
            model_txt = f"ML · {html.escape(model['version'])}"
        elif model.get("error"):
            model_txt = f'<span class="ar-dot" style="--c:{STATUS_COLORS_HEX["CRITICAL"]}"></span>ML · MODEL ERROR'
        else:
            model_txt = "ML · NOT INSTALLED"
        if dl.get("installed"):
            model_txt += f'</span><span class="ar-chip">DL · {html.escape(dl["version"])}'
        elif dl.get("error"):
            model_txt += (f'</span><span class="ar-chip"><span class="ar-dot" style="--c:{STATUS_COLORS_HEX["CRITICAL"]}">'
                          '</span>DL · MODEL ERROR')
        model_chips = f'<span class="ar-chip">{model_txt}</span>'
    subtitle = "real-time health index &amp; ML failure risk" if ML_ON else "real-time health index"
    st.markdown(
        f'<div class="ar-header"><div><div class="ar-title">LECO · AR HEALTH COMMAND CENTRE</div>'
        f'<div class="ar-sub">Lanka Electricity Company · {len(snap)} auto reclosers · {subtitle}</div></div>'
        f'<div class="ar-meta"><span class="ar-chip">{now.astimezone(zone(TZ)):%a %d %b · %H:%M} LOCAL</span>'
        f'<span class="ar-chip"><span class="ar-dot" style="--c:{beat}"></span>{data_txt}</span>'
        f'{model_chips}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ charts
def hi_chart(rows: pd.DataFrame, ml_rows: pd.DataFrame, threshold: float | None):
    bands = pd.DataFrame({"y0": [0, CLASSES["POOR"], CLASSES["FAIR"], CLASSES["GOOD"]],
                          "y1": [CLASSES["POOR"], CLASSES["FAIR"], CLASSES["GOOD"], 100],
                          "c": [STATUS_COLORS_HEX[k] for k in ("CRITICAL", "POOR", "FAIR", "GOOD")]})
    band = alt.Chart(bands).mark_rect(opacity=0.07).encode(y="y0:Q", y2="y1:Q", color=alt.Color("c:N", scale=None))
    base = alt.Chart(rows).encode(x=alt.X("ts_local:T", title=None))
    area = base.mark_area(opacity=0.14, color=CYAN, line=False).encode(y=alt.Y("health_index:Q", scale=alt.Scale(domain=[0, 100])))
    line = base.mark_line(color=CYAN, strokeWidth=1.8).encode(
        y=alt.Y("health_index:Q", title="Health Index", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("ts_local:T", title="time"), alt.Tooltip("health_index:Q", format=".1f", title="HI")])
    chart = alt.layer(band, area, line)
    if ml_rows is not None and not ml_rows.empty:
        ml = ml_rows.assign(ml_pct=ml_rows["p_fail_30d"] * 100)
        ml_line = alt.Chart(ml).mark_line(color="#ff2e93", strokeWidth=1.5, strokeDash=[5, 3]).encode(
            x="ts_local:T", y=alt.Y("ml_pct:Q", title="ML failure risk %", scale=alt.Scale(domain=[0, 100]),
                                    axis=alt.Axis(orient="right", titleColor="#ff2e93")),
            tooltip=[alt.Tooltip("ml_pct:Q", format=".1f", title="ML risk %")])
        chart = alt.layer(chart, ml_line).resolve_scale(y="independent")
    return altair_dark(chart.properties(height=290))


def models_chart(rows: pd.DataFrame, risks: dict):
    """HI (left axis) and the 30-day failure risk of each model (right axis) for one recloser."""
    hi = alt.Chart(rows).mark_line(color=CYAN, strokeWidth=1.6).encode(
        x=alt.X("ts_local:T", title=None),
        y=alt.Y("health_index:Q", title="Health Index", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("ts_local:T", title="time"), alt.Tooltip("health_index:Q", format=".1f", title="HI")])
    parts = [r.assign(model=name, risk_pct=r["p_fail_30d"] * 100)[["ts_local", "risk_pct", "model"]]
             for name, r in risks.items() if r is not None and not r.empty]
    if not parts:
        return altair_dark(hi.properties(height=380))
    lines = alt.Chart(pd.concat(parts, ignore_index=True)).mark_line(strokeWidth=1.5).encode(
        x="ts_local:T",
        y=alt.Y("risk_pct:Q", title="failure risk % (30 days)", scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(orient="right")),
        color=alt.Color("model:N", scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["model", alt.Tooltip("ts_local:T", title="time"),
                 alt.Tooltip("risk_pct:Q", format=".1f", title="risk %")])
    return altair_dark(alt.layer(hi, lines).resolve_scale(y="independent").properties(height=380))


def daily_chart(rows: pd.DataFrame):
    rows = rows.assign(day=pd.to_datetime(rows["day"]), state=rows["hi_healthy"].map({True: "healthy", False: "not healthy"}))
    chart = alt.Chart(rows).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("day:T", title=None), y=alt.Y("avg_hi:Q", title="Daily average HI", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("state:N", scale=alt.Scale(domain=["healthy", "not healthy"],
                                                   range=[STATUS_COLORS_HEX["GOOD"], STATUS_COLORS_HEX["CRITICAL"]]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("day:T"), alt.Tooltip("avg_hi:Q", format=".1f"), alt.Tooltip("min_hi:Q", format=".1f"),
                 alt.Tooltip("unhealthy_hours:Q", format=".1f", title="hours unhealthy")])
    return altair_dark(chart.properties(height=220))


# ------------------------------------------------------------------ page
ts, snap = _snapshot(URL)
model = model_status(CFG) if ML_ON else {}
dl = dl_model_status(CFG) if ML_ON else {}
header(ts, snap, model, dl)
if ts is None or snap.empty:
    st.info("No data yet - run backfill, then start the generator (python main.py service install, or service run)")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙ FILTERS")
    all_branches = sorted(snap["area_name"].unique())
    st.multiselect("LECO branch", all_branches, default=all_branches, key="f_branch")
    st.multiselect("Status", STATUS_ORDER, default=STATUS_ORDER, key="f_status")
    st.text_input("Search recloser", key="f_search", placeholder="e.g. AR-042")
    if ML_ON:
        st.radio("Column height", ["Risk (100 - HI)", "ML failure risk"], key="f_height")
    filtered = _filtered(snap)
    st.markdown("### 🎯 SELECTED RECLOSER")
    options = list(_sorted(filtered)["asset_id"])
    info = snap.set_index("asset_id")
    asset = st.selectbox("Recloser", options, key="asset",
                         format_func=lambda a: f"{a} · {info.loc[a, 'status']} · HI {_fmt(info.loc[a, 'health_index'])}") \
        if options else None
    st.caption("Refreshes every 60 s. Data: telemetry, V1-V42, Health Index" + (" and ML risk" if ML_ON else "")
               + " every 15 min.")

TAB_NAMES = ["Command Centre", "Fleet", "Recloser", *(["ML vs HI", "Deep Learning"] if ML_ON else []), "Maintenance"]
TABS = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))
tab_cc, tab_fleet, tab_ar, tab_maint = (TABS[n] for n in ("Command Centre", "Fleet", "Recloser", "Maintenance"))
NO_MATCH = "No reclosers match the filters - change them in the sidebar."

with tab_cc:
    @st.fragment(run_every=60)
    def command_centre():
        ts_now, snap_now = _snapshot(URL)
        view = _filtered(snap_now)
        if view.empty:
            st.info(NO_MATCH)
            return
        cols = st.columns(len(STATUS_ORDER))
        for col, s in zip(cols, STATUS_ORDER):
            n = int((view["status"] == s).sum())
            col.markdown(kpi_tile(s, n, STATUS_COLORS_HEX[s], KPI_HINT[s],
                                  pulse=s in ("FAULTY", "CRITICAL") and n > 0), unsafe_allow_html=True)
        st.write("")
        kpis = st.columns(4 if ML_ON else 3)
        k1, k3, k4 = kpis[0], kpis[-2], kpis[-1]
        k1.markdown(kpi_tile("FLEET AVERAGE HI", _fmt(view["health_index"].mean()), CYAN, f"{len(view)} reclosers"),
                    unsafe_allow_html=True)
        if ML_ON:
            ml_high = int(view["ml_class"].eq("HIGH").sum()) if view["ml_p"].notna().any() else None
            kpis[1].markdown(kpi_tile("ML HIGH RISK", "-" if ml_high is None else ml_high, "#ff2e93",
                                      "failure likely within 30 days" if ml_high is not None else
                                      ("model cannot be used - see ML vs HI" if model.get("error") else "no ML model yet"),
                                      pulse=bool(ml_high)), unsafe_allow_html=True)
        k3.markdown(kpi_tile("OPEN FAILURES", int((view["open_failures"] != "").sum()), STATUS_COLORS_HEX["FAULTY"],
                             "waiting for repair"), unsafe_allow_html=True)
        k4.markdown(kpi_tile(f"UNHEALTHY (HI < {HEALTHY_MIN:g})", int((view["health_index"] < HEALTHY_MIN).sum()),
                             STATUS_COLORS_HEX["POOR"],
                             "need maintenance planning"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([3.3, 1.35])
        mode = "ml" if ML_ON and st.session_state.get("f_height", "").startswith("ML") else "risk"
        with left:
            br = branches(CFG)
            br = br[br["name"].isin(view["area_name"].unique())]
            st.pydeck_chart(build_deck(view, br, mode, show_ml=ML_ON), height=650)
            st.markdown(legend_html(), unsafe_allow_html=True)
        with right:
            feed = _feed(URL)
            feed = feed[feed["asset_id"].isin(view["asset_id"])]
            items = "".join(feed_item(f"{r.ts_local:%d %b %H:%M}", r.kind, r.asset_id, r.detail,
                                      FEED_COLORS.get(r.kind, CYAN)) for r in feed.itertuples()) \
                or '<div class="ar-feed-item" style="--c:#8aa0c2">No events in the last 48 h</div>'
            st.markdown(f'<div class="ar-panel"><h4>LIVE ALERT FEED · 48 H</h4><div class="ar-feed">{items}</div></div>',
                        unsafe_allow_html=True)
            bars = "".join(branch_bar(r.area_name, r.avg_hi, r.reclosers, r.alerts)
                           for r in branch_summary(view).itertuples())
            st.markdown(f'<div class="ar-panel"><h4>BRANCH HEALTH</h4>{bars}</div>', unsafe_allow_html=True)

    command_centre()

with tab_fleet:
    if filtered.empty:
        st.info(NO_MATCH)
    else:
        summary = branch_summary(filtered)
        cols = st.columns(min(len(summary), 7) or 1)
        for col, r in zip(cols, summary.itertuples()):
            color = STATUS_COLORS_HEX["GOOD"] if (r.avg_hi or 0) >= CLASSES["GOOD"] else (
                STATUS_COLORS_HEX["FAIR"] if (r.avg_hi or 0) >= CLASSES["FAIR"] else STATUS_COLORS_HEX["POOR"])
            col.markdown(kpi_tile(r.area_name.upper(), _fmt(r.avg_hi), color,
                                  f"{r.reclosers} AR · {int(r.alerts)} alert"
                                  + (f" · {int(r.ml_high)} ML high" if ML_ON else "")),
                         unsafe_allow_html=True)
        st.write("")
        view = _sorted(filtered)
        table = pd.DataFrame({
            "Recloser": view["asset_id"], "Branch": view["area_name"], "Status": view["status"],
            "HI": view["health_index"], "ML risk %": view["ml_p"] * 100, "Likely part": view["ml_component"].fillna("-"),
            "Worst variables": view["worst_variables"], "Open failure": view["open_failures"],
            "Battery V": view["battery_voltage"], "Comms": view["comms_connected"], "Switch": view["switch_state"],
            "Feeder": view["feeder_name"]})
        columns = {"HI": st.column_config.ProgressColumn("HI", min_value=0, max_value=100, format="%.1f"),
                   "Battery V": st.column_config.NumberColumn(format="%.2f"),
                   "Comms": st.column_config.CheckboxColumn("Comms up")}
        if ML_ON:
            columns["ML risk %"] = st.column_config.ProgressColumn("ML risk %", min_value=0, max_value=100,
                                                                   format="%.0f%%")
        else:
            table = table.drop(columns=["ML risk %", "Likely part"])
        st.dataframe(table, hide_index=True, width="stretch", height=560, column_config=columns)

with tab_ar:
    if asset is None:
        st.info(NO_MATCH)
    else:
        row = info.loc[asset]
        status_color = STATUS_COLORS_HEX.get(row["status"], CYAN)
        if ML_ON:
            g1, g2, g3, g4 = st.columns([1, 1, 1.25, 1.9])
            ml_val = None if pd.isna(row["ml_p"]) else float(row["ml_p"]) * 100
            g2.markdown(gauge_svg(ml_val, 100, "ML FAILURE RISK", "#ff2e93", unit="% in 30 days"), unsafe_allow_html=True)
        else:
            g1, g3, g4 = st.columns([1, 1.25, 1.9])
        g1.markdown(gauge_svg(row["health_index"], 100, "HEALTH INDEX", status_color, unit="/ 100", decimals=1),
                    unsafe_allow_html=True)
        g3.markdown(radar_svg({label: row[col] for col, label in FACTORS.items()}), unsafe_allow_html=True)
        details = [("Branch / feeder", f"{row['area_name']} · {row['feeder_name']}"),
                   ("Open failure", row["open_failures"] or "none"),
                   *([("Likely failing part (ML)", row["ml_component"] if isinstance(row["ml_component"], str) else "-")]
                     if ML_ON else []),
                   ("Worst variables", row["worst_variables"] or "-"),
                   ("Battery", _fmt(row["battery_voltage"], 2, " V")),
                   ("AC supply", "-" if pd.isna(row["ac_supply_on"]) else ("ON" if row["ac_supply_on"] else "OFF")),
                   ("Comms", "-" if pd.isna(row["comms_connected"]) else ("UP" if row["comms_connected"] else "DOWN")),
                   ("Switch", row["switch_state"] if isinstance(row["switch_state"], str) else "-"),
                   ("Last data", "-" if pd.isna(row["last_telemetry_ts"]) else f"{_local(row['last_telemetry_ts']):%d %b %H:%M}")]
        import html as _html
        rows_html = "".join(f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                            f'border-bottom:1px solid #13213a"><span style="color:{MUTED}">{_html.escape(k)}</span>'
                            f'<span>{_html.escape(str(v))}</span></div>' for k, v in details)
        g4.markdown(f'<div class="ar-panel"><h4>{_html.escape(asset)} &nbsp; {status_pill(row["status"])}</h4>'
                    f'{rows_html}</div>', unsafe_allow_html=True)
        rows, _ = _history(URL, asset)
        st.markdown("#### HEALTH INDEX & ML RISK · 30 DAYS" if ML_ON else "#### HEALTH INDEX · 30 DAYS")
        if not rows.empty:
            st.altair_chart(hi_chart(rows, _ml_history(URL, asset) if ML_ON else None, model.get("threshold")),
                            width="stretch")
        daily = _daily(URL)
        mine = daily[daily["asset_id"] == asset]
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown("#### DAILY AVERAGE HEALTH INDEX")
            if not mine.empty:
                st.altair_chart(daily_chart(mine), width="stretch")
        with c2:
            st.markdown("#### RECENT EVENTS · 30 DAYS")
            st.dataframe(_events(URL, asset), hide_index=True, width="stretch", height=240)
        with st.expander("Latest V1-V42 values (by factor)"):
            values = _variables(URL, asset)
            cols = st.columns(len(FACTOR_VARS))
            for col, (factor, keys) in zip(cols, FACTOR_VARS.items()):
                lines = []
                for k, v in values.items():
                    if k[:3] in keys:
                        shown = f"{v:.2f}" if isinstance(v, float) else str(v)
                        lines.append(f"**{k[:3].upper()}** {k[4:].replace('_', ' ')}: `{shown}`")
                col.markdown(f"**{factor}**\n\n" + "\n\n".join(lines))

def render_ml_tab():
    has_ml = filtered["ml_p"].notna().any() if not filtered.empty else False
    if not model.get("installed"):
        if model.get("error"):
            title = "### The installed ML model cannot be used"
            err = (f"\n\n⚠ **{model['error']}**\n\nThe Health Index goes on as normal; the ML values are hidden "
                   f"until a working model is in `{model['folder']}`.")
        else:
            title, err = "### No ML model installed yet", ""
        st.markdown(f"""
{title}
The machine-learning model is trained in **Google Colab**, not on this PC.{err}

1. Copy `output/ml_generated/ar_ml_dataset_hourly.csv.gz` to a Google Drive folder named **AR_ML**.
2. Open `notebooks/AR_ML_Colab.ipynb` in Colab and choose **Runtime → Run all**.
3. Unzip the downloaded **`ar_ml_model.zip`** into `{model['folder']}`.
4. Run `python main.py ml-score` once (it scores all stored data); after that the collector scores every 15 minutes.

The full step-by-step guide is in `docs/ML_COLAB_GUIDE.md`.
""")
    elif filtered.empty:
        st.info(NO_MATCH)
    else:
        thr = float(model.get("threshold") or 0.5)
        metrics = model.get("metrics") or {}
        chips = "".join(f'<span class="ar-chip">{html.escape(str(k).replace("_", " ").upper())} · '
                        f'{html.escape(str(v))}</span>' for k, v in list(metrics.items())[:6])
        st.markdown(f'<div class="ar-panel"><h4>MODEL {html.escape(str(model.get("version", "-")))} · ALARM LEVEL '
                    f'{thr:.0%}</h4><div class="ar-meta" style="justify-content:flex-start">{chips}</div></div>',
                    unsafe_allow_html=True)
        stale = int(filtered["ml_stale"].sum())
        if stale:
            st.warning(f"{stale} recloser(s) have ML values older than their Health Index, so they are hidden until "
                       "they are scored again. If `ml-score` is running, wait for it to finish and refresh; otherwise "
                       "the next 15-minute collector run catches them up (log: `logs/ar_gen.log`).")
        elif not has_ml:
            st.info("No ML predictions yet. Run `python main.py ml-score` once (it scores all stored data); "
                    "after that the collector scores every 15 minutes.")
        else:
            pts = filtered.dropna(subset=["health_index", "ml_p"]).assign(ml_pct=lambda d: d["ml_p"] * 100)
            hi_ok, ml_ok = pts["health_index"] >= HEALTHY_MIN, pts["ml_p"] < thr
            q1, q2, q3, q4 = st.columns(4)
            q1.markdown(kpi_tile("BOTH HEALTHY", int((hi_ok & ml_ok).sum()), STATUS_COLORS_HEX["GOOD"]), unsafe_allow_html=True)
            q2.markdown(kpi_tile("ONLY ML WARNS", int((hi_ok & ~ml_ok).sum()), "#ff2e93", "early warning?"),
                        unsafe_allow_html=True)
            q3.markdown(kpi_tile("ONLY HI WARNS", int((~hi_ok & ml_ok).sum()), STATUS_COLORS_HEX["POOR"]), unsafe_allow_html=True)
            q4.markdown(kpi_tile("BOTH WARN", int((~hi_ok & ~ml_ok).sum()), STATUS_COLORS_HEX["CRITICAL"], "act now",
                                 pulse=bool((~hi_ok & ~ml_ok).sum())), unsafe_allow_html=True)
            scatter = alt.Chart(pts).mark_circle(size=120, opacity=0.9, stroke="#ffffff", strokeWidth=0.4).encode(
                x=alt.X("health_index:Q", title="Health Index", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("ml_pct:Q", title="ML failure risk % (30 days)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("status:N", scale=alt.Scale(domain=STATUS_ORDER, range=[STATUS_COLORS_HEX[s] for s in STATUS_ORDER]),
                                legend=alt.Legend(orient="right", title=None)),
                tooltip=["asset_id", "area_name", "status", alt.Tooltip("health_index:Q", format=".1f"),
                         alt.Tooltip("ml_pct:Q", format=".1f", title="ML risk %"), "ml_component"])
            rules = alt.Chart(pd.DataFrame({"x": [HEALTHY_MIN]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q") + \
                alt.Chart(pd.DataFrame({"y": [thr * 100]})).mark_rule(color="#ff2e93", strokeDash=[4, 4]).encode(y="y:Q")
            c1, c2 = st.columns([1.1, 1])
            with c1:
                st.markdown("#### HI VS ML RISK · NOW")
                st.altair_chart(altair_dark((scatter + rules).properties(height=420)), width="stretch")
            with c2:
                st.markdown("#### DAILY AGREEMENT · 25 HIGHEST-RISK RECLOSERS")
                daily = _daily(URL)
                daily = daily[daily["asset_id"].isin(filtered["asset_id"]) & daily["ml_healthy"].notna()]
                top = daily.groupby("asset_id")["max_p_fail"].max().sort_values(ascending=False).head(25).index
                daily = daily[daily["asset_id"].isin(top)]
                if daily.empty:
                    st.caption("No daily ML summary yet - it appears after the first ML-scored day.")
                else:
                    daily = daily.assign(day=pd.to_datetime(daily["day"]), verdict=[
                        "both healthy" if h and m else "both warn" if not h and not m else
                        "only ML warns" if h else "only HI warns" for h, m in zip(daily["hi_healthy"], daily["ml_healthy"])])
                    heat = alt.Chart(daily).mark_rect(cornerRadius=2, stroke="#050a14", strokeWidth=1.5).encode(
                        x=alt.X("day:T", title=None), y=alt.Y("asset_id:N", title=None, sort="ascending"),
                        color=alt.Color("verdict:N", scale=alt.Scale(
                            domain=["both healthy", "only ML warns", "only HI warns", "both warn"],
                            range=[STATUS_COLORS_HEX["GOOD"], "#ff2e93", STATUS_COLORS_HEX["POOR"], STATUS_COLORS_HEX["CRITICAL"]]),
                            legend=alt.Legend(orient="top", title=None)),
                        tooltip=["asset_id", alt.Tooltip("day:T"), "verdict", alt.Tooltip("avg_hi:Q", format=".1f"),
                                 alt.Tooltip("max_p_fail:Q", format=".0%", title="max ML risk")])
                    st.altair_chart(altair_dark(heat.properties(height=max(240, 16 * daily["asset_id"].nunique()))),
                                    width="stretch")
            disagree = pts[hi_ok != ml_ok]
            st.markdown("#### WHERE HI AND ML DISAGREE")
            st.dataframe(pd.DataFrame({"Recloser": disagree["asset_id"], "Branch": disagree["area_name"],
                                       "HI": disagree["health_index"], "ML risk %": disagree["ml_pct"],
                                       "Likely part": disagree["ml_component"], "Worst variables": disagree["worst_variables"]}),
                         hide_index=True, width="stretch")

def render_dl_tab():
    if not dl.get("installed"):
        if dl.get("error"):
            title = "### The installed deep-learning model cannot be used"
            err = (f"\n\n⚠ **{dl['error']}**\n\nThe Health Index and the LightGBM model go on as normal; the "
                   f"deep-learning values are hidden until a working model is in `{dl['folder']}`.")
        else:
            title, err = "### No deep-learning model installed yet", ""
        st.markdown(f"""
{title}
The deep-learning model reads the **last 14 days of each recloser as a sequence** and works next to the LightGBM
model. It is trained in **Google Colab** on a free GPU, not on this PC.{err}

1. Keep `ar_ml_dataset_hourly.csv.gz` in the Google Drive folder **AR_ML** (the same file as for LightGBM). For a
   side-by-side test, also put `ar_ml_model.zip` there.
2. Open `notebooks/AR_DL_Colab.ipynb` in Colab, choose **Runtime → Change runtime type → T4 GPU**, then
   **Runtime → Run all**.
3. Unzip the downloaded **`ar_dl_model.zip`** into `{dl['folder']}` (not the LightGBM folder).
4. Run `python main.py dl-score` once. After that, the collector scores every 15 minutes.

PyTorch must be installed on this PC: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
The full guide is in `docs/ML_COLAB_GUIDE.md`.
""")
    elif filtered.empty:
        st.info(NO_MATCH)
    else:
        view = with_dl(filtered, _dl_snapshot(URL), int(CFG["interval_minutes"]))
        dthr = float(dl["threshold"])
        chips = "".join(f'<span class="ar-chip">{html.escape(str(k).replace("_", " ").upper())} · '
                        f'{html.escape(str(v))}</span>' for k, v in list(dl["metrics"].items())[:10])
        st.markdown(f'<div class="ar-panel"><h4>DEEP LEARNING {html.escape(dl["version"])} · '
                    f'{html.escape(dl["arch"].upper())} · WINDOW {html.escape(dl["window"])} · ALARM LEVEL {dthr:.0%}'
                    f'</h4><div class="ar-meta" style="justify-content:flex-start">{chips}</div></div>',
                    unsafe_allow_html=True)
        stale = int(view["dl_stale"].sum())
        if stale:
            st.warning(f"{stale} recloser(s) have deep-learning values older than their Health Index, so they are "
                       "hidden until they are scored again. If `dl-score` is running, wait for it to finish and refresh; "
                       "otherwise the next 15-minute collector run catches them up (log: `logs/ar_gen.log`).")
        elif not view["dl_p"].notna().any():
            st.info("No deep-learning predictions yet. Run `python main.py dl-score` once (it scores all stored "
                    "data); after that the collector scores every 15 minutes.")
        else:
            has_lgbm = bool(model.get("installed")) and view["ml_p"].notna().any()
            ref = "LIGHTGBM" if has_lgbm else "HI"
            pts = view.dropna(subset=["dl_p", "ml_p" if has_lgbm else "health_index"]).assign(
                dl_pct=lambda d: d["dl_p"] * 100, ml_pct=lambda d: d["ml_p"] * 100)
            dl_warn = pts["dl_p"] >= dthr
            ref_warn = pts["ml_p"] >= float(model["threshold"]) if has_lgbm else pts["health_index"] < HEALTHY_MIN
            q1, q2, q3, q4 = st.columns(4)
            n_high = int(pts["dl_class"].eq("HIGH").sum())
            q1.markdown(kpi_tile("DL HIGH RISK", n_high, VIOLET, "failure likely within 30 days", pulse=bool(n_high)),
                        unsafe_allow_html=True)
            q2.markdown(kpi_tile(f"DL + {ref} WARN", int((dl_warn & ref_warn).sum()), STATUS_COLORS_HEX["CRITICAL"],
                                 "both agree: act now"), unsafe_allow_html=True)
            q3.markdown(kpi_tile("ONLY DL WARNS", int((dl_warn & ~ref_warn).sum()), VIOLET, "early warning?"),
                        unsafe_allow_html=True)
            q4.markdown(kpi_tile(f"ONLY {ref} WARNS", int((~dl_warn & ref_warn).sum()),
                                 ML_PINK if has_lgbm else STATUS_COLORS_HEX["POOR"]), unsafe_allow_html=True)
            if has_lgbm:
                x_field, x_title, x_rule = "ml_pct", "LightGBM failure risk % (30 days)", float(model["threshold"]) * 100
            else:
                x_field, x_title, x_rule = "health_index", "Health Index", HEALTHY_MIN
            scatter = alt.Chart(pts).mark_circle(size=120, opacity=0.9, stroke="#ffffff", strokeWidth=0.4).encode(
                x=alt.X(f"{x_field}:Q", title=x_title, scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("dl_pct:Q", title="Deep-learning failure risk % (30 days)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("status:N", scale=alt.Scale(domain=STATUS_ORDER,
                                                            range=[STATUS_COLORS_HEX[s] for s in STATUS_ORDER]),
                                legend=alt.Legend(orient="right", title=None)),
                tooltip=["asset_id", "area_name", "status", alt.Tooltip("health_index:Q", format=".1f", title="HI"),
                         alt.Tooltip("ml_pct:Q", format=".1f", title="LightGBM risk %"),
                         alt.Tooltip("dl_pct:Q", format=".1f", title="DL risk %"), "dl_component"])
            rules = alt.Chart(pd.DataFrame({"x": [x_rule]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q") + \
                alt.Chart(pd.DataFrame({"y": [dthr * 100]})).mark_rule(color=VIOLET, strokeDash=[4, 4]).encode(y="y:Q")
            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.markdown(f"#### DEEP LEARNING VS {ref} · NOW")
                st.altair_chart(altair_dark((scatter + rules).properties(height=380)), width="stretch")
            with c2:
                if asset is not None:
                    st.markdown(f"#### {html.escape(asset)} · HI & FAILURE RISK · 30 DAYS")
                    rows, _ = _history(URL, asset)
                    if not rows.empty:
                        risks = {"LightGBM": _ml_history(URL, asset) if model.get("installed") else None,
                                 "Deep learning": _dl_history(URL, asset)}
                        st.altair_chart(models_chart(rows, risks), width="stretch")
            ranked = view.sort_values("dl_p", ascending=False, na_position="last")
            st.markdown("#### RISK RANKING · DEEP LEARNING")
            st.dataframe(pd.DataFrame({
                "Recloser": ranked["asset_id"], "Branch": ranked["area_name"], "Status": ranked["status"],
                "HI": ranked["health_index"], "LightGBM risk %": ranked["ml_p"] * 100, "DL risk %": ranked["dl_p"] * 100,
                "DL class": ranked["dl_class"], "DL likely part": ranked["dl_component"],
                "LightGBM likely part": ranked["ml_component"]}), hide_index=True, width="stretch", height=420,
                column_config={
                    "HI": st.column_config.ProgressColumn("HI", min_value=0, max_value=100, format="%.1f"),
                    "LightGBM risk %": st.column_config.ProgressColumn("LightGBM risk %", min_value=0, max_value=100,
                                                                       format="%.0f%%"),
                    "DL risk %": st.column_config.ProgressColumn("DL risk %", min_value=0, max_value=100,
                                                                 format="%.0f%%")})

if ML_ON:
    with TABS["ML vs HI"]:
        render_ml_tab()
    with TABS["Deep Learning"]:
        render_dl_tab()

with tab_maint:
    if asset is None:
        st.info(NO_MATCH)
    else:
        latest_local = _local(ts)
        st.markdown(f"#### MAINTENANCE DONE · {asset}")
        with st.form("maintenance"):
            st.markdown("Resets the Health Index of the repaired part(s) from the maintenance time; the normal "
                        "15-minute calculation continues afterwards. Choose the recloser in the sidebar.")
            parts = st.multiselect("Repaired part(s)", ["ALL", *RESET_GROUPS], key="parts",
                                   format_func=lambda p: PART_LABELS[p])
            c1, c2 = st.columns(2)
            day = c1.date_input("Date (local)", value=latest_local.date(), key="mdate")
            tim = c2.time_input("Time (local)", value=latest_local.time().replace(second=0, microsecond=0), key="mtime")
            note = st.text_input("Note (technician / work done)", key="mnote")
            submitted = st.form_submit_button("Save & recalculate")
        if submitted:
            if not parts:
                st.error("Choose at least one repaired part.")
            else:
                at_ms = dt_to_ms(datetime.combine(day, tim).replace(tzinfo=zone(TZ)))
                try:
                    with st.spinner("Saving and recalculating the Health Index ..."):
                        res = pipeline.maintenance_done(CFG, asset, parts, at_ms, note)
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
                else:
                    st.cache_data.clear()
                    st.success(f"Saved for {asset}: HI at the maintenance time {_fmt(res['hi_before'])} → "
                               f"{_fmt(res['hi_after'])} ({res['recomputed_rows']} rows recalculated). "
                               "The normal 15-minute calculation continues from here.")
        st.markdown("#### RECENT HEALTH-INDEX RESETS")
        st.dataframe(_resets(URL), hide_index=True, width="stretch")
