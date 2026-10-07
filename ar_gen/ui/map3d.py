"""Dark 3D map of the LECO fleet (pydeck): glowing risk columns, problem halos, feeders, branch labels."""
from __future__ import annotations

import html

import pandas as pd
import pydeck as pdk

from ..dashboard_data import feeder_paths

PROBLEM = {"FAULTY", "CRITICAL", "LOCKOUT", "POOR", "OFFLINE"}


def _txt(v, fmt="{:.1f}", none="-"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return none
    try:
        return html.escape(fmt.format(v))
    except (TypeError, ValueError):
        return html.escape(str(v))


def view_state(snapshot: pd.DataFrame) -> pdk.ViewState:
    """Fit the whole (filtered) fleet; pitched 3D view."""
    placed = snapshot.dropna(subset=["latitude", "longitude"])
    if placed.empty:
        return pdk.ViewState(latitude=6.65, longitude=80.0, zoom=8.5, pitch=50, bearing=-18)
    lat_lo, lat_hi = placed["latitude"].min(), placed["latitude"].max()
    lon_lo, lon_hi = placed["longitude"].min(), placed["longitude"].max()
    span = max(lat_hi - lat_lo, lon_hi - lon_lo)
    zoom = 10.6 if span < 0.25 else (9.8 if span < 0.6 else (9.0 if span < 1.0 else 8.45))
    return pdk.ViewState(latitude=float((lat_lo + lat_hi) / 2 - 0.06 * span),
                         longitude=float((lon_lo + lon_hi) / 2 + 0.12), zoom=zoom, pitch=50, bearing=-20)


def build_deck(snapshot: pd.DataFrame, branches: pd.DataFrame, height_mode: str = "risk") -> pdk.Deck:
    """height_mode 'risk' = 100 - HI, 'ml' = ML failure probability x 100."""
    data = snapshot.dropna(subset=["latitude", "longitude"]).copy()
    if height_mode == "ml" and "ml_p" in data:
        data["elev"] = (pd.to_numeric(data["ml_p"], errors="coerce").fillna(0.0) * 100.0).clip(lower=4.0)
    else:
        data["elev"] = data["risk"].clip(lower=4.0)
    data["fill"] = [list(c) + [235] for c in data["color"]]
    data["halo"] = [list(c) + [55] for c in data["color"]]
    data["line"] = [list(c) + [220] for c in data["color"]]
    data["hi_txt"] = data["health_index"].map(_txt)
    data["ml_txt"] = data.get("ml_p", pd.Series(index=data.index, dtype=float)).map(
        lambda v: _txt(v, "{:.0%}"))
    data["part_txt"] = data.get("ml_component", pd.Series(index=data.index, dtype=object)).map(
        lambda v: _txt(v, "{}"))
    data["batt_txt"] = data["battery_voltage"].map(lambda v: _txt(v, "{:.2f} V"))
    data["comms_txt"] = data["comms_connected"].map(lambda v: "-" if pd.isna(v) else ("UP" if v else "DOWN"))
    data["worst_txt"] = data["worst_variables"].fillna("-").map(lambda v: html.escape(str(v)))
    data["fail_txt"] = data["open_failures"].replace("", "none").map(lambda v: html.escape(str(v)))
    for c in ("asset_id", "area_name", "feeder_name", "status", "health_class", "switch_state"):
        data[c + "_e"] = data[c].fillna("-").map(lambda v: html.escape(str(v)))

    layers = [
        pdk.Layer("PathLayer", data=feeder_paths(snapshot).assign(c=[[0, 229, 255, 110]] * len(feeder_paths(snapshot))),
                  get_path="path", get_color="c", width_min_pixels=1.6, pickable=False),
        pdk.Layer("ScatterplotLayer", data=data[data["status"].isin(PROBLEM)], get_position=["longitude", "latitude"],
                  get_radius=1700, get_fill_color="halo", get_line_color="line", stroked=True, filled=True,
                  line_width_min_pixels=1.5, pickable=False),
        pdk.Layer("ColumnLayer", data=data, get_position=["longitude", "latitude"], get_elevation="elev",
                  elevation_scale=230, radius=650, get_fill_color="fill", pickable=True, extruded=True,
                  auto_highlight=True, highlight_color=[255, 255, 255, 120]),
    ]
    if branches is not None and not branches.empty:
        b = branches.assign(name_e=branches["name"].map(lambda v: html.escape(str(v)).upper()))
        layers += [
            pdk.Layer("ScatterplotLayer", data=b, get_position=["longitude", "latitude"], get_radius=900,
                      radius_min_pixels=7, get_fill_color=[0, 229, 255, 70], get_line_color=[0, 229, 255, 255],
                      stroked=True, line_width_min_pixels=2, pickable=False),
            pdk.Layer("TextLayer", data=b, get_position=["longitude", "latitude"], get_text="name_e", get_size=16,
                      size_units="pixels", get_color=[210, 250, 255, 255], get_pixel_offset=[0, -28],
                      get_text_anchor="'middle'", get_alignment_baseline="'bottom'", billboard=True, pickable=False),
        ]
    tooltip = {"html": "<b style='font-family:Orbitron'>{asset_id_e}</b> &nbsp; {status_e}<br/>"
                       "Health Index <b>{hi_txt}</b> ({health_class_e}) &nbsp; ML risk <b>{ml_txt}</b><br/>"
                       "Likely part: {part_txt} &nbsp; Worst: {worst_txt}<br/>Open failure: {fail_txt}<br/>"
                       "Battery {batt_txt} &nbsp; Comms {comms_txt} &nbsp; Switch {switch_state_e}<br/>"
                       "<span style='color:#8aa0c2'>{area_name_e} · {feeder_name_e}</span>",
               "style": {"backgroundColor": "rgba(5,10,20,0.92)", "color": "#e6f1ff", "fontSize": "12px",
                         "border": "1px solid rgba(0,229,255,0.4)", "borderRadius": "10px"}}
    return pdk.Deck(layers=layers, map_style="dark", initial_view_state=view_state(snapshot), tooltip=tooltip)
