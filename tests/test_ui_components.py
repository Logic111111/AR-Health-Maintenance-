"""SVG/HTML building blocks of the futuristic dashboard: well-formed, clamped, escaped."""
import xml.etree.ElementTree as ET

from ar_gen.ui.components import gauge_svg, kpi_tile, radar_svg, status_pill
from ar_gen.ui.theme import STATUS_COLORS_HEX


def _parse(svg):
    return ET.fromstring(svg)


def test_gauge_is_valid_svg_and_clamps_values():
    for value in (-20, 0, 37.5, 100, 250, None):
        root = _parse(gauge_svg(value, 100, "Health <Index>", "#00e5ff"))
        assert root.tag.endswith("svg")
    text = gauge_svg(250, 100, "HI", "#00e5ff")
    assert ">100<" in text                                       # shown value clamped to the maximum
    assert "&lt;Index&gt;" in gauge_svg(50, 100, "Health <Index>", "#00e5ff")
    assert ">-<" in gauge_svg(None, 100, "HI", "#00e5ff")         # missing value


def test_radar_has_one_axis_per_factor_and_escapes_labels():
    scores = {"Battery": 80, "Faults": 55, "Switching": 100, "Comms": 30, "Controlling": None, "Aging": 70, "Env<": 120}
    svg = radar_svg(scores)
    root = _parse(svg)
    texts = [t.text for t in root.iter() if t.tag.endswith("text")]
    assert "Env&lt;" in svg and len([t for t in texts if t]) >= 7


def test_kpi_tile_and_status_pill_escape_text():
    tile = kpi_tile("FAULTY <b>", 3, STATUS_COLORS_HEX["FAULTY"], pulse=True)
    assert "&lt;b&gt;" in tile and "ar-pulse" in tile
    assert "ar-pulse" not in kpi_tile("GOOD", 90, STATUS_COLORS_HEX["GOOD"])
    pill = status_pill("CRITICAL")
    assert "CRITICAL" in pill and STATUS_COLORS_HEX["CRITICAL"] in pill
    assert "&lt;x&gt;" in status_pill("<x>")


def test_map_tooltip_text_is_always_escaped():
    from ar_gen.ui.map3d import _txt
    assert _txt("<img src=x onerror=alert(1)>", "{}") == "&lt;img src=x onerror=alert(1)&gt;"
    assert _txt(0.25, "{:.0%}") == "25%"
