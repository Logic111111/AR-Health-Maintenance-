"""Collects the rows one simulated chunk produces, allocates ids and tracks alarm state."""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field

from .state import AssetState, SimState
from .timeutil import ms_to_dt


@dataclass
class Chunk:
    rows: dict
    asset_updates: list
    ground_truth: list
    last_ms: int
    next_ids: dict = field(default_factory=dict)


def _r(x, nd):
    return None if x is None else round(float(x), nd)


class Out:
    def __init__(self, state: SimState):
        self.state = state
        self.rows: dict[str, list] = defaultdict(list)
        self.ground_truth: list = []
        self._dirty_assets: dict[str, AssetState] = {}

    # ------------------------------------------------------------ ids / queue
    def next_id(self, key: str) -> int:
        v = self.state.next_ids[key]
        self.state.next_ids[key] = v + 1
        return v

    def schedule(self, a: AssetState, t: int, kind: str, payload: dict | None = None) -> None:
        self.state.seq += 1
        heapq.heappush(a.queue, [int(t), self.state.seq, kind, payload or {}])

    def touch_asset(self, a: AssetState) -> None:
        self._dirty_assets[a.asset_id] = a

    # ------------------------------------------------------------ alarms
    def _alarm_row(self, a, t, category, signal, state, detail):
        self.rows["alarm_events"].append(dict(
            event_id=self.next_id("alarm"), ts=ms_to_dt(t), asset_id=a.asset_id,
            category=category, signal_name=signal, state=state,
            detail=None if detail is None else str(detail)[:128]))

    def raise_alarm(self, a: AssetState, t: int, signal: str, category: str = "WARNING",
                    detail=None, clear_at: int | None = None) -> bool:
        if signal in a.alarms:
            return False
        a.alarms[signal] = {"cat": category, "since": int(t),
                            "clear_at": None if clear_at is None else int(clear_at)}
        self._alarm_row(a, t, category, signal, "RAISED", detail)
        return True

    def clear_alarm(self, a: AssetState, t: int, signal: str, detail=None) -> bool:
        info = a.alarms.pop(signal, None)
        if info is None:
            return False
        self._alarm_row(a, max(int(t), info["since"] + 1), info["cat"], signal, "CLEARED", detail)
        return True

    def event(self, a: AssetState, t: int, signal: str, category: str = "WARNING", detail=None) -> None:
        self._alarm_row(a, t, category, signal, "EVENT", detail)

    def flush_alarm_clears(self, a: AssetState, t1: int) -> None:
        due = sorted((info["clear_at"], sig) for sig, info in a.alarms.items()
                     if info.get("clear_at") is not None and info["clear_at"] <= t1)
        for t, sig in due:
            self.clear_alarm(a, t, sig)

    # ------------------------------------------------------------ other events
    def comms_event(self, a: AssetState, t: int, event: str, channel: str | None = None) -> None:
        self.rows["comms_events"].append(dict(
            event_id=self.next_id("comms"), ts=ms_to_dt(t), asset_id=a.asset_id,
            channel=channel or a.channel, event=event))

    def battery_test(self, a, t, trigger, result, reason=None, v_before=None, v_min=None, load_a=None):
        self.rows["battery_tests"].append(dict(
            test_id=self.next_id("test"), ts=ms_to_dt(t), asset_id=a.asset_id, trigger=trigger,
            result=result, not_performed_reason=reason, voltage_before=_r(v_before, 2),
            voltage_min_during_test=_r(v_min, 2), test_load_current_a=_r(load_a, 1)))

    def failure(self, a, t, component, detected_by, description) -> int:
        fid = self.next_id("failure")
        self.rows["failure_log"].append(dict(
            failure_id=fid, ts=ms_to_dt(t), asset_id=a.asset_id, component=component,
            detected_by=detected_by, description=description[:255]))
        return fid

    def maintenance(self, a, t, component, maint_type, description, failure_id=None,
                    changes: dict | None = None, extra: dict | None = None) -> int:
        mid = self.next_id("maint")
        details = dict(extra or {})
        if changes:
            details["changes"] = changes
        self.rows["maintenance_log"].append(dict(
            maint_id=mid, ts=ms_to_dt(t), asset_id=a.asset_id, component=component,
            maint_type=maint_type, failure_id=failure_id, description=description[:255],
            details=details or None))
        self.touch_asset(a)
        return mid

    def op(self, a, t, op_type, source, ar_state, currents, op_time_ms, result, fault_id=None) -> dict:
        """Build a Close/Open log row (id assigned when emitted)."""
        ia, ib, ic, i_n = currents if currents is not None else (None, None, None, None)
        return dict(op_id=None, ts_ms=int(t), asset_id=a.asset_id, op_type=op_type, source=source,
                    ar_state=ar_state, trip_ia=_r(ia, 1), trip_ib=_r(ib, 1), trip_ic=_r(ic, 1),
                    trip_in=_r(i_n, 1), operating_time_ms=_r(op_time_ms, 1), result=result,
                    fault_id=fault_id)

    def emit_op(self, row: dict) -> None:
        row = dict(row)
        row["op_id"] = self.next_id("op")
        row["ts"] = ms_to_dt(row.pop("ts_ms"))
        self.rows["co_operations"].append(row)

    def fault(self, row: dict) -> None:
        self.rows["fault_events"].append(row)

    def telemetry(self, row: dict) -> None:
        self.rows["telemetry_15min"].append(row)

    # ------------------------------------------------------------ finish
    def finish(self, asset_update_fn) -> Chunk:
        updates = [asset_update_fn(a) for a in self._dirty_assets.values()]
        return Chunk(rows=dict(self.rows), asset_updates=updates, ground_truth=self.ground_truth,
                     last_ms=self.state.last_ms, next_ids=dict(self.state.next_ids))
