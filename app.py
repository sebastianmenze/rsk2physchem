"""
CTD Profile Browser - Dash Web Application
Ports the PyQt5 CTD Profile Browser to a Dash web app with Leaflet maps.
"""

import os
import re
from dotenv import load_dotenv
load_dotenv()
import uuid
import json
import base64
import tempfile
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, ctx, ALL
import dash_leaflet as dl
import dash_bootstrap_components as dbc

try:
    import pyrsktools
    PYRSK_AVAILABLE = True
except ImportError:
    PYRSK_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# ── Configuration from environment (see .env)
TOKTLOGGER_CRUISES_URL    = os.getenv("TOKTLOGGER_CRUISES_URL")
TOKTLOGGER_ACTIVITIES_URL = os.getenv("TOKTLOGGER_ACTIVITIES_URL")
PHYSCHEM_API_URL          = os.getenv("PHYSCHEM_API_URL", "https://physchem-api.hi.no")
S3_ENDPOINT_URL           = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID          = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY      = os.getenv("S3_SECRET_ACCESS_KEY")
S3_BUCKET                 = os.getenv("S3_BUCKET")
S3_DEST_PREFIX            = os.getenv("S3_DEST_PREFIX")
PASSWORD                  = os.getenv("PASSWORD")


# ─────────────────────────────────────────────
# Data helpers  (ported from original script)
# ─────────────────────────────────────────────

def get_cruises_from_api(base_url=None):
    try:
        resp = requests.get(base_url or TOKTLOGGER_CRUISES_URL, params={"format": "json"}, timeout=10)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        if len(df) and "startTime" in df.columns:
            df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
            df["endTime"]   = pd.to_datetime(df["endTime"],   utc=True)
        return df
    except Exception:
        return pd.DataFrame()


def match_cruise_by_dates(start_date, end_date, df_cruises):
    if len(df_cruises) == 0:
        return None
    start_date = pd.Timestamp(start_date)
    end_date   = pd.Timestamp(end_date)
    if start_date.tz is None:
        start_date = start_date.tz_localize("UTC")
    else:
        start_date = start_date.tz_convert("UTC")
    if end_date.tz is None:
        end_date = end_date.tz_localize("UTC")
    else:
        end_date = end_date.tz_convert("UTC")

    matches = []
    for _, cruise in df_cruises.iterrows():
        cs = pd.Timestamp(cruise["startTime"])
        ce = pd.Timestamp(cruise["endTime"])
        cs = cs.tz_localize("UTC") if cs.tz is None else cs.tz_convert("UTC")
        ce = ce.tz_localize("UTC") if ce.tz is None else ce.tz_convert("UTC")
        if start_date <= ce and end_date >= cs:
            overlap = (min(end_date, ce) - max(start_date, cs)).total_seconds()
            matches.append({"cruise": cruise, "overlap": overlap})
    if not matches:
        return None
    return max(matches, key=lambda x: x["overlap"])["cruise"].to_dict()


def check_if_operation_in_physchem(meta):
    """Return True if this operation already exists in the PhysChem database."""
    try:
        platform       = meta.get("mission.platform", "")
        mission_number = meta.get("mission.missionNumber", "")
        time_start     = meta.get("operation.timeStart", "")
        if not platform or not mission_number or not time_start:
            return False

        resp = requests.get(
            f"{PHYSCHEM_API_URL}/mission/list",
            params={"platform": platform},
            timeout=10,
        )
        resp.raise_for_status()
        df_missions = pd.DataFrame(resp.json())
        if df_missions.empty:
            return False

        match = df_missions["missionNumber"] == int(mission_number)
        if match.sum() == 0:
            return False

        mission_id = df_missions.loc[match, "id"].values[0]

        resp2 = requests.get(
            f"{PHYSCHEM_API_URL}/mission/{mission_id}/operation/list",
            params={"extend": "false", "instrumentTypeList": "false"},
            timeout=10,
        )
        resp2.raise_for_status()
        df_ops = pd.DataFrame(resp2.json())
        if df_ops.empty or "timeStart" not in df_ops.columns:
            return False

        return bool(np.isin(df_ops["timeStart"], time_start).sum() > 0)
    except Exception:
        return False


def get_mission_number_from_physchem(cruise_number, platform, year):
    """Look up missionNumber from the physchem API by matching cruise number."""
    try:
        url  = f"{PHYSCHEM_API_URL}/mission/list"
        resp = requests.get(url, params={"startYear": year, "platform": platform}, timeout=10)
        resp.raise_for_status()
        for mission in resp.json():
            if str(mission.get("cruise", "")) == str(cruise_number):
                return str(mission["missionNumber"])
    except Exception:
        pass
    return ""


def get_activities_from_api(after, before, base_url=None):
    resp = requests.get(base_url or TOKTLOGGER_ACTIVITIES_URL,
                        params={"after": after, "before": before, "format": "json"},
                        timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if len(df) == 0:
        return df
    if "startPosition" in df.columns:
        df["startLon"] = df["startPosition"].apply(
            lambda x: x["coordinates"][0] if x and "coordinates" in x else None)
        df["startLat"] = df["startPosition"].apply(
            lambda x: x["coordinates"][1] if x and "coordinates" in x else None)
        df["endLon"] = df["endPosition"].apply(
            lambda x: x["coordinates"][0] if x and "coordinates" in x else None)
        df["endLat"] = df["endPosition"].apply(
            lambda x: x["coordinates"][1] if x and "coordinates" in x else None)
    if "startTime" in df.columns:
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    if "endTime" in df.columns:
        df["endTime"] = pd.to_datetime(df["endTime"], utc=True)
    return df


def process_rsk_file(filepath):
    """Read a single RSK file and return (df, meta)."""
    rsk = pyrsktools.RSK(filepath)
    rsk.open()
    rsk.readdata()
    rsk.deriveseapressure()
    rsk.derivesalinity()
    rsk.derivedepth()
    rsk.derivevelocity()
    df = pd.DataFrame(rsk.data)
    meta = {
        "instrument.instrumentSerialNumber": rsk.instrument.serialID,
        "instrument.instrumentModel":        rsk.instrument.model,
    }
    rsk.close()
    if "chlorophyll_a" in df.columns:
        df = df.rename(columns={"chlorophyll_a": "chlorophyll"})
    return df, meta


def ensure_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def get_station_indices_for_ctd(df_rsk, df_tk,
                                  time_tolerance_minutes=1,
                                  max_time_correction_minutes=15,
                                  surface_depth_threshold=2.0):
    station_indices = {}
    time_buffer     = pd.Timedelta(minutes=time_tolerance_minutes)
    correction_window = pd.Timedelta(minutes=max_time_correction_minutes)

    ts = pd.to_datetime(df_rsk["timestamp"])
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")

    ctd_stations = df_tk[df_tk["activityMainGroupName"] == "CTD"].copy()

    for _, station in ctd_stations.iterrows():
        t_start = ensure_utc(station["startTime"])
        t_end   = ensure_utc(station["endTime"])

        mask = (ts >= t_start - time_buffer) & (ts <= t_end + time_buffer)
        matching = df_rsk.index[mask].tolist()

        corrected_start = t_start
        time_corrected  = False
        correction_info = ""

        if matching and "depth" in df_rsk.columns:
            first_depth = df_rsk.loc[matching[0], "depth"]
            if first_depth > surface_depth_threshold:
                lb_mask = (ts >= t_start - correction_window) & (ts < t_start - time_buffer)
                lb_idx  = df_rsk.index[lb_mask].tolist()
                if lb_idx:
                    lb_data  = df_rsk.loc[lb_idx]
                    near_surf = lb_data[lb_data["depth"] <= surface_depth_threshold]
                    if len(near_surf):
                        actual_idx   = near_surf.index[0]
                        actual_time  = ensure_utc(df_rsk.loc[actual_idx, "timestamp"])
                        diff_min = (t_start - actual_time).total_seconds() / 60
                        corrected_start = actual_time
                        time_corrected  = True
                        correction_info = (f"Time corrected by {diff_min:.1f} min "
                                           f"(depth: {first_depth:.1f}m → "
                                           f"{df_rsk.loc[actual_idx,'depth']:.1f}m)")
                        mask = (ts >= corrected_start - time_buffer) & (ts <= t_end + time_buffer)
                        matching = df_rsk.index[mask].tolist()

        key = f"{station['name']}_{station['activityNumber']}"

        def fmt(t): return t.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

        station_indices[key] = {
            "df_rsk_indices": matching,
            "n_datapoints":   len(matching),
            "station_info": {
                "name":             station["name"],
                "activityNumber":   station["activityNumber"],
                "startTime":        fmt(corrected_start),
                "endTime":          fmt(t_end),
                "startLat":         station["startLat"],
                "startLon":         station["startLon"],
                "comment":          station.get("comment", ""),
                "time_corrected":   time_corrected,
                "correction_info":  correction_info,
                "original_startTime": fmt(t_start) if time_corrected else None,
            },
        }
    return station_indices


# ─────────────────────────────────────────────
# Downcast detection
# ─────────────────────────────────────────────

def detect_downcast(df_profile):
    """Return boolean array marking the downcast portion."""
    if len(df_profile) < 100:
        return np.zeros(len(df_profile), dtype=bool)

    peaks, p = find_peaks(df_profile["depth"],
                           height=df_profile["depth"].max() * 0.25,
                           width=500)
    if len(peaks) == 0:
        return np.zeros(len(df_profile), dtype=bool)

    lb = np.zeros(len(df_profile))
    k, pp_before = 1, 0
    for i, pp in enumerate(peaks):
        dmax = df_profile.loc[int(p["left_ips"][i]):int(p["right_ips"][i]), "depth"].max()
        ix = (df_profile["depth"] > 0.5) & (df_profile["depth"] < dmax) & \
             (df_profile["salinity"] > 0) & (df_profile.index < pp) & \
             (df_profile.index > pp_before) & \
             (df_profile["velocity"].rolling(100, center=True).mean() > 0.1)
        lb[ix] = k
        k += 1
        pp_before = pp

    lb_updated = np.zeros(len(lb))
    d_old, k = 0, 1
    for j in np.unique(lb)[1:]:
        ix1 = np.where(lb == j)[0][0]
        ix2 = np.where(lb == j)[0][-1]
        d_new = df_profile.loc[ix1, "depth"]
        if (d_old - d_new) < df_profile["depth"].max() * 0.5:
            lb_updated[lb == j] = k
        else:
            k += 1
            lb_updated[lb == j] = k
        d_old = df_profile.loc[ix2, "depth"]

    for j in np.unique(lb_updated)[1:]:
        span = np.where(lb_updated == j)[0]
        if span[-1] - span[0] < 500:
            lb_updated[lb_updated == j] = 0

    return lb_updated == 1


# ─────────────────────────────────────────────
# NPC calculation
# ─────────────────────────────────────────────

def calculate_df_npc(df_profile, selected_indices, excluded_indices,
                      include_o2, include_chl,
                      cruise_start_time, cruise_end_time,
                      cruise_number, vessel_name, mission_number,
                      platform_number, current_op_number, rsk_meta,
                      station_info):
    valid = [i for i in selected_indices if i not in excluded_indices]
    if not valid:
        return pd.DataFrame(), {}

    sel = df_profile.loc[valid]
    depth_min = sel["depth"].min()
    depth_max = sel["depth"].max()
    if depth_max <= depth_min:
        return pd.DataFrame(), {}

    interval = 1
    bin_min   = np.round(depth_min)
    bin_max   = np.round(depth_max)
    bins_calc = np.arange(bin_min - interval / 2, bin_max + interval, interval)
    if len(bins_calc) < 2:
        return pd.DataFrame(), {}
    bins = bins_calc[:-1] + interval / 2

    # Build channel lists
    channels      = ["timestamp", "conductivity", "temperature", "pressure", "salinity"]
    codes         = ["DATETIME",  "COND",         "TEMP",        "PRES",    "PSAL"]
    names         = ["Date and time",
                     "Electrical conductivity of water",
                     "Temperature of water",
                     "Sea Pressure",
                     "Practical salinity"]
    sup_names     = ["DateTime", "Conductivity", "Temperature", "Pressure", "Salinity"]
    units         = ["ISO8601 UTC", "mS/cm", "degC", "dbar", "PSU"]
    sup_units     = ["ISO8601 UTC", "mS/cm", "degC", "dbar", "PSU"]
    acq           = ["1019900"] * 5
    proc          = ["L0"] * 5

    if include_o2 and "dissolved_o2_concentration" in df_profile.columns:
        if df_profile["dissolved_o2_concentration"].dropna().any():
            channels.append("dissolved_o2_concentration")
            codes.append("DOX"); names.append("Dissolved oxygen from in-situ sensor")
            sup_names.append("Dissolved O2"); units.append("umol/l")
            sup_units.append("umol/l"); acq.append("1019900"); proc.append("L0")

    if include_chl and "chlorophyll" in df_profile.columns:
        if df_profile["chlorophyll"].dropna().any():
            channels.append("chlorophyll")
            codes.append("ChlA_SENS"); names.append("Chlorophyll-a fluorescence from in-situ sensor")
            sup_names.append("Chlorophyll"); units.append("ug/l")
            sup_units.append("ug/l"); acq.append("1019900"); proc.append("L0")

    # Build column list for df_npc
    cols = []
    for code in codes:
        if code == "DATETIME":
            cols += [code + ".value", code + ".sampleSize"]
        else:
            cols += [code + ".value", code + ".std", code + ".sampleSize"]
    cols.append("DEPTH.value")

    # Build metadata
    year = pd.Timestamp(cruise_start_time).year if cruise_start_time else datetime.utcnow().year
    meta = {
        "mission.missionNumber":       mission_number,
        "mission.missionStartDate":    pd.Timestamp(cruise_start_time).strftime("%Y-%m-%dT%H:%M:%SZ") if cruise_start_time else "",
        "mission.missionStopDate":     pd.Timestamp(cruise_end_time).strftime("%Y-%m-%dT%H:%M:%SZ") if cruise_end_time else "",
        "mission.missionType":         "14",
        "mission.platform":            platform_number,
        "mission.startYear":           year,
        "mission.platformName":        vessel_name,
        "mission.missionTypeName":     "Cruise",
        "mission.purpose":             "Cruise",
        "mission.missionName":         cruise_number,
        "mission.cruise":              cruise_number,
        "mission.responsibleLaboratory": 3,
        "operation.operationType":     "CTD",
        "operation.operationNumber":   current_op_number,
        "operation.timeStart":         pd.Timestamp(df_profile["timestamp"].min()).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operation.timeEnd":           pd.Timestamp(df_profile["timestamp"].max()).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operation.timeStartQuality":  0,
        "operation.timeEndQuality":    0,
        "operation.featureType":       4,
        "operation.latitudeStart":     station_info["startLat"],
        "operation.longitudeStart":    station_info["startLon"],
        "operation.positionStartQuality": 0,
        "operation.stationType":       1000,
        "operation.localCdiId":        str(uuid.uuid1()),
        "operation.operationComment":  station_info["comment"],
        "operation.operationPlatform": platform_number,
        "instrument.instrumentNumber": 1,
        "instrument.instrumentType":   "CTD",
        "instrument.instrumentSerialNumber": rsk_meta.get("instrument.instrumentSerialNumber", ""),
        "instrument.instrumentModel":        rsk_meta.get("instrument.instrumentModel", ""),
        "instrument.instrumentDataOwner":    3,
        "instrument.instrumentProperty.profileDirection": "D",
    }

    for i, (code, name, sp, u, su, a, pl) in enumerate(
            zip(codes, names, sup_names, units, sup_units, acq, proc), 1):
        meta[f"parameter{{{i}}}.parameterCode"]           = code
        meta[f"parameter{{{i}}}.units"]                   = u
        meta[f"parameter{{{i}}}.suppliedUnits"]           = su
        meta[f"parameter{{{i}}}.parameterName"]           = name
        meta[f"parameter{{{i}}}.suppliedParameterName"]   = sp
        meta[f"parameter{{{i}}}.acquirementMethod"]       = a
        meta[f"parameter{{{i}}}.processingLevel"]         = pl

    n = len(codes)
    meta[f"parameter{{{n+1}}}.parameterCode"]         = "DEPTH"
    meta[f"parameter{{{n+1}}}.units"]                 = "m"
    meta[f"parameter{{{n+1}}}.suppliedUnits"]         = "m"
    meta[f"parameter{{{n+1}}}.parameterName"]         = "Depth below sea level"
    meta[f"parameter{{{n+1}}}.suppliedParameterName"] = "Sea Pressure"
    meta[f"parameter{{{n+1}}}.acquirementMethod"]     = "1019900"
    meta[f"parameter{{{n+1}}}.processingLevel"]       = "L0"

    # Bin the data
    rows = []
    for i in range(len(bins)):
        mask = (sel["depth"] >= bins_calc[i]) & (sel["depth"] < bins_calc[i + 1])
        if mask.sum() == 0:
            continue
        row = []
        for c, code in zip(channels, codes):
            vals = sel.loc[mask, c].values
            if c == "timestamp":
                t = pd.to_datetime(vals).mean().strftime("%Y-%m-%dT%H:%M:%SZ")
                row += [t, len(vals)]
            else:
                row += [float(np.nanmean(vals)), float(np.nanstd(vals)), len(vals)]
        row.append(int(bins[i]))
        rows.append(row)

    df_npc = pd.DataFrame(rows, columns=cols)
    df_npc.insert(0, "sampleNumber", np.arange(1, len(df_npc) + 1))
    return df_npc, meta


def npc_write(meta, df, filename):
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Metadata:\n")
        for k, v in meta.items():
            f.write(f"{k}:\t{v}\n")
        f.write("% Readings:\n")
        f.write(df.to_csv(sep="\t", index=False, lineterminator="\n"))


def _npc_filename(cruise_number, start_time):
    """Return  cruisenumber_YYYYMMDD.npc  (fallback: unknown_unknown.npc)."""
    cn   = (cruise_number or "unknown").strip().replace(" ", "_")
    try:
        date = pd.Timestamp(start_time).strftime("%Y%m%d")
    except Exception:
        date = "unknown"
    return f"{cn}_{date}.npc"


def npc_to_string(meta, df):
    buf = StringIO()
    buf.write("# Metadata:\n")
    for k, v in meta.items():
        buf.write(f"{k}:\t{v}\n")
    buf.write("% Readings:\n")
    buf.write(df.to_csv(sep="\t", index=False, lineterminator="\n"))
    return buf.getvalue()


# ─────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────

def build_profile_figure(df_profile, span_start, span_end, df_npc, excluded_indices):
    """Build the 4-panel profile figure (temperature, salinity, O2, chlorophyll)."""
    has_o2  = "dissolved_o2_concentration" in df_profile.columns and \
              df_profile["dissolved_o2_concentration"].dropna().any()
    has_chl = "chlorophyll" in df_profile.columns and \
              df_profile["chlorophyll"].dropna().any()

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Temperature", "Salinity", "Dissolved O₂", "Chlorophyll"),
        shared_yaxes=True,
    )

    excl      = set(i for i in (excluded_indices or []) if i in df_profile.index)
    span_set  = set(range(int(span_start), int(span_end) + 1))

    # Span points: go into NPC averaging — must be selectable so lasso
    # exclusions are guaranteed to affect the bin calculation.
    keep_span    = [i for i in df_profile.index if i in span_set  and i not in excl]
    # Context points: outside the span — shown for reference, no customdata so
    # lasso selections on these rows are silently ignored by collect_exclusions.
    keep_context = [i for i in df_profile.index if i not in span_set and i not in excl]
    excl_list    = list(excl)

    vars_cfg = [
        ("temperature",                  "°C",      "TEMP.value",      "T=%{x:.3f}°C d=%{y:.1f}m"),
        ("salinity",                     "PSU",     "PSAL.value",      "S=%{x:.3f} PSU d=%{y:.1f}m"),
        ("dissolved_o2_concentration",   "µmol/l",  "DOX.value",       None),
        ("chlorophyll",                  "µg/l",    "ChlA_SENS.value", None),
    ]
    no_data_labels = {
        "dissolved_o2_concentration": "No O₂ data",
        "chlorophyll": "No Chl data",
    }

    for col_i, (col_name, xlabel, npc_col, htmpl) in enumerate(vars_cfg, start=1):
        has_col = col_name in df_profile.columns and df_profile[col_name].dropna().any()

        if not has_col:
            if col_name in no_data_labels:
                fig.add_annotation(
                    text=no_data_labels[col_name], x=0.5, y=0.5,
                    xref=f"x{col_i} domain", yref=f"y{col_i} domain",
                    showarrow=False,
                )
            continue

        # Context points (outside span): gray, not selectable
        if keep_context:
            fig.add_trace(go.Scatter(
                x=df_profile.loc[keep_context, col_name].tolist(),
                y=(-df_profile.loc[keep_context, "depth"]).tolist(),
                mode="markers",
                marker=dict(size=3, color="lightgray"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col_i)

        # Span points (used in NPC): Viridis, selectable via customdata
        if keep_span:
            kw = dict(hovertemplate=f"{htmpl}<extra></extra>") if htmpl else {}
            fig.add_trace(go.Scatter(
                x=df_profile.loc[keep_span, col_name].tolist(),
                y=(-df_profile.loc[keep_span, "depth"]).tolist(),
                mode="markers",
                marker=dict(size=4, color=keep_span, colorscale="Viridis"),
                showlegend=False, customdata=keep_span,
                **kw,
            ), row=1, col=col_i)

        # Excluded (blue X)
        if excl_list:
            fig.add_trace(go.Scatter(
                x=df_profile.loc[excl_list, col_name].tolist(),
                y=(-df_profile.loc[excl_list, "depth"]).tolist(),
                mode="markers", marker=dict(size=6, color="blue", symbol="x"),
                showlegend=False,
            ), row=1, col=col_i)

        # NPC binned line
        if npc_col and len(df_npc) and npc_col in df_npc.columns:
            fig.add_trace(go.Scatter(
                x=df_npc[npc_col].tolist(), y=(-df_npc["DEPTH.value"]).tolist(),
                mode="lines", line=dict(color="red", width=2), showlegend=False,
            ), row=1, col=col_i)

        # Fix axis ranges to span (Viridis) data so gray context dots don't
        # push the view out; a small 5 % margin keeps edge points visible.
        span_src = keep_span + [i for i in excl_list if i in span_set]
        if span_src and has_col:
            x_vals = df_profile.loc[span_src, col_name].dropna()
            y_vals = -df_profile.loc[span_src, "depth"]
            if len(x_vals):
                xpad = (x_vals.max() - x_vals.min()) * 0.05 or 0.5
                ypad = (y_vals.max() - y_vals.min()) * 0.05 or 0.5
                fig.update_xaxes(range=[x_vals.min() - xpad, x_vals.max() + xpad],
                                 title_text=xlabel, row=1, col=col_i)
                fig.update_yaxes(range=[y_vals.min() - ypad, y_vals.max() + ypad],
                                 row=1, col=col_i)
                continue

        fig.update_xaxes(title_text=xlabel, row=1, col=col_i)

    fig.update_yaxes(title_text="Depth (m)", row=1, col=1)
    fig.update_layout(
        margin=dict(l=40, r=20, t=50, b=40),
        dragmode="zoom",
        autosize=True,
    )
    return fig


def build_timeseries_figure(df_profile, span_start, span_end):
    """Build the depth-vs-time figure with highlighted selected span."""
    fig = go.Figure()

    ts_str = df_profile["timestamp"].astype(str).tolist()
    depths = (-df_profile["depth"]).tolist()

    # Full cast (gray background line)
    fig.add_trace(go.Scatter(
        x=ts_str, y=depths,
        mode="lines", line=dict(color="#aaaaaa", width=1),
        showlegend=False, name="full cast",
    ))

    # Selected span
    if span_start is not None and span_end is not None:
        sp_idx = list(range(int(span_start), min(int(span_end) + 1, len(df_profile))))
        if sp_idx:
            sp_df = df_profile.iloc[sp_idx]
            sp_ts = sp_df["timestamp"].astype(str).tolist()
            sp_d  = (-sp_df["depth"]).tolist()

            # Shaded region
            fig.add_vrect(
                x0=sp_ts[0], x1=sp_ts[-1],
                fillcolor="steelblue", opacity=0.18, line_width=0,
            )
            # Highlighted markers
            fig.add_trace(go.Scatter(
                x=sp_ts, y=sp_d,
                mode="markers", marker=dict(size=3, color="steelblue"),
                showlegend=False, name="selected",
            ))

    fig.update_layout(
        height=250,
        margin=dict(l=50, r=10, t=30, b=40),
        dragmode="zoom",
        selectdirection="h",
        xaxis_title="Time (UTC)",
        yaxis_title="Depth (m)",
        title_text="Depth vs Time – drag to select span",
        title_font_size=12,
        uirevision=f"{span_start}_{span_end}",
    )
    return fig


# ─────────────────────────────────────────────
# Leaflet map helper
# ─────────────────────────────────────────────

def build_map_markers(station_matches, current_index):
    markers = []
    keys = list(station_matches.keys())
    for i, key in enumerate(keys):
        si   = station_matches[key]["station_info"]
        lat  = si["startLat"]
        lon  = si["startLon"]
        if lat is None or lon is None:
            continue
        is_current = (i == current_index)
        color = "red" if is_current else ("orange" if si.get("time_corrected") else "blue")
        icon = dict(
            iconUrl=f"https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-{color}.png",
            shadowUrl="https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
            iconSize=[25, 41], iconAnchor=[12, 41], popupAnchor=[1, -34],
        )
        popup = dl.Popup(
            html.Div([
                html.Div(si["name"],
                         style={"fontWeight": "bold", "fontSize": "13px",
                                "marginBottom": "4px"}),
                html.Table([
                    html.Tr([html.Td("Activity", style={"color": "#888", "paddingRight": "8px"}),
                             html.Td(si["activityNumber"])]),
                    html.Tr([html.Td("Time",     style={"color": "#888", "paddingRight": "8px"}),
                             html.Td(str(si["startTime"])[:19])]),
                    html.Tr([html.Td("Lat",      style={"color": "#888", "paddingRight": "8px"}),
                             html.Td(f"{lat:.4f}°")]),
                    html.Tr([html.Td("Lon",      style={"color": "#888", "paddingRight": "8px"}),
                             html.Td(f"{lon:.4f}°")]),
                ], style={"fontSize": "12px", "borderCollapse": "collapse"}),
                dbc.Button("Select profile",
                           id={"type": "select-profile-btn", "index": i},
                           color="primary", size="sm",
                           style={"marginTop": "8px", "width": "100%"}),
            ], style={"minWidth": "160px"})
        )
        markers.append(
            dl.Marker(
                position=[lat, lon],
                icon=icon,
                id={"type": "station-marker", "index": i},
                children=popup,
            )
        )
    return markers


def map_center_zoom(station_matches):
    lats = [v["station_info"]["startLat"] for v in station_matches.values()
            if v["station_info"]["startLat"] is not None]
    lons = [v["station_info"]["startLon"] for v in station_matches.values()
            if v["station_info"]["startLon"] is not None]
    if not lats:
        return [60, 5], 5
    center = [np.mean(lats), np.mean(lons)]
    lat_r = max(lats) - min(lats)
    lon_r = max(lons) - min(lons)
    span  = max(lat_r, lon_r)
    zoom  = max(3, int(8 - np.log2(max(span, 0.01))))
    return center, zoom


# ─────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────

# Allow serving under a sub-path (e.g. /rsk2physchem/) via URL_PREFIX env var.
_url_prefix = os.getenv("URL_PREFIX", "/").rstrip("/") + "/"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    url_base_pathname=_url_prefix,
)
app.title = "CTD Profile Browser"

# Hidden stores for application state
stores = html.Div([
    dcc.Store(id="store-authenticated",  data=False),
    dcc.Store(id="store-station-matches", data={}),
    dcc.Store(id="store-rsk-df",         data={}),   # JSON via orient="split"
    dcc.Store(id="store-rsk-meta",       data={}),
    dcc.Store(id="store-current-index",  data=0),
    dcc.Store(id="store-excluded",       data=[]),    # list of df indices
    dcc.Store(id="store-npc",            data={}),
    dcc.Store(id="store-npc-meta",       data={}),
    dcc.Store(id="store-span-indices",   data=[]),
    dcc.Store(id="store-span-range",     data=[0, 0]),
    dcc.Store(id="store-cruise-times",   data={}),
    dcc.Store(id="store-tmpfiles",       data=[]),
])

login_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("CTD Profile Browser"), close_button=False),
    dbc.ModalBody([
        html.P("Enter the password to access this application.", className="text-muted"),
        dbc.InputGroup([
            dbc.InputGroupText(html.I(className="bi bi-lock")),
            dcc.Input(
                id="login-password-input",
                type="password",
                placeholder="Password",
                n_submit=0,
                className="form-control",
            ),
        ], className="mb-2"),
        html.Div(id="login-error-msg", className="text-danger small"),
    ]),
    dbc.ModalFooter(
        dbc.Button("Login", id="login-btn", color="primary", n_clicks=0),
    ),
], id="login-modal", is_open=True, backdrop="static", keyboard=False, centered=True)

left_panel = dbc.Card([
    dbc.CardBody([

        # ── File upload
        dbc.Label("Upload RSK Files from Hans Brattstrøm cruise", className="fw-bold"),
        dcc.Upload(
            id="upload-rsk",
            children=html.Div(["Drag & drop or ", html.A("select RSK files")]),
            style={
                "width": "100%", "height": "60px", "lineHeight": "60px",
                "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                "textAlign": "center", "marginBottom": "8px",
            },
            multiple=True,
        ),
        dcc.Loading(
            html.Div(id="upload-status", className="text-muted small mb-2"),
            type="circle", fullscreen=True,
            style={"backgroundColor": "rgba(0,0,0,0.3)"},
        ),

        html.Hr(),

        # ── Cruise parameters
        dbc.Label("Cruise Parameters", className="fw-bold"),
        dbc.InputGroup([
            dbc.InputGroupText("Cruise #"),
            dbc.Input(id="input-cruise-number",  placeholder="auto-filled", size="sm"),
        ], className="mb-1 input-group-sm"),
        dbc.InputGroup([
            dbc.InputGroupText("Vessel"),
            dbc.Input(id="input-vessel-name",    placeholder="auto-filled", size="sm"),
        ], className="mb-1 input-group-sm"),
        dbc.InputGroup([
            dbc.InputGroupText("Mission #"),
            dbc.Input(id="input-mission-number", placeholder="auto-filled", size="sm"),
        ], className="mb-1 input-group-sm"),
        dbc.InputGroup([
            dbc.InputGroupText("Platform #"),
            dbc.Input(id="input-platform",       placeholder="auto-filled", size="sm"),
        ], className="mb-2 input-group-sm"),

        html.Hr(),

        # ── Station navigation
        dbc.Label("Navigation", className="fw-bold"),
        html.Div(id="nav-label", className="text-center fw-bold mb-1"),
        dbc.ButtonGroup([
            dbc.Button("← Prev", id="btn-prev", color="secondary",
                       size="sm", disabled=True),
            dbc.Button("Next →", id="btn-next", color="secondary",
                       size="sm", disabled=True),
        ], className="w-100 mb-2"),

        html.Hr(),

        # ── Station info
        dbc.Label("Station Info", className="fw-bold"),
        html.Pre(id="station-info-text",
                 style={"fontSize": "11px", "maxHeight": "130px",
                        "overflowY": "auto", "background": "#f8f9fa",
                        "padding": "6px", "borderRadius": "4px"}),

        html.Hr(),

        # ── Export parameters
        dbc.Label("Export Parameters", className="fw-bold"),
        dbc.Checklist(
            id="checklist-params",
            options=[
                {"label": " Include Dissolved O₂", "value": "o2"},
                {"label": " Include Chlorophyll",   "value": "chl"},
            ],
            value=["o2", "chl"],
            className="mb-2",
        ),

        html.Hr(),

        # ── QC controls
        dbc.Label("QC Controls", className="fw-bold"),
        html.Div([
            dbc.Button("Clear Exclusions", id="btn-clear-excl",
                       color="warning", size="sm", className="me-1 mb-1"),
        ]),
        html.Div(id="excl-count-label",
                 className="text-danger small mb-2"),

        html.Hr(),

        # ── Actions
        dbc.Label("Actions", className="fw-bold"),
        dbc.Button("Download NPC File", id="btn-download-npc",
                   color="success", size="sm", className="w-100 mb-1", disabled=True),
        dbc.Button("Upload to PhysChem (S3)", id="btn-upload-s3",
                   color="primary", size="sm", className="w-100 mb-1", disabled=True),
        html.Div(id="action-status", className="small mt-1"),

        html.Div([
            html.A("Documentation & Code",
                   href="https://github.com/sebastianmenze/rsk2physchem",
                   target="_blank",
                   className="small text-muted"),
        ], className="mt-2"),

        dcc.Download(id="download-npc"),
    ]),
], style={"height": "100vh", "overflowY": "auto"})

right_panel = dbc.Card([
    dbc.CardBody([
        # ── Top row: Map (left) + Depth-time plot (right)
        dbc.Row([
            dbc.Col([
                dl.Map(
                    id="leaflet-map",
                    center=[60, 5],
                    zoom=5,
                    children=[
                        dl.TileLayer(
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                            attribution="© OpenStreetMap contributors",
                        ),
                        dl.LayerGroup(id="map-markers"),
                    ],
                    style={"height": "280px", "borderRadius": "6px"},
                ),
            ], width=6, className="pe-2"),
            dbc.Col([
                dcc.Loading(
                    dcc.Graph(
                        id="timeseries-plot",
                        config={"displayModeBar": True, "scrollZoom": False,
                                "modeBarButtonsToAdd": ["select2d"]},
                        style={"height": "250px"},
                    ),
                ),
                # Range slider for span selection
                html.Div([
                    dcc.RangeSlider(
                        id="span-slider",
                        min=0, max=100, step=1,
                        value=[0, 100],
                        marks={},
                        allowCross=False,
                        allow_direct_input=False,
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="mt-1",
                    ),
                ], style={"paddingLeft": "50px", "paddingRight": "10px"}),
            ], width=6, className="ps-0"),
        ], style={"flexShrink": "0"}),

        # ── Status bar (fixed, above profile plots)
        html.Div([
            html.Div(id="npc-loading-target", style={"display": "inline-block", "marginRight": "6px"}),
            html.Span(id="status-bar", className="text-info small"),
        ], style={"flexShrink": "0", "minHeight": "20px", "paddingTop": "2px",
                  "display": "flex", "alignItems": "center"}),

        # ── Profile plots – fill all remaining vertical space
        dcc.Loading(
            dcc.Graph(
                id="profile-plot",
                config={"displayModeBar": True, "scrollZoom": True,
                        "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                        "responsive": True},
                style={"height": "600px", "minHeight": "600px"},
            ),
            style={"flexShrink": "0", "display": "flex",
                   "flexDirection": "column"},
        ),
    ], style={"display": "flex", "flexDirection": "column",
              "height": "100%", "padding": "8px", "gap": "4px",
              "overflowY": "auto"}),
], style={"height": "calc(100vh - 16px)", "overflow": "hidden"})

app.layout = dbc.Container([
    stores,
    login_modal,
    dbc.Row([
        dbc.Col(left_panel,  width=3, style={"padding": "0"}),
        dbc.Col(right_panel, width=9, style={"padding": "0 0 0 8px"}),
    ], style={"height": "calc(100vh - 16px)"}),
], fluid=True, style={"padding": "8px"})


# ─────────────────────────────────────────────
# Login / password-gate callback
# ─────────────────────────────────────────────

@app.callback(
    Output("login-modal",         "is_open"),
    Output("store-authenticated", "data"),
    Output("login-error-msg",     "children"),
    Input("login-btn",            "n_clicks"),
    Input("login-password-input", "n_submit"),
    State("login-password-input", "value"),
    State("store-authenticated",  "data"),
    prevent_initial_call=True,
)
def check_password(n_clicks, n_submit, entered, already_authed):
    if already_authed:
        return False, True, ""
    if not entered:
        return True, False, "Please enter a password."
    expected = os.getenv("PASSWORD") or PASSWORD
    if expected and entered.strip() == expected.strip():
        return False, True, ""
    return True, False, "Incorrect password."


# ─────────────────────────────────────────────
# Server-side upload accumulator
# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

@app.callback(
    Output("store-station-matches", "data"),
    Output("store-rsk-df",          "data"),
    Output("store-rsk-meta",        "data"),
    Output("store-cruise-times",    "data"),
    Output("store-tmpfiles",        "data"),
    Output("upload-status",         "children"),
    Output("input-cruise-number",   "value"),
    Output("input-vessel-name",     "value"),
    Output("input-mission-number",  "value"),
    Output("input-platform",        "value"),
    Input("upload-rsk", "contents"),
    State("upload-rsk", "filename"),
    prevent_initial_call=True,
)
def process_uploaded_files(contents_list, filenames):
    if not contents_list:
        return no_update

    if not PYRSK_AVAILABLE:
        return ({}, {}, {}, {}, [], "Error: pyrsktools not installed",
                "", "", "", "")

    # Normalise to lists (single-file upload may pass bare strings)
    if isinstance(contents_list, str):
        contents_list = [contents_list]
    if not isinstance(filenames, list):
        filenames = [filenames] if filenames else []
    while len(filenames) < len(contents_list):
        filenames.append(f"file_{len(filenames)+1}.rsk")

    # Save uploads to temp files
    tmp_paths = []
    for content, fname in zip(contents_list, filenames):
        _, b64 = content.split(",", 1)
        raw = base64.b64decode(b64)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".rsk")
        tmp.write(raw)
        tmp.close()
        tmp_paths.append(tmp.name)

    print(f"[upload] processing {len(tmp_paths)} file(s)", flush=True)

    try:
        # Read RSK files
        df_all   = pd.DataFrame()
        rsk_meta = {}
        for path in tmp_paths:
            df, meta = process_rsk_file(path)
            df_all   = pd.concat([df_all, df], ignore_index=True)
            rsk_meta = meta  # take last file's instrument meta

        df_all = df_all.sort_values("timestamp").reset_index(drop=True)

        # Extract date range
        ts = pd.to_datetime(df_all["timestamp"])
        ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
        t_min = ts.min()
        t_max = ts.max()

        start_buf = (t_min - pd.Timedelta(days=1))
        end_buf   = (t_max + pd.Timedelta(days=1))

        after_str  = start_buf.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        before_str = end_buf.strftime("%Y-%m-%dT%H:%M:%S.999Z")

        # Fetch cruise & activities from API
        df_cruises = get_cruises_from_api()
        cruise_number = vessel_name = mission_number = platform = ""
        if len(df_cruises):
            matched = match_cruise_by_dates(t_min, t_max, df_cruises)
            if matched:
                cruise_number = str(matched.get("cruiseNumber", ""))
                vessel_name   = str(matched.get("vesselName",   ""))
                platform      = str(matched.get("platform",     ""))
                year          = t_min.year
                mission_number = get_mission_number_from_physchem(cruise_number, platform, year)

        df_tk = get_activities_from_api(after_str, before_str)

        cruise_times = {}
        if len(df_tk):
            cruise_times = {
                "start": df_tk["startTime"].min().isoformat(),
                "end":   df_tk["endTime"].max().isoformat(),
            }

        # Match stations
        if len(df_tk):
            station_matches = get_station_indices_for_ctd(df_all, df_tk)
        else:
            station_matches = {}

        n_files    = len(tmp_paths)
        n_stations = len(station_matches)
        status_msg = (
            f"Loaded {n_files} file(s) · "
            f"{len(df_all):,} data points · "
            f"{n_stations} CTD stations matched"
        )

        # Serialise RSK data
        df_json = df_all.to_json(date_format="iso", orient="split")

        return (
            station_matches,
            df_json,
            rsk_meta,
            cruise_times,
            tmp_paths,
            status_msg,
            cruise_number,
            vessel_name,
            mission_number,
            platform,
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        return ({}, {}, {}, {}, tmp_paths,
                f"Error: {e}", "", "", "", "")


# ── Navigation (buttons + map popup "Select profile" button)
@app.callback(
    Output("store-current-index", "data"),
    Output("store-excluded",      "data"),
    Input("btn-prev",  "n_clicks"),
    Input("btn-next",  "n_clicks"),
    Input("btn-clear-excl", "n_clicks"),
    Input({"type": "select-profile-btn", "index": ALL}, "n_clicks"),
    State("store-current-index",  "data"),
    State("store-station-matches","data"),
    State("store-excluded",       "data"),
    prevent_initial_call=True,
)
def navigate(n_prev, n_next, n_clear, select_clicks,
             current_idx, station_matches, excluded):
    triggered = ctx.triggered_id
    keys = list(station_matches.keys()) if station_matches else []
    n = len(keys)
    if triggered == "btn-prev":
        return max(0, current_idx - 1), []
    if triggered == "btn-next":
        return min(n - 1, current_idx + 1), []
    if triggered == "btn-clear-excl":
        return current_idx, []
    if isinstance(triggered, dict) and triggered.get("type") == "select-profile-btn":
        # Guard against ghost fires: when update_display rebuilds map-markers,
        # Dash re-mounts the select-profile-btn components and fires this
        # callback with n_clicks=None (not a real click).  Only act when the
        # triggering button has an actual positive click count.
        triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
        if not triggered_value:
            return no_update, no_update
        return triggered["index"], []
    return current_idx, excluded


# ── Collect excluded points from plot selections
@app.callback(
    Output("store-excluded", "data", allow_duplicate=True),
    Input("profile-plot", "selectedData"),
    State("store-excluded", "data"),
    State("store-current-index", "data"),
    State("store-station-matches", "data"),
    prevent_initial_call=True,
)
def collect_exclusions(selected_data, excluded, current_idx, station_matches):
    if not selected_data or not selected_data.get("points"):
        return no_update
    new_excl = set(excluded or [])
    for pt in selected_data["points"]:
        cd = pt.get("customdata")
        if cd is not None:
            new_excl.add(int(cd))
    return list(new_excl)


# ── Initialise slider when station changes
@app.callback(
    Output("span-slider",       "min"),
    Output("span-slider",       "max"),
    Output("span-slider",       "marks"),
    Output("span-slider",       "value"),
    Input("store-current-index",   "data"),
    State("store-rsk-df",          "data"),
    State("store-station-matches", "data"),
    prevent_initial_call=True,
)
def init_slider(current_idx, rsk_df_json, station_matches):
    if not station_matches or not rsk_df_json:
        return 0, 100, {}, [0, 100]

    keys       = list(station_matches.keys())
    data       = station_matches[keys[current_idx]]
    df_all     = pd.read_json(StringIO(rsk_df_json), orient="split")
    df_profile = df_all.loc[data["df_rsk_indices"]].copy().reset_index(drop=True)
    N          = len(df_profile)
    if N == 0:
        return 0, 0, {}, [0, 0]

    ix_down = detect_downcast(df_profile)
    if ix_down.any():
        down_idx   = df_profile.index[ix_down].tolist()
        span_start = min(down_idx)
        span_end   = max(down_idx)
    else:
        span_start, span_end = 0, N - 1

    ts = df_profile["timestamp"]
    marks = {}
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = int(frac * (N - 1))
        t   = pd.Timestamp(ts.iloc[idx])
        marks[idx] = {"label": t.strftime("%H:%M"), "style": {"fontSize": "10px"}}

    return 0, N - 1, marks, [span_start, span_end]


# ── Slider interaction → update span range store
@app.callback(
    Output("store-span-range", "data", allow_duplicate=True),
    Input("span-slider", "value"),
    prevent_initial_call=True,
)
def update_span_from_slider(slider_value):
    if slider_value is None:
        return no_update
    return slider_value


# ── Timeseries drag-select → update span range store + sync slider
@app.callback(
    Output("store-span-range", "data", allow_duplicate=True),
    Output("span-slider",      "value", allow_duplicate=True),
    Input("timeseries-plot",   "relayoutData"),
    State("store-current-index",   "data"),
    State("store-rsk-df",          "data"),
    State("store-station-matches", "data"),
    State("store-span-range",      "data"),
    prevent_initial_call=True,
)
def update_span_from_timeseries(relayout_data, current_idx, rsk_df_json,
                                 station_matches, current_span):
    if not relayout_data or not station_matches or not rsk_df_json:
        return no_update, no_update

    x0 = relayout_data.get("selections[0].x0")
    x1 = relayout_data.get("selections[0].x1")
    if not x0 or not x1:
        return no_update, no_update

    keys       = list(station_matches.keys())
    data       = station_matches[keys[current_idx]]
    df_all     = pd.read_json(StringIO(rsk_df_json), orient="split")
    df_profile = df_all.loc[data["df_rsk_indices"]].copy().reset_index(drop=True)

    ts = pd.to_datetime(df_profile["timestamp"])
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")

    def _parse_ts(s):
        t = pd.Timestamp(s)
        return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")

    t0, t1 = sorted([_parse_ts(x0), _parse_ts(x1)])
    mask    = (ts >= t0) & (ts <= t1)
    indices = df_profile.index[mask].tolist()
    if not indices:
        return no_update, no_update

    span = [min(indices), max(indices)]
    return span, span


# ── Compute NPC whenever span range or exclusions change
@app.callback(
    Output("store-npc",            "data"),
    Output("store-npc-meta",       "data"),
    Output("store-span-indices",   "data"),
    Output("npc-loading-target",   "children"),
    Input("store-span-range",   "data"),
    Input("store-excluded",     "data"),
    Input("checklist-params",   "value"),
    State("store-current-index",  "data"),
    State("store-station-matches","data"),
    State("store-rsk-df",         "data"),
    State("store-rsk-meta",       "data"),
    State("store-cruise-times",   "data"),
    State("input-cruise-number",  "value"),
    State("input-vessel-name",    "value"),
    State("input-mission-number", "value"),
    State("input-platform",       "value"),
    prevent_initial_call=True,
)
def compute_npc(span_range, excluded, param_vals,
                current_idx, station_matches, rsk_df_json, rsk_meta,
                cruise_times,
                cruise_number, vessel_name, mission_number, platform):
    # Always include station index + unique timestamp in meta so update_profile
    # can detect and discard stale results from a previous station.
    def _meta_sentinel(idx):
        return json.dumps({"_station_idx": idx, "_ts": str(uuid.uuid1())})

    if not station_matches or not rsk_df_json or not span_range:
        return "{}", _meta_sentinel(current_idx), [], ""

    span_start, span_end = span_range
    keys        = list(station_matches.keys())
    station_key = keys[current_idx]
    data        = station_matches[station_key]
    df_indices  = data["df_rsk_indices"]

    df_all      = pd.read_json(StringIO(rsk_df_json), orient="split")
    df_profile  = df_all.loc[df_indices].copy().reset_index(drop=True)

    new_span = list(range(int(span_start), min(int(span_end) + 1, len(df_profile))))
    if not new_span:
        return "{}", _meta_sentinel(current_idx), [], ""

    ct_start = cruise_times.get("start") if cruise_times else None
    ct_end   = cruise_times.get("end")   if cruise_times else None

    try:
        df_npc, meta = calculate_df_npc(
            df_profile, new_span, set(excluded or []),
            "o2" in (param_vals or []),
            "chl" in (param_vals or []),
            ct_start, ct_end,
            cruise_number or "", vessel_name or "",
            mission_number or "", platform or "",
            current_idx + 1, rsk_meta or {},
            data["station_info"],
        )
    except Exception as exc:
        print(f"[compute_npc] ERROR station={current_idx} span={span_range}: {exc}", flush=True)
        return "{}", _meta_sentinel(current_idx), [], ""

    npc_json = df_npc.to_json(orient="split") if len(df_npc) else "{}"
    try:
        meta_json = json.dumps({**meta, "_station_idx": current_idx, "_ts": str(uuid.uuid1())})
    except (TypeError, ValueError) as exc:
        print(f"[compute_npc] meta serialisation error (station={current_idx}): {exc}", flush=True)
        meta_json = _meta_sentinel(current_idx)
    return npc_json, meta_json, new_span, ""


# ── Timeseries figure (depth vs time with span highlight)
@app.callback(
    Output("timeseries-plot", "figure"),
    Input("store-current-index",   "data"),
    Input("span-slider",           "value"),
    State("store-rsk-df",          "data"),
    State("store-station-matches", "data"),
)
def update_timeseries(current_idx, slider_value, rsk_df_json, station_matches):
    empty = go.Figure()
    empty.update_layout(
        height=250, margin=dict(l=50, r=10, t=30, b=40),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="No data", showarrow=False)],
    )
    if not station_matches or not rsk_df_json:
        return empty

    try:
        keys       = list(station_matches.keys())
        data       = station_matches[keys[current_idx]]
        df_all     = pd.read_json(StringIO(rsk_df_json), orient="split")
        df_profile = df_all.loc[data["df_rsk_indices"]].copy().reset_index(drop=True)
        span_start = int(slider_value[0]) if slider_value else None
        span_end   = int(slider_value[1]) if slider_value else None
        return build_timeseries_figure(df_profile, span_start, span_end)
    except Exception as exc:
        import traceback
        print(f"[timeseries] ERROR station={current_idx} slider={slider_value}: {exc}", flush=True)
        traceback.print_exc()
        return empty


# ── Main display: map + UI state (no plots)
@app.callback(
    Output("leaflet-map",       "center"),
    Output("leaflet-map",       "zoom"),
    Output("map-markers",       "children"),
    Output("station-info-text", "children"),
    Output("nav-label",         "children"),
    Output("btn-prev",          "disabled"),
    Output("btn-next",          "disabled"),
    Output("excl-count-label",  "children"),
    Output("status-bar",        "children"),
    Input("store-station-matches", "data"),
    Input("store-current-index",   "data"),
    Input("store-excluded",        "data"),
    State("store-npc",             "data"),   # State — NPC count shown, but NPC changes
                                              # must NOT re-render map markers (would
                                              # ghost-fire the navigate callback)
)
def update_display(station_matches, current_idx, excluded, npc_json):
    if not station_matches:
        return ([60, 5], 5, [], "No data loaded", "─", True, True, "", "")

    keys  = list(station_matches.keys())
    n     = len(keys)
    key   = keys[current_idx]
    data  = station_matches[key]
    si    = data["station_info"]

    center, zoom = map_center_zoom(station_matches)
    markers      = build_map_markers(station_matches, current_idx)

    df_npc_len = 0
    if npc_json and npc_json != "{}":
        try:
            df_npc_len = len(pd.read_json(StringIO(npc_json), orient="split"))
        except Exception:
            pass

    corr_note = ""
    if si.get("time_corrected"):
        corr_note = (f"\n⚠ TIME CORRECTED\n"
                     f"  Original: {si['original_startTime']}\n"
                     f"  {si['correction_info']}")
    info = (
        f"Name:     {si['name']}\n"
        f"Activity: {si['activityNumber']}\n"
        f"Start:    {si['startTime']}\n"
        f"End:      {si['endTime']}\n"
        f"Lat:      {si['startLat']:.5f}°N\n"
        f"Lon:      {si['startLon']:.5f}°E\n"
        f"Comment:  {si['comment']}\n"
        f"Points:   {data['n_datapoints']}"
        + corr_note
    )

    nav_label  = f"Profile {current_idx+1} / {n}"
    excl_count = f"Excluded: {len(excluded or [])} points"
    status_msg = f"Station {current_idx+1}/{n} · {df_npc_len} depth bins computed"

    return (
        center, zoom, markers,
        info, nav_label,
        current_idx <= 0, current_idx >= n - 1,
        excl_count, status_msg,
    )


# ── Status bar NPC-count update (store-npc as Input here, NOT in update_display,
# so the map markers are never re-built when the NPC changes — which would
# ghost-fire the pattern-matching navigate callback and reset to station 0).
@app.callback(
    Output("status-bar", "children", allow_duplicate=True),
    Input("store-npc",             "data"),
    State("store-current-index",   "data"),
    State("store-station-matches", "data"),
    prevent_initial_call=True,
)
def update_status_npc(npc_json, current_idx, station_matches):
    if not station_matches:
        return no_update
    n = len(station_matches)
    df_npc_len = 0
    if npc_json and npc_json != "{}":
        try:
            df_npc_len = len(pd.read_json(StringIO(npc_json), orient="split"))
        except Exception:
            pass
    return f"Station {current_idx+1}/{n} · {df_npc_len} depth bins computed"


# ── Profile figure — part 1: immediate render on upload or lasso
# Fires from store-station-matches (upload) and store-excluded (lasso) so the
# user sees raw data / excluded-point markers without waiting for compute_npc.
# Uses whatever NPC is currently in store-npc (may be stale between stations;
# part 2 will overwrite with the validated fresh NPC once it is ready).
@app.callback(
    Output("profile-plot", "figure"),
    Input("store-station-matches", "data"),
    Input("store-excluded",        "data"),
    State("store-current-index",   "data"),
    State("store-span-range",      "data"),
    State("store-rsk-df",          "data"),
    State("store-npc",             "data"),
)
def update_profile_immediate(station_matches, excluded,
                             current_idx, span_range, rsk_df_json, npc_json):
    return _render_profile(station_matches, excluded, npc_json,
                           current_idx, span_range, rsk_df_json)


# ── Profile figure — part 2: render with validated fresh NPC
# Fires from store-npc-meta (always has a new UUID after each compute_npc run).
# Checks _station_idx in the meta: if it doesn't match the currently displayed
# station the result is stale (a queued callback from a previous station) and
# we skip the render entirely to avoid overwriting with wrong data.
@app.callback(
    Output("profile-plot", "figure", allow_duplicate=True),
    Input("store-npc-meta",        "data"),
    State("store-npc",             "data"),
    State("store-current-index",   "data"),
    State("store-excluded",        "data"),
    State("store-span-range",      "data"),
    State("store-rsk-df",          "data"),
    State("store-station-matches", "data"),
    prevent_initial_call=True,
)
def update_profile_with_npc(npc_meta_json, npc_json, current_idx, excluded,
                             span_range, rsk_df_json, station_matches):
    # Validate that this NPC result belongs to the station currently on screen.
    try:
        meta = json.loads(npc_meta_json) if isinstance(npc_meta_json, str) else (npc_meta_json or {})
        station_in_meta = meta.get("_station_idx")
        if station_in_meta != current_idx:
            print(f"[profile_npc] stale skip: meta_idx={station_in_meta} current={current_idx}", flush=True)
            return no_update   # stale result from a previous station — skip
    except Exception as exc:
        print(f"[profile_npc] meta parse error: {exc!r}", flush=True)
        return no_update
    print(f"[profile_npc] rendering station={current_idx} span={span_range}", flush=True)
    return _render_profile(station_matches, excluded, npc_json,
                           current_idx, span_range, rsk_df_json)


def _render_profile(station_matches, excluded, npc_json,
                    current_idx, span_range, rsk_df_json):
    """Shared rendering logic for both profile callbacks."""
    empty = go.Figure()
    empty.update_layout(
        margin=dict(l=40, r=20, t=50, b=40),
        autosize=True,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="Upload RSK files to begin",
                          showarrow=False, font=dict(size=18))],
    )
    if not station_matches or not rsk_df_json:
        return empty

    try:
        keys       = list(station_matches.keys())
        key        = keys[current_idx]
        data       = station_matches[key]
        df_all     = pd.read_json(StringIO(rsk_df_json), orient="split")
        df_profile = df_all.loc[data["df_rsk_indices"]].copy().reset_index(drop=True)

        df_npc = pd.DataFrame()
        npc_str = npc_json if isinstance(npc_json, str) else None
        if npc_str and npc_str != "{}":
            try:
                df_npc = pd.read_json(StringIO(npc_str), orient="split")
            except Exception:
                pass

        span_start = span_range[0] if span_range else 0
        span_end   = span_range[1] if span_range else len(df_profile) - 1

        fig = build_profile_figure(
            df_profile, span_start, span_end, df_npc, set(excluded or [])
        )
        fig.update_layout(title_text=key)
        return fig
    except Exception as exc:
        import traceback
        print(f"[_render_profile] ERROR station={current_idx} span={span_range}: {exc}", flush=True)
        traceback.print_exc()
        return empty


# ── Download NPC
@app.callback(
    Output("download-npc",  "data"),
    Output("action-status", "children", allow_duplicate=True),
    Input("btn-download-npc", "n_clicks"),
    State("store-span-range",      "data"),
    State("store-excluded",        "data"),
    State("checklist-params",      "value"),
    State("store-current-index",   "data"),
    State("store-station-matches", "data"),
    State("store-rsk-df",          "data"),
    State("store-rsk-meta",        "data"),
    State("store-cruise-times",    "data"),
    State("input-cruise-number",   "value"),
    State("input-vessel-name",     "value"),
    State("input-mission-number",  "value"),
    State("input-platform",        "value"),
    prevent_initial_call=True,
)
def download_npc(n_clicks, span_range, excluded, param_vals,
                 current_idx, station_matches, rsk_df_json, rsk_meta,
                 cruise_times, cruise_number, vessel_name, mission_number, platform):
    if not station_matches or not rsk_df_json or not span_range:
        return no_update, "No NPC data available – select a span first."
    try:
        span_start, span_end = span_range
        keys        = list(station_matches.keys())
        key         = keys[current_idx]
        data        = station_matches[key]
        df_all      = pd.read_json(StringIO(rsk_df_json), orient="split")
        df_profile  = df_all.loc[data["df_rsk_indices"]].copy().reset_index(drop=True)
        new_span    = list(range(int(span_start), min(int(span_end) + 1, len(df_profile))))
        if not new_span:
            return no_update, "No NPC data available – select a span first."
        ct_start = cruise_times.get("start") if cruise_times else None
        ct_end   = cruise_times.get("end")   if cruise_times else None
        df_npc, meta = calculate_df_npc(
            df_profile, new_span, set(excluded or []),
            "o2" in (param_vals or []),
            "chl" in (param_vals or []),
            ct_start, ct_end,
            cruise_number or "", vessel_name or "",
            mission_number or "", platform or "",
            current_idx + 1, rsk_meta or {},
            data["station_info"],
        )
        fname   = _npc_filename(cruise_number, ct_start)
        content = npc_to_string(meta, df_npc)
        return (dict(content=content, filename=fname, type="text/plain"),
                f"Downloaded {fname}")
    except Exception as e:
        return no_update, f"Download error: {e}"


# ── Enable/disable action buttons based on whether a valid NPC exists
@app.callback(
    Output("btn-download-npc", "disabled"),
    Output("btn-upload-s3",    "disabled"),
    Output("action-status",    "children"),
    Input("store-npc",             "data"),
    Input("store-npc-meta",        "data"),
    Input("input-cruise-number",   "value"),
    Input("input-vessel-name",     "value"),
    Input("input-mission-number",  "value"),
    Input("input-platform",        "value"),
    prevent_initial_call=True,
)
def check_physchem_on_profile_change(npc_json, meta_json,
                                     cruise_number, vessel_name,
                                     mission_number, platform):
    no_npc = not npc_json or npc_json == "{}"
    if no_npc:
        return True, True, ""
    fields_complete = all([cruise_number, vessel_name, mission_number, platform])
    # NPC exists – check if already in PhysChem
    try:
        meta = json.loads(meta_json) if meta_json and meta_json != "{}" else {}
    except Exception:
        meta = {}
    if meta and check_if_operation_in_physchem(meta):
        return False, True, "⚠ This profile is already uploaded to PhysChem."
    return False, not fields_complete, ""


# ── Upload to S3
@app.callback(
    Output("action-status", "children", allow_duplicate=True),
    Input("btn-upload-s3",  "n_clicks"),
    State("store-span-range",      "data"),
    State("store-excluded",        "data"),
    State("checklist-params",      "value"),
    State("store-current-index",   "data"),
    State("store-station-matches", "data"),
    State("store-rsk-df",          "data"),
    State("store-rsk-meta",        "data"),
    State("store-cruise-times",    "data"),
    State("input-cruise-number",   "value"),
    State("input-vessel-name",     "value"),
    State("input-mission-number",  "value"),
    State("input-platform",        "value"),
    prevent_initial_call=True,
)
def upload_to_s3(n_clicks, span_range, excluded, param_vals,
                 current_idx, station_matches, rsk_df_json, rsk_meta,
                 cruise_times, cruise_number, vessel_name, mission_number, platform):
    if not BOTO3_AVAILABLE:
        return "boto3 not installed – cannot upload."
    if not station_matches or not rsk_df_json or not span_range:
        return "No NPC data available – select a span first."
    try:
        span_start, span_end = span_range
        keys        = list(station_matches.keys())
        data        = station_matches[keys[current_idx]]
        df_all      = pd.read_json(StringIO(rsk_df_json), orient="split")
        df_profile  = df_all.loc[data["df_rsk_indices"]].copy().reset_index(drop=True)
        new_span    = list(range(int(span_start), min(int(span_end) + 1, len(df_profile))))
        if not new_span:
            return "No NPC data available – select a span first."
        ct_start = cruise_times.get("start") if cruise_times else None
        ct_end   = cruise_times.get("end")   if cruise_times else None
        df_npc, meta = calculate_df_npc(
            df_profile, new_span, set(excluded or []),
            "o2" in (param_vals or []),
            "chl" in (param_vals or []),
            ct_start, ct_end,
            cruise_number or "", vessel_name or "",
            mission_number or "", platform or "",
            current_idx + 1, rsk_meta or {},
            data["station_info"],
        )

        os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"]  = "when_required"
        os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"]  = "when_required"

        s3 = boto3.resource(
            service_name="s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".npc") as f:
            tmp_path = f.name
        npc_write(meta, df_npc, tmp_path)

        ct_start_s3 = cruise_times.get("start") if cruise_times else None
        fname_s3    = _npc_filename(cruise_number, ct_start_s3)
        dest = f"{S3_DEST_PREFIX.rstrip('/')}/{fname_s3}"
        with open(tmp_path, "rb") as fh:
            s3.Bucket(S3_BUCKET).put_object(Key=dest, Body=fh)
        os.unlink(tmp_path)
        return f"Uploaded successfully → {dest}"
    except Exception as e:
        return f"Upload failed: {e}"


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
