"""
CTD Profile Browser - Dash Web Application
Ports the PyQt5 CTD Profile Browser to a Dash web app with Leaflet maps.
"""

import os
import re
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
from dash import dcc, html, Input, Output, State, callback, no_update, ctx
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


# ─────────────────────────────────────────────
# Data helpers  (ported from original script)
# ─────────────────────────────────────────────

def get_cruises_from_api(base_url="http://toktlogger-hansb.hi.no/api/cruises/all"):
    try:
        resp = requests.get(base_url, params={"format": "json"}, timeout=10)
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


def get_activities_from_api(after, before,
                             base_url="http://toktlogger-hansb.hi.no/api/activities/inPeriod"):
    resp = requests.get(base_url,
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

def build_profile_figure(df_profile, ix_down, df_npc, excluded_indices):
    """Build the 4-panel + time-series profile figure."""
    has_o2  = "dissolved_o2_concentration" in df_profile.columns and \
              df_profile["dissolved_o2_concentration"].dropna().any()
    has_chl = "chlorophyll" in df_profile.columns and \
              df_profile["chlorophyll"].dropna().any()

    fig = make_subplots(
        rows=3, cols=4,
        row_heights=[0.4, 0.4, 0.2],
        specs=[[{}, {}, {}, {}],
               [{}, {}, {}, {}],
               [{"colspan": 4}, None, None, None]],
        subplot_titles=("Temperature", "Salinity",
                        "Dissolved O₂", "Chlorophyll",
                        "", "", "", "",
                        "Depth vs Time (drag to select span)"),
    )

    excl = [i for i in excluded_indices if i in df_profile.index]
    keep = [i for i in df_profile.index if i not in excluded_indices]

    color_keep = df_profile.loc[keep].index.values.tolist() if keep else []
    color_excl = ["red"] * len(excl)

    # ── Temperature
    if keep:
        fig.add_trace(go.Scatter(
            x=df_profile.loc[keep, "temperature"].tolist(),
            y=(-df_profile.loc[keep, "depth"]).tolist(),
            mode="markers", marker=dict(size=4, color=color_keep, colorscale="Viridis"),
            name="raw", showlegend=False,
            customdata=keep, hovertemplate="T=%{x:.3f}°C d=%{y:.1f}m<extra></extra>",
        ), row=1, col=1)
    if excl:
        fig.add_trace(go.Scatter(
            x=df_profile.loc[excl, "temperature"].tolist(),
            y=(-df_profile.loc[excl, "depth"]).tolist(),
            mode="markers", marker=dict(size=6, color="red", symbol="x"),
            name="excluded", showlegend=False,
        ), row=1, col=1)
    if len(df_npc) and "TEMP.value" in df_npc.columns:
        fig.add_trace(go.Scatter(
            x=df_npc["TEMP.value"].tolist(), y=(-df_npc["DEPTH.value"]).tolist(),
            mode="lines", line=dict(color="red", width=2), name="binned",
            showlegend=False,
        ), row=1, col=1)

    # ── Salinity
    if keep:
        fig.add_trace(go.Scatter(
            x=df_profile.loc[keep, "salinity"].tolist(),
            y=(-df_profile.loc[keep, "depth"]).tolist(),
            mode="markers", marker=dict(size=4, color=color_keep, colorscale="Viridis"),
            name="raw", showlegend=False,
            customdata=keep, hovertemplate="S=%{x:.3f} PSU d=%{y:.1f}m<extra></extra>",
        ), row=1, col=2)
    if excl:
        fig.add_trace(go.Scatter(
            x=df_profile.loc[excl, "salinity"].tolist(),
            y=(-df_profile.loc[excl, "depth"]).tolist(),
            mode="markers", marker=dict(size=6, color="red", symbol="x"),
            name="excluded", showlegend=False,
        ), row=1, col=2)
    if len(df_npc) and "PSAL.value" in df_npc.columns:
        fig.add_trace(go.Scatter(
            x=df_npc["PSAL.value"].tolist(), y=(-df_npc["DEPTH.value"]).tolist(),
            mode="lines", line=dict(color="red", width=2), showlegend=False,
        ), row=1, col=2)

    # ── O₂
    if has_o2 and keep:
        fig.add_trace(go.Scatter(
            x=df_profile.loc[keep, "dissolved_o2_concentration"].tolist(),
            y=(-df_profile.loc[keep, "depth"]).tolist(),
            mode="markers", marker=dict(size=4, color=color_keep, colorscale="Viridis"),
            showlegend=False,
        ), row=1, col=3)
    if len(df_npc) and "DOX.value" in df_npc.columns:
        fig.add_trace(go.Scatter(
            x=df_npc["DOX.value"].tolist(), y=(-df_npc["DEPTH.value"]).tolist(),
            mode="lines", line=dict(color="red", width=2), showlegend=False,
        ), row=1, col=3)
    if not has_o2:
        fig.add_annotation(text="No O₂ data", xref="x3 domain", yref="y3 domain",
                           x=0.5, y=0.5, showarrow=False, row=1, col=3)

    # ── Chlorophyll
    if has_chl and keep:
        fig.add_trace(go.Scatter(
            x=df_profile.loc[keep, "chlorophyll"].tolist(),
            y=(-df_profile.loc[keep, "depth"]).tolist(),
            mode="markers", marker=dict(size=4, color=color_keep, colorscale="Viridis"),
            showlegend=False,
        ), row=1, col=4)
    if len(df_npc) and "ChlA_SENS.value" in df_npc.columns:
        fig.add_trace(go.Scatter(
            x=df_npc["ChlA_SENS.value"].tolist(), y=(-df_npc["DEPTH.value"]).tolist(),
            mode="lines", line=dict(color="red", width=2), showlegend=False,
        ), row=1, col=4)
    if not has_chl:
        fig.add_annotation(text="No Chl data", xref="x4 domain", yref="y4 domain",
                           x=0.5, y=0.5, showarrow=False, row=1, col=4)

    # ── Time series (row 3)
    fig.add_trace(go.Scatter(
        x=df_profile["timestamp"].astype(str).tolist(),
        y=(-df_profile["depth"]).tolist(),
        mode="lines", line=dict(color="black", width=1),
        showlegend=False,
    ), row=3, col=1)
    down_df = df_profile[ix_down]
    if len(down_df):
        fig.add_trace(go.Scatter(
            x=down_df["timestamp"].astype(str).tolist(),
            y=(-down_df["depth"]).tolist(),
            mode="markers", marker=dict(size=3, color="red"),
            showlegend=False,
        ), row=3, col=1)

    # Axis labels
    for col, xlabel in [(1, "°C"), (2, "PSU"), (3, "µmol/l"), (4, "µg/l")]:
        fig.update_xaxes(title_text=xlabel, row=1, col=col)
        fig.update_yaxes(title_text="Depth (m)", row=1, col=col)

    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)
    fig.update_yaxes(title_text="Depth (m)",  row=3, col=1)

    fig.update_layout(
        height=700,
        margin=dict(l=40, r=20, t=60, b=40),
        dragmode="select",
        selectdirection="h",   # horizontal span selection on time axis
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
        popup_text = (
            f"<b>{si['name']}</b><br>"
            f"Activity: {si['activityNumber']}<br>"
            f"{si['startTime']}<br>"
            f"Lat: {lat:.4f}  Lon: {lon:.4f}"
        )
        markers.append(
            dl.Marker(
                position=[lat, lon],
                icon=icon,
                id={"type": "station-marker", "index": i},
                children=dl.Popup(html.Div(
                    dangerouslyAllowHTML=True,
                    children=popup_text,
                )),
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

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "CTD Profile Browser"

# Hidden stores for application state
stores = html.Div([
    dcc.Store(id="store-station-matches", data={}),
    dcc.Store(id="store-rsk-df",         data={}),   # JSON via orient="split"
    dcc.Store(id="store-rsk-meta",       data={}),
    dcc.Store(id="store-current-index",  data=0),
    dcc.Store(id="store-excluded",       data=[]),    # list of df indices
    dcc.Store(id="store-npc",            data={}),
    dcc.Store(id="store-npc-meta",       data={}),
    dcc.Store(id="store-span-indices",   data=[]),
    dcc.Store(id="store-cruise-times",   data={}),
    dcc.Store(id="store-tmpfiles",       data=[]),
])

left_panel = dbc.Card([
    dbc.CardHeader(html.H5("CTD Profile Browser", className="mb-0")),
    dbc.CardBody([

        # ── File upload
        dbc.Label("Upload RSK Files", className="fw-bold"),
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
        html.Div(id="upload-status", className="text-muted small mb-2"),

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
                   color="success", size="sm", className="w-100 mb-1"),
        dbc.Button("Upload to PhysChem (S3)", id="btn-upload-s3",
                   color="primary", size="sm", className="w-100 mb-1"),
        html.Div(id="action-status", className="small mt-1"),

        dcc.Download(id="download-npc"),
    ]),
], style={"height": "100vh", "overflowY": "auto"})

right_panel = dbc.Card([
    dbc.CardBody([
        # Map
        html.Div([
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
        ], className="mb-3"),

        # Profile plot
        dcc.Loading(
            dcc.Graph(
                id="profile-plot",
                config={"displayModeBar": True, "scrollZoom": True,
                        "modeBarButtonsToAdd": ["select2d", "lasso2d"]},
                style={"height": "700px"},
            ),
        ),

        # Status bar
        html.Div(id="status-bar",
                 className="text-info small mt-1",
                 style={"minHeight": "20px"}),
    ]),
])

app.layout = dbc.Container([
    stores,
    dbc.Row([
        dbc.Col(left_panel,  width=3, style={"padding": "0"}),
        dbc.Col(right_panel, width=9, style={"padding": "0 0 0 8px"}),
    ]),
], fluid=True, style={"padding": "8px"})


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

    # Save uploads to temp files
    tmp_paths = []
    for content, fname in zip(contents_list, filenames):
        _, b64 = content.split(",", 1)
        raw = base64.b64decode(b64)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".rsk")
        tmp.write(raw)
        tmp.close()
        tmp_paths.append(tmp.name)

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


# ── Navigation
@app.callback(
    Output("store-current-index", "data"),
    Output("store-excluded",      "data"),
    Input("btn-prev",  "n_clicks"),
    Input("btn-next",  "n_clicks"),
    Input("btn-clear-excl", "n_clicks"),
    State("store-current-index",  "data"),
    State("store-station-matches","data"),
    State("store-excluded",       "data"),
    prevent_initial_call=True,
)
def navigate(n_prev, n_next, n_clear, current_idx, station_matches, excluded):
    triggered = ctx.triggered_id
    keys = list(station_matches.keys()) if station_matches else []
    n = len(keys)
    if triggered == "btn-prev":
        new_idx = max(0, current_idx - 1)
        return new_idx, []   # clear exclusions on station change
    if triggered == "btn-next":
        new_idx = min(n - 1, current_idx + 1)
        return new_idx, []
    if triggered == "btn-clear-excl":
        return current_idx, []
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


# ── Compute NPC whenever span or exclusions change
@app.callback(
    Output("store-npc",          "data"),
    Output("store-npc-meta",     "data"),
    Output("store-span-indices", "data"),
    Input("profile-plot",       "relayoutData"),
    Input("store-excluded",     "data"),
    Input("checklist-params",   "value"),
    State("store-current-index",  "data"),
    State("store-station-matches","data"),
    State("store-rsk-df",         "data"),
    State("store-rsk-meta",       "data"),
    State("store-span-indices",   "data"),
    State("store-cruise-times",   "data"),
    State("input-cruise-number",  "value"),
    State("input-vessel-name",    "value"),
    State("input-mission-number", "value"),
    State("input-platform",       "value"),
    prevent_initial_call=True,
)
def compute_npc(relayout_data, excluded, param_vals,
                current_idx, station_matches, rsk_df_json, rsk_meta,
                span_indices, cruise_times,
                cruise_number, vessel_name, mission_number, platform):
    if not station_matches or not rsk_df_json:
        return {}, {}, []

    keys        = list(station_matches.keys())
    station_key = keys[current_idx]
    data        = station_matches[station_key]
    df_indices  = data["df_rsk_indices"]

    df_all     = pd.read_json(StringIO(rsk_df_json), orient="split")
    df_profile = df_all.loc[df_indices].copy().reset_index(drop=True)

    # Detect downcast for initial span
    ix_down = detect_downcast(df_profile)

    # Update span from relayout (time axis selection on row=3 subplot)
    new_span = span_indices or []
    if relayout_data:
        x0 = relayout_data.get("xaxis9.range[0]") or relayout_data.get("selections[0].x0")
        x1 = relayout_data.get("xaxis9.range[1]") or relayout_data.get("selections[0].x1")
        if x0 and x1:
            ts = pd.to_datetime(df_profile["timestamp"])
            ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
            t0 = pd.Timestamp(x0, tz="UTC") if "+" not in str(x0) and "Z" not in str(x0) \
                 else pd.Timestamp(x0).tz_convert("UTC")
            t1 = pd.Timestamp(x1, tz="UTC") if "+" not in str(x1) and "Z" not in str(x1) \
                 else pd.Timestamp(x1).tz_convert("UTC")
            mask = (ts >= t0) & (ts <= t1)
            new_span = df_profile.index[mask].tolist()

    if not new_span and ix_down.any():
        new_span = df_profile.index[ix_down].tolist()

    if not new_span:
        return {}, {}, []

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

    npc_json  = df_npc.to_json(orient="split")  if len(df_npc) else "{}"
    meta_json = json.dumps(meta)

    return npc_json, meta_json, new_span


# ── Main display: map + plots + UI state
@app.callback(
    Output("leaflet-map",       "center"),
    Output("leaflet-map",       "zoom"),
    Output("map-markers",       "children"),
    Output("profile-plot",      "figure"),
    Output("station-info-text", "children"),
    Output("nav-label",         "children"),
    Output("btn-prev",          "disabled"),
    Output("btn-next",          "disabled"),
    Output("excl-count-label",  "children"),
    Output("status-bar",        "children"),
    Input("store-station-matches", "data"),
    Input("store-current-index",   "data"),
    Input("store-excluded",        "data"),
    Input("store-npc",             "data"),
    State("store-rsk-df",          "data"),
    State("store-span-indices",    "data"),
)
def update_display(station_matches, current_idx, excluded,
                   npc_json, rsk_df_json, span_indices):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="Upload RSK files to begin",
                          showarrow=False, font=dict(size=18))],
        height=700,
    )

    if not station_matches:
        return ([60, 5], 5, [], empty_fig,
                "No data loaded", "─", True, True, "", "")

    keys  = list(station_matches.keys())
    n     = len(keys)
    key   = keys[current_idx]
    data  = station_matches[key]
    si    = data["station_info"]

    # Map
    center, zoom = map_center_zoom(station_matches)
    markers = build_map_markers(station_matches, current_idx)

    # Profile
    df_npc = pd.DataFrame()
    if npc_json and npc_json != "{}":
        try:
            df_npc = pd.read_json(StringIO(npc_json), orient="split")
        except Exception:
            pass

    fig = empty_fig
    if rsk_df_json:
        try:
            df_all     = pd.read_json(StringIO(rsk_df_json), orient="split")
            df_indices = data["df_rsk_indices"]
            df_profile = df_all.loc[df_indices].copy().reset_index(drop=True)
            ix_down    = detect_downcast(df_profile)
            fig = build_profile_figure(df_profile, ix_down, df_npc, set(excluded or []))
            fig.update_layout(title_text=f"{key}")
        except Exception as e:
            pass

    # Station info text
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

    nav_label    = f"Profile {current_idx+1} / {n}"
    excl_count   = f"Excluded: {len(excluded or [])} points"
    status_msg   = f"Station {current_idx+1}/{n} · {len(df_npc)} depth bins computed"

    return (
        center, zoom, markers, fig,
        info, nav_label,
        current_idx <= 0, current_idx >= n - 1,
        excl_count, status_msg,
    )


# ── Download NPC
@app.callback(
    Output("download-npc",  "data"),
    Output("action-status", "children", allow_duplicate=True),
    Input("btn-download-npc", "n_clicks"),
    State("store-npc",       "data"),
    State("store-npc-meta",  "data"),
    State("store-current-index",  "data"),
    State("store-station-matches","data"),
    prevent_initial_call=True,
)
def download_npc(n_clicks, npc_json, meta_json, current_idx, station_matches):
    if not npc_json or npc_json == "{}":
        return no_update, "No NPC data available – select a span first."
    try:
        df_npc = pd.read_json(StringIO(npc_json), orient="split")
        meta   = json.loads(meta_json)
        keys   = list(station_matches.keys())
        key    = keys[current_idx]
        fname  = f"{key}_binned.npc"
        content = npc_to_string(meta, df_npc)
        return (dict(content=content, filename=fname, type="text/plain"),
                f"Downloaded {fname}")
    except Exception as e:
        return no_update, f"Download error: {e}"


# ── Upload to S3
@app.callback(
    Output("action-status", "children"),
    Input("btn-upload-s3",  "n_clicks"),
    State("store-npc",      "data"),
    State("store-npc-meta", "data"),
    prevent_initial_call=True,
)
def upload_to_s3(n_clicks, npc_json, meta_json):
    if not BOTO3_AVAILABLE:
        return "boto3 not installed – cannot upload."
    if not npc_json or npc_json == "{}":
        return "No NPC data available – select a span first."
    try:
        df_npc = pd.read_json(StringIO(npc_json), orient="split")
        meta   = json.loads(meta_json)

        os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"]  = "when_required"
        os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"]  = "when_required"

        s3 = boto3.resource(
            service_name="s3",
            endpoint_url="https://s3.hi.no",
            aws_access_key_id="6lpqTL2pz42cRefC1R4c",
            aws_secret_access_key="W0eUlbebUidLGiKXM0iQdS8WL0slhMDdirZ6kICj",
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".npc") as f:
            tmp_path = f.name
        npc_write(meta, df_npc, tmp_path)

        dest = (f"physchem/incoming/regular_stations/test//"
                f"{os.path.basename(tmp_path)}")
        with open(tmp_path, "rb") as fh:
            s3.Bucket("transient-data").put_object(Key=dest, Body=fh)
        os.unlink(tmp_path)
        return f"Uploaded successfully → {dest}"
    except Exception as e:
        return f"Upload failed: {e}"


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
