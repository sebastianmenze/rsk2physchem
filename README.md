# CTD Profile Browser

A web application for processing, quality-controlling, and exporting CTD (Conductivity-Temperature-Depth) cast data from RBR instruments. Uploaded RSK files are automatically matched to cruise and station metadata, interactively visualised, and exported as standardised NPC files for submission to the Norwegian PhysChem database.

---

## Features

- Upload one or more RBR RSK binary files (multi-cast cruises supported)
- Automatic station matching against Toktlogger cruise/activity APIs
- Overview of all profiles as images (downcast + profiles + NPC line) for quick QC, stacked vertically, with a station map
- Per-profile Include tick box to leave profiles out of the NPC export and PhysChem upload
- PhysChem status per profile (already in PhysChem / new / uploaded)
- Download all NPC files as one zip, or upload all new profiles to PhysChem at once
- Interactive 4-panel profile plot (temperature, salinity, dissolved O₂, chlorophyll)
- Depth-vs-time timeseries with drag-to-select span
- Time-range slider for downcast selection (auto-detected)
- Point exclusion via lasso/box select directly on the profile plot
- Download or upload processed NPC files to PhysChem (S3)
- Login-protected access

---

## Installation

### Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `PASSWORD` | Login password for the app |
| `URL_PREFIX` | URL sub-path if served behind a proxy (e.g. `/rsk2physchem`) |
| `TOKTLOGGER_CRUISES_URL` | Toktlogger API endpoint for cruise list |
| `TOKTLOGGER_ACTIVITIES_URL` | Toktlogger API endpoint for CTD activities |
| `S3_ENDPOINT_URL` | S3-compatible object storage URL |
| `S3_ACCESS_KEY_ID` | S3 access key |
| `S3_SECRET_ACCESS_KEY` | S3 secret key |
| `S3_BUCKET` | Target bucket name |
| `S3_DEST_PREFIX` | Key prefix for uploaded NPC files |
| `PHYSCHEM_MATCH_MINUTES` | Optional (default 10). Max. start-time difference for a profile to match a PhysChem operation |
| `PHYSCHEM_MATCH_KM` | Optional (default 2). Max. distance between start positions for a match |
| `PHYSCHEM_OPERATION_URL` | Optional. Link shown next to **In PhysChem**; `{mission_id}` and `{operation_id}` are filled in. Defaults to the PhysChem editor (`https://physchem-editor.hi.no/mission/{mission_id}/operation/{operation_id}/instrument`) |

### Run locally

```bash
python app.py
```

Open [http://localhost:8050](http://localhost:8050) in your browser.

### Run with Docker

```bash
docker-compose up
```

The app is served on port 8050.

---

## Tutorial

### 1. Log in

On first load a password dialog appears. Enter the password set in `PASSWORD` in your `.env` file.

---

### 2. Upload RSK files

Drag and drop one or more `.rsk` files onto the **Upload RSK Files** area, or click to browse. Multiple files from the same cruise can be uploaded together — they will be concatenated and treated as a single dataset.

While files are being parsed a fullscreen loading spinner is shown.

After processing, the app:
- Queries the Toktlogger API to find cruise and station metadata matching the data timestamps
- Populates the **Cruise Parameters** fields (cruise number, vessel, mission, platform)
- Skips CTD stations that contain no RSK data points
- Numbers operations (`operation.operationNumber`) by each station's position among all Toktlogger CTD activities of the whole cruise, sorted by start time, so numbers stay the same when a cruise's RSK files are uploaded in several stages. Mission start/stop dates also come from the whole cruise. If the cruise's activity list can't be fetched, stations are numbered in upload order and the status line warns about it.
- Auto-detects the downcast of every profile, computes its NPC bins, and draws an overview image
- Checks which profiles are already in PhysChem

---

### 3. Overview of all profiles

At the top of the overview a map shows every station as a numbered marker, coloured by PhysChem status (green: in PhysChem, orange: new, light blue: uploaded, grey: unknown). Click a marker for the station's date, activity, position, number of points and status, and an **Edit profile** button.

Below the map, the overview lists one image per profile (scroll vertically): depth vs time with the selected downcast shaded, and temperature, salinity, O₂ and chlorophyll with the NPC bin averages as a red line. Excluded points are red ×.

Each profile has an **Include** tick box (all ticked by default). Unticked profiles are greyed out, faded on the map, and left out of **Download all NPC files** and **Upload new profiles**.

Each image has badges:

| Badge | Meaning |
|---|---|
| **In PhysChem** | A CTD operation of this mission in PhysChem starts within ±10 min of the profile and (if both have positions) within 2 km; the **open in PhysChem ↗** link opens it, and hovering shows the time difference and distance |
| **Possible match in PhysChem – check** | A CTD operation starts at the same time but its position is further away than 2 km. Open the link to check; these profiles are not uploaded automatically |
| **New** | Not yet in PhysChem — will be uploaded by **Upload new profiles** |
| **Uploaded – awaiting PhysChem** | Uploaded to the S3 inbox in this session, not yet listed by PhysChem. After PhysChem has ingested it, **Check PhysChem status** shows **In PhysChem** and **Uploaded** together |
| **Edited** | Span or exclusions were changed and saved by hand |
| **No NPC data** | The span produced no depth bins — check this profile |
| **PhysChem status unknown** | PhysChem could not be queried (check mission # / platform #) |

Click **Edit profile** (on the profile or in its map popup), or **double-click** the image, to open that profile in the interactive view (steps 4–6). There, press **Save** to keep your changes (the overview image is redrawn) and **← Back to overview** to return. Changes that are not saved are discarded when you go back or move to another profile; the toolbar shows **● Unsaved changes** until you save. **Reset to auto downcast** restores the automatic span and clears exclusions.

In the interactive view, use **← Prev** / **Next →**, **Go to #**, or a map marker's **Select profile** to move between profiles.

The **Station Info** panel shows the station name, activity number, start/end times, coordinates, and the total number of data points. Stations where the CTD trigger time was automatically corrected are shown as orange markers with a warning note.

---

### 4. Inspect and adjust the time span

The **Depth vs Time** plot (top right) shows the full cast as a grey line. The highlighted blue region is the currently selected span — the portion of the data that will be binned and exported.

The **time-range slider** below the plot lets you adjust the span manually:
- Drag the left handle to change the start of the span
- Drag the right handle to change the end
- Time labels at 0 %, 25 %, 50 %, 75 %, and 100 % show the UTC time at those positions

Alternatively, **drag horizontally** directly on the Depth vs Time plot to select a time range — the slider will update to match.

On station load the span is initialised to the automatically detected downcast (the continuous descent through the water column).

---

### 5. Review the profile plots

The bottom panel shows four side-by-side plots, all sharing the same depth axis (metres, increasing downward):

| Panel | Parameter | Unit |
|---|---|---|
| Temperature | Water temperature | °C |
| Salinity | Practical salinity | PSU |
| Dissolved O₂ | Oxygen concentration | µmol/l |
| Chlorophyll | Fluorescence proxy | µg/l |

Points are coloured by their position in the cast using the Viridis scale. The **red line** shows the 1-metre depth-bin averages that will be written to the NPC file.

O₂ and chlorophyll panels are only shown if those sensors were active and data are present.

---

### 6. Exclude bad points (QC)

To mark data points as bad:

1. In the profile plot toolbar, choose **Box select** or **Lasso select**
2. Draw a selection around the points you want to exclude
3. Selected points turn into blue × markers and are immediately removed from the bin averages and the red NPC line

Excluded points remain visible so you can see what was removed. To start over, click **Clear Exclusions** in the left panel.

The exclusion count is shown below the Clear button and is also reported in the status bar.

---

### 7. Set export parameters

Use the **Export Parameters** checkboxes to include or exclude dissolved O₂ and chlorophyll from the exported NPC file. Unchecking a parameter removes it from the binned output even if sensor data exist.

---

### 8. Edit cruise metadata

The **Cruise Parameters** fields are auto-filled from the Toktlogger API but can be edited freely:

- **Cruise #** — cruise identifier
- **Vessel** — ship name
- **Mission #** — PhysChem mission number
- **Platform #** — instrument platform identifier

Any edits are included in the NPC file the next time you download or upload — the data are always recomputed fresh at that point.

---

#### How the PhysChem check works

1. The mission is looked up by **Platform #** and **Mission #** (`/mission/list`). If several years use the same mission number, the one whose start year matches the cruise is used.
2. The mission's operations are fetched (`/mission/{id}/operation/list`). For each profile, the CTD operation with the closest `timeStart` within ±`PHYSCHEM_MATCH_MINUTES` is taken, comparing real timestamps (time zones and formats don't matter).
3. If both the profile (Toktlogger start position) and the operation have a start position, the distance must be within `PHYSCHEM_MATCH_KM`; otherwise the profile is flagged **Possible match – check**.

If PhysChem can't be reached, profiles show **PhysChem status unknown** and are not uploaded.

### 9. Download or upload all NPC files

The **All Profiles** section in the left panel works on every profile at once, using each profile's saved span and exclusions:

- **Download all NPC files (.zip)** — one `.npc` file per included profile, named `cruisenumber_YYYYMMDD_HHMMSS.npc`
- **Upload new profiles to PhysChem** — after a confirmation, sends only the included profiles marked **New** to the configured S3 bucket. Unticked profiles, profiles already in PhysChem, uploaded earlier in the session, or with unknown status are skipped.
- **Check PhysChem status** — re-queries PhysChem, e.g. after correcting the mission or platform number

Both actions recompute the NPC data from scratch, so edits to the cruise parameters and export parameters are always included. The upload button is disabled until all cruise parameters are filled in and at least one profile is new.

---

## File formats

### RSK (input)

Binary files produced by RBR CTD loggers (e.g. RBRconcerto³). Read via the `pyrsktools` library. The app derives sea pressure, practical salinity, depth, and vertical velocity from the raw sensor channels.

### NPC (output)

Tab-separated text format used for submission to the Norwegian PhysChem database. The file has two sections:

- **Header** (`#` lines) — cruise metadata, parameter definitions, units, and data collection details
- **Data** (`%` lines) — one row per 1-metre depth bin, containing mean value, standard deviation, and sample count for each parameter

---

## Architecture overview

```
app.py          Main Dash application — layout, callbacks, and data processing
requirements.txt Python dependencies
docker-compose.yml Docker service definition
.env.example    Template for environment configuration
```

All application logic lives in `app.py`. Key internal functions:

| Function | Purpose |
|---|---|
| `process_rsk_file()` | Parse a single RSK binary file using pyrsktools |
| `get_station_indices_for_ctd()` | Match RSK timestamps to Toktlogger activities |
| `detect_downcast()` | Identify the downcast portion of a cast automatically |
| `calculate_df_npc()` | Bin profile data into 1-metre depth intervals |
| `npc_write()` / `npc_to_string()` | Serialise binned data to NPC text format |
| `build_profile_figure()` | Render the 4-panel profile plot |
| `build_timeseries_figure()` | Render the depth-vs-time cast overview |
