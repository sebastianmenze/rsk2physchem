# CTD Profile Browser

A web application for processing, quality-controlling, and exporting CTD (Conductivity-Temperature-Depth) cast data from RBR instruments. Uploaded RSK files are automatically matched to cruise and station metadata, interactively visualised, and exported as standardised NPC files for submission to the Norwegian PhysChem database.

---

## Features

- Upload one or more RBR RSK binary files (multi-cast cruises supported)
- Automatic station matching against Toktlogger cruise/activity APIs
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
- Displays all matched CTD stations as markers on the map
- Loads the first station automatically

---

### 3. Browse stations

Use the **← Prev** and **Next →** buttons to step through the matched CTD stations, or click any marker on the map and press **Select profile** in the popup.

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

### 9. Download or upload the NPC file

Once you are satisfied with the span selection, exclusions, and metadata:

- **Download NPC File** — saves a `.npc` text file to your computer
- **Upload to PhysChem (S3)** — sends the file directly to the configured S3 bucket for ingestion into the PhysChem database

Both actions recompute the NPC dataset from scratch so that any last-minute edits to the text fields are captured.

The upload button is disabled if the profile is already present in PhysChem or if S3 credentials are not configured.

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
