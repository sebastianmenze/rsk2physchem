
"""
Created on Tue Nov  4 13:13:39 2025

@author: a5278

CTD Profile Browser - Full Auto Version with Lasso Selection
A PyQt5 application for viewing and analyzing CTD oceanographic data.
Supports RSK file processing and station matching with interactive mapping.

Fully Automated Features:
- Auto-extracts date range from RSK files (no manual date entry)
- Auto-fetches and matches cruise parameters from API
- Auto-processes data immediately after file selection (no button needed!)
- Auto-corrects profile start times when starting below surface
- Interactive matplotlib maps with coastlines (cartopy) - with zoom/pan
- Lasso selection tool for data quality control
- Robust timezone handling (all timestamps normalized to UTC)
- Auto-detect O2 and Chl parameters with selective export

Workflow:
1. Select RSK files → Everything happens automatically!
2. Browse profiles with zoomable map
3. Use lasso tool to exclude bad data
4. Select which parameters to export (O2/Chl auto-detected)
5. Export/upload to PhysChem
"""

# [Previous imports remain the same...]
# Standard library imports
import sys
import os
import tempfile
import glob
from datetime import datetime

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: cartopy not available. Install with: pip install cartopy")

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QSplitter, QFrame, QGroupBox, 
    QFileDialog, QProgressBar, QMessageBox, QLineEdit, QCheckBox
)
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QFont
QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector, LassoSelector
from matplotlib.path import Path
import re
from io import StringIO
import uuid

# AWS and API imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
import requests

# Optional dependency: pyrsktools for RSK file processing
try:
    import pyrsktools
    PYRSK_AVAILABLE = True
except ImportError:
    PYRSK_AVAILABLE = False
    print("Warning: pyrsktools not available. Install with: pip install pyrsktools")


# [All the helper functions remain the same - get_cruises_from_api, match_cruise_by_dates, etc...]
def get_cruises_from_api(base_url: str = "http://toktlogger-hansb.hi.no/api/cruises/all") -> pd.DataFrame:
    """
    Fetch all cruises from the toktlogger API.
    
    Parameters:
    -----------
    base_url : str, optional
        Base URL for the API endpoint
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all cruise data with timezone-aware timestamps
    """
    params = {'format': 'json'}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        
        if len(df) > 0 and 'startTime' in df.columns:
            # Parse timestamps and ensure they're timezone-aware (UTC)
            df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
            df['endTime'] = pd.to_datetime(df['endTime'], utc=True)
        
        return df
    except Exception as e:
        print(f"Error fetching cruises from API: {e}")
        return pd.DataFrame()


def match_cruise_by_dates(start_date, end_date, df_cruises):
    """
    Match a cruise based on date overlap.
    
    Parameters:
    -----------
    start_date : datetime
        Start date from RSK files
    end_date : datetime
        End date from RSK files
    df_cruises : pd.DataFrame
        DataFrame with cruise data
        
    Returns:
    --------
    dict or None
        Matched cruise data or None if no match found
    """
    if len(df_cruises) == 0:
        return None
    
    # Convert to pandas Timestamp and ensure timezone-aware (UTC)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Localize to UTC if timezone-naive
    if start_date.tz is None:
        start_date = start_date.tz_localize('UTC')
    else:
        start_date = start_date.tz_convert('UTC')
    
    if end_date.tz is None:
        end_date = end_date.tz_localize('UTC')
    else:
        end_date = end_date.tz_convert('UTC')
    
    # Find cruises that overlap with our date range
    matches = []
    for idx, cruise in df_cruises.iterrows():
        cruise_start = pd.Timestamp(cruise['startTime'])
        cruise_end = pd.Timestamp(cruise['endTime'])
        
        # Ensure cruise times are timezone-aware (UTC)
        if cruise_start.tz is None:
            cruise_start = cruise_start.tz_localize('UTC')
        else:
            cruise_start = cruise_start.tz_convert('UTC')
        
        if cruise_end.tz is None:
            cruise_end = cruise_end.tz_localize('UTC')
        else:
            cruise_end = cruise_end.tz_convert('UTC')
        
        # Check for overlap: our dates overlap with cruise dates
        if (start_date <= cruise_end) and (end_date >= cruise_start):
            # Calculate overlap duration
            overlap_start = max(start_date, cruise_start)
            overlap_end = min(end_date, cruise_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            
            matches.append({
                'cruise': cruise,
                'overlap_duration': overlap_duration
            })
    
    if not matches:
        return None
    
    # Return the cruise with the longest overlap
    best_match = max(matches, key=lambda x: x['overlap_duration'])
    return best_match['cruise'].to_dict()


def get_activities_from_api(
    after: str,
    before: str,
    base_url: str = "http://toktlogger-hansb.hi.no/api/activities/inPeriod"
) -> pd.DataFrame:
    """
    Fetch activity data from the toktlogger API and return as a pandas DataFrame.
    
    Parameters:
    -----------
    after : str
        Start date in ISO format (e.g., '2025-10-04T16:21:56.000Z')
    before : str
        End date in ISO format (e.g., '2025-11-04T16:21:35.880Z')
    base_url : str, optional
        Base URL for the API endpoint
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the activity data with flattened position coordinates
    """
    
    # Prepare request parameters
    params = {
        'after': after,
        'before': before,
        'format': 'json'
    }
    
    # Make API request
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise an error for bad status codes
    
    # Parse JSON response
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        return df
    
    # Flatten the position data
    if 'startPosition' in df.columns:
        # Extract coordinates from startPosition
        df['startLon'] = df['startPosition'].apply(
            lambda x: x['coordinates'][0] if x and 'coordinates' in x else None
        )
        df['startLat'] = df['startPosition'].apply(
            lambda x: x['coordinates'][1] if x and 'coordinates' in x else None
        )
        
        # Extract coordinates from endPosition
        df['endLon'] = df['endPosition'].apply(
            lambda x: x['coordinates'][0] if x and 'coordinates' in x else None
        )
        df['endLat'] = df['endPosition'].apply(
            lambda x: x['coordinates'][1] if x and 'coordinates' in x else None
        )
    
    # Convert datetime strings to datetime objects with timezone (UTC)
    if 'startTime' in df.columns:
        df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
    if 'endTime' in df.columns:
        df['endTime'] = pd.to_datetime(df['endTime'], utc=True)
    
    # Rename columns to match expected format if needed
    df = df.rename(columns={
        'name': 'name',
        'activityNumber': 'activityNumber'
    })
    
    return df


class NPCFileHandler:
    """Handler for reading and writing NPC (NetCDF-style) files."""
    
    @staticmethod
    def read(filename):
        """
        Read metadata and data from an NPC file.
        
        Args:
            filename: Path to the NPC file
            
        Returns:
            tuple: (metadata dict, data DataFrame)
        """
        with open(filename, 'r') as file:
            content = file.read()
        
        meta = {}
        m1 = re.search('# Metadata:', content)
        m2 = re.search('% Readings:', content)
        
        meta_text = content[m1.span()[1]:m2.span()[0]].strip()
        
        for line in meta_text.splitlines():  
            sep = re.search(':', line)
            fieldname = line[:sep.span()[0]]
            value = line[sep.span()[1]:].strip()
            meta[fieldname] = value
        
        tbl = StringIO(content[m2.span()[1]:])
        df = pd.read_csv(tbl, sep="\t")
        df = df.dropna(axis=1, how='all')
        
        return meta, df
    
    @staticmethod
    def write(meta, df, filename):
        """
        Write metadata and data to an NPC file.
        
        Args:
            meta: Dictionary of metadata
            df: DataFrame with data
            filename: Output file path
        """
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            f.write('# Metadata:\n')
            
            for key, value in meta.items():
                f.write(f'{key}:\t{str(value)}\n')
            
            f.write('% Readings:\n')
            f.write(df.to_csv(sep='\t', index=False, lineterminator='\n'))


def check_if_operation_in_physchem(npcfile):
    """
    Check if an operation already exists in PhysChem database.
    
    Args:
        npcfile: Path to NPC file
        
    Returns:
        bool: True if operation exists in database
    """
    meta, df = NPCFileHandler.read(npcfile)
    
    response = requests.get(
        f"https://physchem-api.hi.no/mission/list?"
        f"platform={meta['mission.platform']}"
    )
    js = json.loads(response.content)
    df_missions = pd.DataFrame.from_dict(js)
    
    if len(df_missions) == 0:
        return False
    
    mission_match = df_missions['missionNumber'] == int(meta['mission.missionNumber'])
    if mission_match.sum() == 0:
        return False
    
    mission_id = df_missions.loc[mission_match, 'id'].values[0]
    
    response = requests.get(
        f'https://physchem-api.hi.no/mission/{mission_id}/'
        f'operation/list?extend=false&instrumentTypeList=false'
    )
    js = json.loads(response.content)
    df_operations = pd.DataFrame.from_dict(js)
    
    return np.isin(df_operations['timeStart'], meta['operation.timeStart']).sum() > 0


class npc():
    def read(filename):
        file = open(filename,'r')
        content = file.read()
        meta={}
        
        m1 = re.search('# Metadata:', content)
        m2 = re.search('% Readings:', content)
        
        meta_text = content[ m1.span()[1] : m2.span()[0] ].strip()
        
        for line in meta_text.splitlines():  
            sep = re.search(':', line )
            fieldname =  line[:  sep.span()[0] ]
            value =  line[ sep.span()[1] :]
            value=    value.strip()
            meta[ fieldname ]   = value
            
        tblmatchstr = re.search('% Readings:', content)
        tbl = StringIO( content[ tblmatchstr.span()[1]: ] )
        df = pd.read_csv(tbl, sep="\t")
        df = df.dropna(axis=1, how='all')
        file.close()
        
        return meta , df
    
    def write(meta,df,filename):
        with open(filename, "w", encoding="utf-8", newline="\n")  as f:
            f.write('# Metadata:\n')
            
            for x, y in meta.items():
                f.write(x + ':\t'+ str(y))
                f.write('\n')
            
            f.write('% Readings:\n')
            f.write( df.to_csv(sep='\t',index=False,lineterminator='\n') )


# [DataProcessingThread class remains the same...]
class DataProcessingThread(QThread):
    """
    Background thread for processing RSK files and matching with station data from API.
    Prevents UI freezing during heavy data processing operations.
    Now includes automatic start time correction.
    """
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    data_ready = pyqtSignal(object, object)  # station_matches, df_rsk_all
    error_occurred = pyqtSignal(str)
    
    def __init__(self, rsk_files, start_date, end_date):
        super().__init__()
        self.rsk_files = rsk_files
        self.start_date = start_date
        self.end_date = end_date
    
    def run(self):
        """Main processing routine executed in background thread"""
        try:
            # Load station data from API
            self.status_update.emit("Loading station data from API...")
            
            # Format dates for API call
            after_str = self.start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            before_str = self.end_date.strftime('%Y-%m-%dT%H:%M:%S.999Z')
            
            # Fetch data from API
            df_tk = get_activities_from_api(after_str, before_str)
            
            if len(df_tk) == 0:
                self.error_occurred.emit("No activities found in the specified date range")
                return
            
            self.status_update.emit(f"Loaded {len(df_tk)} activities from API")
            
            # Process RSK files
            self.status_update.emit("Processing RSK files...")
            df_rsk_all = pd.DataFrame()
            
            for i, filepath in enumerate(self.rsk_files):
                try:
                    self.status_update.emit(f"Processing {os.path.basename(filepath)}...")
                    
                    if not PYRSK_AVAILABLE:
                        self.error_occurred.emit("pyrsktools is required to process RSK files")
                        return
                    
                    # Read and process RSK file using pyrsktools
                    rsk = pyrsktools.RSK(filepath)
                    rsk.open()
                    rsk.readdata()
                    rsk.deriveseapressure()
                    rsk.derivesalinity()
                    rsk.derivedepth()
                    rsk.derivevelocity()
                    
                    df_rsk = pd.DataFrame(rsk.data)
                    
                    rsk_meta={}
                    rsk_meta['instrument.instrumentSerialNumber']=rsk.instrument.serialID                
                    rsk_meta['instrument.instrumentModel']=rsk.instrument.model
                    
                    # Correct column name if needed (chlorophyll_a -> chlorophyll)
                    if "chlorophyll_a" in df_rsk.columns:
                        df_rsk = df_rsk.rename(columns={"chlorophyll_a": "chlorophyll"})
                    
                    df_rsk_all = pd.concat([df_rsk_all, df_rsk])
                    
                    # Update progress (first 50% for RSK processing)
                    progress = int((i + 1) / len(self.rsk_files) * 50)
                    self.progress_update.emit(progress)
                    
                except Exception as e:
                    self.error_occurred.emit(f"Error processing {filepath}: {str(e)}")
                    return
            
            # Sort and reset index for combined data
            df_rsk_all = df_rsk_all.sort_values('timestamp')
            df_rsk_all = df_rsk_all.reset_index(drop=True)
            
            # Match CTD data to stations with automatic time correction
            self.status_update.emit("Matching CTD data to stations (with time correction)...")
            station_matches = self.get_station_indices_for_ctd(df_rsk_all, df_tk)
            
            self.progress_update.emit(100)
            self.status_update.emit("Data processing complete!")
            
            # Emit the results
            self.data_ready.emit(station_matches, df_rsk_all)
            
        except Exception as e:
            self.error_occurred.emit(f"Data processing error: {str(e)}")
    
    def get_station_indices_for_ctd(self, df_rsk, df_tk, time_tolerance_minutes=1, 
                                     max_time_correction_minutes=15, surface_depth_threshold=2.0):
        """
        Match CTD data points to stations based on time windows.
        Automatically corrects start times if profile starts below surface.
        
        Parameters:
        - df_rsk: DataFrame with CTD data including 'timestamp' column
        - df_tk: DataFrame with station data including 'startTime' and 'endTime'
        - time_tolerance_minutes: Extra time buffer around station times
        - max_time_correction_minutes: Maximum time to look back for correcting start time
        - surface_depth_threshold: Depth threshold (m) to consider as "at surface"
        
        Returns:
        - Dictionary with station info as keys and df_rsk indices as values
        """
        station_indices = {}
        time_buffer = pd.Timedelta(minutes=time_tolerance_minutes)
        correction_window = pd.Timedelta(minutes=max_time_correction_minutes)
        
        # Ensure RSK timestamps are timezone-aware (UTC)
        df_rsk_timestamps = pd.to_datetime(df_rsk['timestamp'])
        # Note: df_rsk_timestamps is a Series, so we use .dt accessor
        if df_rsk_timestamps.dt.tz is None:
            df_rsk_timestamps = df_rsk_timestamps.dt.tz_localize('UTC')
        else:
            df_rsk_timestamps = df_rsk_timestamps.dt.tz_convert('UTC')
        
        # Filter for CTD stations only
        ctd_stations = df_tk[df_tk['activityMainGroupName'] == 'CTD'].copy()
        
        for idx, station in ctd_stations.iterrows():
            # Ensure station times are timezone-aware (UTC)
            station_start_time = pd.to_datetime(station['startTime'])
            station_end_time = pd.to_datetime(station['endTime'])
            
            if station_start_time.tz is None:
                station_start_time = station_start_time.tz_localize('UTC')
            else:
                station_start_time = station_start_time.tz_convert('UTC')
            
            if station_end_time.tz is None:
                station_end_time = station_end_time.tz_localize('UTC')
            else:
                station_end_time = station_end_time.tz_convert('UTC')
            
            # Initial time window
            station_start = station_start_time - time_buffer
            station_end = station_end_time + time_buffer
            
            # Find CTD data points within this time window
            time_mask = (df_rsk_timestamps >= station_start) & (df_rsk_timestamps <= station_end)
            matching_indices = df_rsk.index[time_mask].tolist()
            
            corrected_start_time = station_start_time
            time_was_corrected = False
            correction_info = ""
            
            # Check if we need to correct the start time
            if len(matching_indices) > 0 and 'depth' in df_rsk.columns:
                # Get the first matched data point
                first_idx = matching_indices[0]
                first_depth = df_rsk.loc[first_idx, 'depth']
                
                # If first point is significantly below surface, look for earlier data
                if first_depth > surface_depth_threshold:
                    self.status_update.emit(
                        f"Station {station['name']}: First point at {first_depth:.1f}m depth. "
                        f"Searching for actual profile start..."
                    )
                    
                    # Look backwards in time to find where profile actually starts
                    lookback_start = station_start_time - correction_window
                    lookback_mask = (df_rsk_timestamps >= lookback_start) & \
                                   (df_rsk_timestamps < station_start)
                    
                    lookback_indices = df_rsk.index[lookback_mask].tolist()
                    
                    if len(lookback_indices) > 0:
                        # Find the earliest point near the surface
                        lookback_data = df_rsk.loc[lookback_indices]
                        near_surface = lookback_data[lookback_data['depth'] <= surface_depth_threshold]
                        
                        if len(near_surface) > 0:
                            # Found data near surface before the CSV start time
                            actual_start_idx = near_surface.index[0]
                            actual_start_time = df_rsk.loc[actual_start_idx, 'timestamp']
                            
                            # Ensure actual_start_time is timezone-aware for comparison
                            actual_start_time_tz = pd.to_datetime(actual_start_time)
                            if actual_start_time_tz.tz is None:
                                actual_start_time_tz = actual_start_time_tz.tz_localize('UTC')
                            else:
                                actual_start_time_tz = actual_start_time_tz.tz_convert('UTC')
                            
                            # Calculate time difference
                            time_diff = (station_start_time - actual_start_time_tz).total_seconds() / 60
                            
                            self.status_update.emit(
                                f"✓ Station {station['name']}: Corrected start time by {time_diff:.1f} minutes. "
                                f"New start depth: {df_rsk.loc[actual_start_idx, 'depth']:.1f}m"
                            )
                            
                            # Update the corrected start time (already timezone-aware)
                            corrected_start_time = actual_start_time_tz
                            time_was_corrected = True
                            correction_info = f"Time corrected by {time_diff:.1f} min (depth: {first_depth:.1f}m → {df_rsk.loc[actual_start_idx, 'depth']:.1f}m)"
                            
                            # Rematch with corrected time
                            station_start = corrected_start_time - time_buffer
                            time_mask = (df_rsk_timestamps >= station_start) & \
                                       (df_rsk_timestamps <= station_end)
                            matching_indices = df_rsk.index[time_mask].tolist()
            
            # Create station identifier
            station_key = f"{station['name']}_{station['activityNumber']}"
            
            # Format times for storage (remove timezone info for display)
            if time_was_corrected:
                # Convert to timezone-naive for display
                corrected_start_time_naive = corrected_start_time.replace(tzinfo=None)
                start_time_str = corrected_start_time_naive.strftime('%Y-%m-%d %H:%M:%S')
                original_start_str = station_start_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # Convert station time to naive for display
                station_start_naive = station_start_time.replace(tzinfo=None)
                start_time_str = station_start_naive.strftime('%Y-%m-%d %H:%M:%S')
                original_start_str = None
            
            # Convert end time to naive for display
            station_end_naive = station_end_time.replace(tzinfo=None)
            end_time_str = station_end_naive.strftime('%Y-%m-%d %H:%M:%S')
            
            # Store results with corrected time if applicable
            station_info = {
                'name': station['name'],
                'activityNumber': station['activityNumber'],
                'startTime': start_time_str,
                'endTime': end_time_str,
                'startLat': station['startLat'],
                'startLon': station['startLon'],
                'comment': station.get('comment', ''),
                'original_startTime': original_start_str,
                'time_corrected': time_was_corrected,
                'correction_info': correction_info
            }
            
            station_indices[station_key] = {
                'df_rsk_indices': matching_indices,
                'station_info': station_info,
                'n_datapoints': len(matching_indices)
            }
        
        return station_indices


class CTDProfileBrowser(QMainWindow):
    """
    Main application window for CTD Profile Browser - Fully Automated.
    
    Key Features:
    - ONE-CLICK workflow: Select files → Everything processes automatically
    - Auto-extracts dates from RSK files
    - Auto-fetches and matches cruise from API
    - Auto-fills cruise parameters
    - Interactive zoomable map with coastlines
    - Automatic profile start time correction
    - API-based station data retrieval
    - Lasso selection tool for data quality control
    - NPC file export and upload to PhysChem
    - Auto-detect O2 and Chl with selective export
    """
    
    def __init__(self, station_matches=None, df_rsk_all=None):
        super().__init__()
        self.station_matches = station_matches or {}
        self.df_rsk_all = df_rsk_all or pd.DataFrame()
        self.station_keys = list(self.station_matches.keys()) if station_matches else []
        self.current_index = 0
        
        # Initialize file paths
        self.rsk_files = []
        
        # Initialize date range (will be extracted from RSK files)
        self.start_date = None
        self.end_date = None
        
        # Store plot axes for updating
        self.plot_axes = {}
        
        # Initialize df_npc and meta as empty
        self.df_npc = pd.DataFrame()
        self.meta = {}
        
        # Track excluded data points (indices)
        self.excluded_indices = set()
        
        self.initUI()
        
        # Display content based on data availability
        if station_matches and df_rsk_all is not None and not df_rsk_all.empty:
            self.display_current_profile()
        else:
            self.show_welcome_message()
    
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle('CTD Profile Browser (One-Click Auto + Lasso QC)')
        self.setGeometry(100, 100, 1600, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter to divide left panel and right area
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel for controls and info
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right area for plots
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Create matplotlib canvas for plots
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, right_widget)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        main_splitter.addWidget(right_widget)
        
        # Set splitter proportions (25% left, 75% right)
        main_splitter.setSizes([400, 1200])
    
    def show_welcome_message(self):
        """Display welcome message when no data is loaded"""
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.6, 'CTD Profile Browser', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=24, fontweight='bold')
        ax.text(0.5, 0.45, 'Fully Automated + Lasso QC + Parameter Selection', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16, style='italic', color='blue')
        ax.text(0.5, 0.32, '✓ Auto-process  ✓ Auto-cruise  ✓ Lasso QC  ✓ O₂/Chl selection', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=11, color='green')
        ax.text(0.5, 0.2, 'Select RSK files to begin (will process automatically)', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()
    
    def create_left_panel(self):
        """Create the left control panel with all widgets"""
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setMaximumWidth(450)
        layout = QVBoxLayout(left_panel)
        
        # File selection group
        layout.addWidget(self.create_file_selection_group())
        
        # Navigation controls
        layout.addWidget(self.create_navigation_group())
        
        # Station information
        layout.addWidget(self.create_station_info_group())
        
        layout.addWidget(self.create_cruise_params_group())

        # Folium map section
        layout.addWidget(self.create_map_group())
        
        # Parameter selection group (NEW)
        layout.addWidget(self.create_parameter_selection_group())
        
        # Export/Action buttons
        layout.addWidget(self.create_actions_group())
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return left_panel
    
    def create_file_selection_group(self):
        """Create file selection widgets"""
        file_group = QGroupBox("Data Files")
        file_layout = QVBoxLayout(file_group)
        
        # RSK files selection
        self.select_rsk_button = QPushButton('Select RSK Files')
        self.select_rsk_button.clicked.connect(self.select_rsk_files)
        file_layout.addWidget(self.select_rsk_button)
        
        self.rsk_files_label = QLabel("No RSK files selected")
        self.rsk_files_label.setWordWrap(True)
        self.rsk_files_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        file_layout.addWidget(self.rsk_files_label)
        
        # Date range display (auto-detected from RSK files)
        date_group = QGroupBox("Date Range (Auto-detected from RSK)")
        date_layout = QVBoxLayout(date_group)
        
        self.date_range_label = QLabel("Select RSK files to detect date range")
        self.date_range_label.setWordWrap(True)
        self.date_range_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        date_layout.addWidget(self.date_range_label)
        
        file_layout.addWidget(date_group)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("QLabel { color: #0066cc; font-size: 10px; }")
        file_layout.addWidget(self.status_label)
        
        # Progress bar (hidden by default, shown during processing)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        file_layout.addWidget(self.progress_bar)
        
        return file_group
    
    def create_navigation_group(self):
        """Create navigation controls"""
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)
        
        # Current profile info
        self.profile_label = QLabel("No data loaded")
        self.profile_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.profile_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.profile_label)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton('← Previous')
        self.prev_button.clicked.connect(self.previous_profile)
        self.prev_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton('Next →')
        self.next_button.clicked.connect(self.next_profile)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)
        
        nav_layout.addLayout(button_layout)
        return nav_group
    
    def create_station_info_group(self):
        """Create station information display"""
        info_group = QGroupBox("Station Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        return info_group
    
    def create_cruise_params_group(self):
        """Create cruise parameters input group (auto-filled from API)"""
        params_group = QGroupBox("Cruise Parameters (Auto-filled)")
        params_layout = QVBoxLayout(params_group)
        
         # Cruise number input
        cruise_layout = QHBoxLayout()
        cruise_label = QLabel("Cruise Number:")
        cruise_label.setMinimumWidth(80)
        self.cruise_number_input = QLineEdit("")
        cruise_layout.addWidget(cruise_label)
        cruise_layout.addWidget(self.cruise_number_input)
        params_layout.addLayout(cruise_layout)
        
        # Vessel name input
        vessel_layout = QHBoxLayout()
        vessel_label = QLabel("Vessel Name:")
        vessel_label.setMinimumWidth(80)
        self.vessel_name_input = QLineEdit("")
        vessel_layout.addWidget(vessel_label)
        vessel_layout.addWidget(self.vessel_name_input)
        params_layout.addLayout(vessel_layout)
        
        # # callsign  input
        # callsign_layout = QHBoxLayout()
        # callsign_label = QLabel("Call Signal:")
        # callsign_label.setMinimumWidth(80)
        # self.callsign_name_input = QLineEdit("")
        # callsign_layout.addWidget(callsign_label)
        # callsign_layout.addWidget(self.callsign_name_input)
        # params_layout.addLayout(callsign_layout)
        
        missionnumber_layout = QHBoxLayout()
        missionnumber_label = QLabel("Mission number:")
        missionnumber_label.setMinimumWidth(80)
        self.missionnumber_name_input = QLineEdit("")
        missionnumber_layout.addWidget(missionnumber_label)
        missionnumber_layout.addWidget(self.missionnumber_name_input)
        params_layout.addLayout(missionnumber_layout)
        
        #Cruise number input
        paltform_layout = QHBoxLayout()
        platf_label = QLabel("Platform Number:")
        platf_label.setMinimumWidth(80)
        self.platform_number_input = QLineEdit("")
        paltform_layout.addWidget(platf_label)
        paltform_layout.addWidget(self.platform_number_input)
        params_layout.addLayout(paltform_layout)
        
        return params_group
    
    def create_map_group(self):
        """Create matplotlib map section with coastlines"""
        map_group = QGroupBox("Station Map")
        map_layout = QVBoxLayout(map_group)
        
        # Create matplotlib figure for map
        self.map_figure = Figure(figsize=(5, 4))
        self.map_canvas = FigureCanvas(self.map_figure)
        self.map_canvas.setMinimumHeight(250)
        self.map_canvas.setMaximumHeight(350)
        
        # Add navigation toolbar for zooming/panning
        self.map_toolbar = NavigationToolbar(self.map_canvas, map_group)
        
        map_layout.addWidget(self.map_toolbar)
        map_layout.addWidget(self.map_canvas)
        
        return map_group
    
    def create_parameter_selection_group(self):
        """Create parameter selection checkboxes for export"""
        param_group = QGroupBox("Export Parameters")
        param_layout = QVBoxLayout(param_group)
        
        # Add info label
        info_label = QLabel("Select which parameters to include in NPC export:")
        info_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
        param_layout.addWidget(info_label)
        
        # Checkbox for oxygen
        self.oxygen_checkbox = QCheckBox("Include Dissolved Oxygen")
        self.oxygen_checkbox.setEnabled(False)  # Will be enabled when data is available
        self.oxygen_checkbox.toggled.connect(self.on_parameter_checkbox_changed)
        param_layout.addWidget(self.oxygen_checkbox)
        
        # Checkbox for chlorophyll
        self.chlorophyll_checkbox = QCheckBox("Include Chlorophyll")
        self.chlorophyll_checkbox.setEnabled(False)  # Will be enabled when data is available
        self.chlorophyll_checkbox.toggled.connect(self.on_parameter_checkbox_changed)
        param_layout.addWidget(self.chlorophyll_checkbox)
        
        # Status label for parameter availability
        self.param_status_label = QLabel("Parameters will be auto-detected from profile data")
        self.param_status_label.setStyleSheet("QLabel { font-size: 9px; color: #0066cc; }")
        self.param_status_label.setWordWrap(True)
        param_layout.addWidget(self.param_status_label)
        
        return param_group
    
    def create_actions_group(self):
        """Create export/action buttons"""
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        # Lasso selection controls
        lasso_layout = QHBoxLayout()
        self.lasso_mode_button = QPushButton('Enable Lasso Selection')
        self.lasso_mode_button.setCheckable(True)
        self.lasso_mode_button.clicked.connect(self.toggle_lasso_mode)
        self.lasso_mode_button.setEnabled(False)
        lasso_layout.addWidget(self.lasso_mode_button)
        
        self.clear_exclusions_button = QPushButton('Clear Exclusions')
        self.clear_exclusions_button.clicked.connect(self.clear_exclusions)
        self.clear_exclusions_button.setEnabled(False)
        lasso_layout.addWidget(self.clear_exclusions_button)
        
        action_layout.addLayout(lasso_layout)
        
        self.exclusion_label = QLabel("Excluded points: 0")
        self.exclusion_label.setStyleSheet("QLabel { color: #cc0000; font-size: 10px; }")
        action_layout.addWidget(self.exclusion_label)
        
        self.export_span_button = QPushButton('Export Selected Span')
        self.export_span_button.clicked.connect(self.export_selected_span)
        self.export_span_button.setEnabled(False)
        action_layout.addWidget(self.export_span_button)
        
        # Save NPC file button
        self.save_npc_button = QPushButton('Save selection as NPC File (1 m bins)')
        self.save_npc_button.clicked.connect(self.save_npc_file)
        self.save_npc_button.setEnabled(False)
        self.save_npc_button.setStyleSheet("QPushButton { font-weight: bold; color: #006600; }")
        action_layout.addWidget(self.save_npc_button)
        
        # Upload NPC to AWS button
        self.upload_npc_button = QPushButton('Upload selection to PhysChem')
        self.upload_npc_button.clicked.connect(self.upload_npc_to_aws)
        self.upload_npc_button.setEnabled(False)
        self.upload_npc_button.setStyleSheet("QPushButton { font-weight: bold; color: #0066cc; }")
        # action_layout.addWidget(self.upload_npc_button)
        
        return action_group
    
    def on_parameter_checkbox_changed(self):
        """Handle parameter checkbox state changes"""
        # Recalculate df_npc when checkbox state changes
        if hasattr(self, 'selected_span_indices') and len(self.selected_span_indices) > 0:
            self.df_npc, self.meta = self.calculate_df_npc(self.current_df_profile, self.selected_span_indices)
            self.update_profile_plots()
    
    def check_parameter_availability(self, df_profile):
        """Check which parameters are available in the current profile and update checkboxes"""
        # Check for oxygen data
        has_oxygen = False
        if 'dissolved_o2_concentration' in df_profile.columns:
            o2_data = df_profile['dissolved_o2_concentration'].dropna()
            has_oxygen = len(o2_data) > 0
        
        # Check for chlorophyll data
        has_chlorophyll = False
        if 'chlorophyll' in df_profile.columns:
            chl_data = df_profile['chlorophyll'].dropna()
            has_chlorophyll = len(chl_data) > 0
        
        # Update checkboxes based on availability
        self.oxygen_checkbox.setEnabled(has_oxygen)
        self.oxygen_checkbox.setChecked(has_oxygen)  # Auto-check if available
        
        self.chlorophyll_checkbox.setEnabled(has_chlorophyll)
        self.chlorophyll_checkbox.setChecked(has_chlorophyll)  # Auto-check if available
        
        # Update status label
        status_parts = []
        if has_oxygen:
            status_parts.append("O₂ ✓")
        else:
            status_parts.append("O₂ ✗")
        
        if has_chlorophyll:
            status_parts.append("Chl ✓")
        else:
            status_parts.append("Chl ✗")
        
        self.param_status_label.setText(f"Available parameters: {', '.join(status_parts)}")
        
        return has_oxygen, has_chlorophyll
    
    # [All file selection and data processing methods remain the same...]
    def select_rsk_files(self):
        """Select multiple RSK files and automatically process them"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select RSK Files", "", "RSK Files (*.rsk);;All Files (*)"
        )
        
        if files:
            self.rsk_files = files
            if len(files) == 1:
                self.rsk_files_label.setText(f"Selected: {os.path.basename(files[0])}")
            else:
                self.rsk_files_label.setText(f"Selected: {len(files)} RSK files")
            
            # Extract date range from RSK files (which will auto-process)
            self.extract_date_range_from_rsk()
    
    def extract_date_range_from_rsk(self):
        """Extract start and end dates from RSK files and match cruise"""
        if not PYRSK_AVAILABLE:
            self.date_range_label.setText("Error: pyrsktools not installed")
            self.start_date = None
            self.end_date = None
            return
        
        try:
            self.status_label.setText("Reading RSK files to detect date range...")
            QApplication.processEvents()
            
            all_timestamps = []
            
            for filepath in self.rsk_files:
                try:
                    rsk = pyrsktools.RSK(filepath)
                    rsk.open()
                    rsk.readdata()
                    
                    df_rsk = pd.DataFrame(rsk.data)
                    
                    if 'timestamp' in df_rsk.columns:
                        all_timestamps.extend(df_rsk['timestamp'].tolist())
                    
                    rsk.close()
                    
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")
                    continue
            
            if all_timestamps:
                # Convert to datetime and find min/max
                all_timestamps = pd.to_datetime(all_timestamps)
                
                # Ensure timezone-aware (UTC)
                # Note: all_timestamps is a DatetimeIndex, so access tz directly (no .dt)
                if all_timestamps.tz is None:
                    all_timestamps = all_timestamps.tz_localize('UTC')
                else:
                    all_timestamps = all_timestamps.tz_convert('UTC')
                
                min_time = all_timestamps.min()
                max_time = all_timestamps.max()
                
                # Store as timezone-aware datetime objects
                min_time_dt = min_time.to_pydatetime()
                max_time_dt = max_time.to_pydatetime()
                
                # Add some buffer (1 day before and after) for API queries
                self.start_date = (min_time - pd.Timedelta(days=1)).to_pydatetime()
                self.end_date = (max_time + pd.Timedelta(days=1)).to_pydatetime()
                
                # Update label
                self.date_range_label.setText(
                    f"Start: {self.start_date.strftime('%Y-%m-%d %H:%M')}\n"
                    f"End: {self.end_date.strftime('%Y-%m-%d %H:%M')}\n"
                    f"(Extracted from RSK files)"
                )
                
                self.status_label.setText("Fetching cruise information from API...")
                QApplication.processEvents()
                
                # Fetch cruise data and try to match
                df_cruises = get_cruises_from_api()
                
                if len(df_cruises) > 0:
                    # Use the actual data range (without buffer) for cruise matching
                    matched_cruise = match_cruise_by_dates(
                        min_time_dt, 
                        max_time_dt, 
                        df_cruises
                    )
                    
                    if matched_cruise:
                        # Fill in the cruise parameters
                        self.vessel_name_input.setText(str(matched_cruise.get('vesselName', '')))
                        self.cruise_number_input.setText(str(matched_cruise.get('cruiseNumber', '')))
                        self.platform_number_input.setText(str(matched_cruise.get('platform', '')))
                        
                        self.status_label.setText(
                            f"Cruise matched: {matched_cruise.get('cruiseName', 'Unknown')}. Starting data processing..."
                        )
                    else:
                        self.status_label.setText(
                            f"No matching cruise found. Starting data processing..."
                        )
                else:
                    self.status_label.setText(
                        f"Could not fetch cruise data. Starting data processing..."
                    )
                
                # Automatically start processing data
                QApplication.processEvents()
                self.process_data()
                
            else:
                self.date_range_label.setText("Error: No timestamps found in RSK files")
                self.start_date = None
                self.end_date = None
                self.status_label.setText("Error: Could not extract dates from RSK files")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.date_range_label.setText(f"Error reading RSK files: {str(e)}")
            self.start_date = None
            self.end_date = None
            self.status_label.setText(f"Error: {str(e)}")
    
    def process_data(self):
        """Start data processing in background thread (called automatically)"""
        if not self.rsk_files:
            return
        
        # Check if dates are available
        if not hasattr(self, 'start_date') or not hasattr(self, 'end_date'):
            QMessageBox.critical(self, "No Date Range", "Could not extract date range from RSK files")
            return
        
        if self.start_date is None or self.end_date is None:
            QMessageBox.critical(self, "No Date Range", "Could not extract date range from RSK files")
            return
        
        if not PYRSK_AVAILABLE:
            QMessageBox.critical(
                self, "Missing Dependency", 
                "pyrsktools is required to process RSK files.\n\n"
                "Install with: pip install pyrsktools"
            )
            return
        
        # Disable UI during processing
        self.select_rsk_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start processing thread with extracted dates
        self.processing_thread = DataProcessingThread(self.rsk_files, self.start_date, self.end_date)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.data_ready.connect(self.on_data_ready)
        self.processing_thread.error_occurred.connect(self.on_error)
        self.processing_thread.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
    
    def on_data_ready(self, station_matches, df_rsk_all):
        """Handle completed data processing"""
        self.station_matches = station_matches
        self.df_rsk_all = df_rsk_all
        self.station_keys = list(station_matches.keys())
        self.current_index = 0

        # Get cruise infos from API data using stored dates
        try:
            after_str = self.start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            before_str = self.end_date.strftime('%Y-%m-%dT%H:%M:%S.999Z')
            
            df_tk = get_activities_from_api(after_str, before_str)
            self.cruise_start_time = df_tk['startTime'].min()
            self.cruise_end_time = df_tk['endTime'].max()
        except Exception as e:
            print(f"Warning: Could not load cruise times: {e}")
            self.cruise_start_time = self.start_date
            self.cruise_end_time = self.end_date
        
        rsk = pyrsktools.RSK(self.rsk_files[0])
        rsk.open()
        rsk_meta={}
        rsk_meta['instrument.instrumentSerialNumber']=rsk.instrument.serialID                
        rsk_meta['instrument.instrumentModel']=rsk.instrument.model     
        self.rsk_meta = rsk_meta
  
        # Re-enable UI
        self.select_rsk_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Enable navigation and action buttons
        self.prev_button.setEnabled(len(self.station_keys) > 1)
        self.next_button.setEnabled(len(self.station_keys) > 1)
        self.export_span_button.setEnabled(len(self.station_keys) > 0)
        self.save_npc_button.setEnabled(len(self.station_keys) > 0)
        self.upload_npc_button.setEnabled(len(self.station_keys) > 0)
        self.lasso_mode_button.setEnabled(len(self.station_keys) > 0)
        self.clear_exclusions_button.setEnabled(len(self.station_keys) > 0)

        # Create map and display first profile
        if self.station_keys:
            if CARTOPY_AVAILABLE:
                self.create_matplotlib_map()
            else:
                self.create_simple_map()
            self.display_current_profile()
            
            # Count how many stations had time corrections
            corrected_count = sum(1 for k in self.station_keys 
                                if self.station_matches[k]['station_info'].get('time_corrected', False))
            
            if corrected_count > 0:
                self.status_label.setText(
                    f"Loaded {len(self.station_keys)} stations successfully! "
                    f"({corrected_count} start times corrected)"
                )
            else:
                self.status_label.setText(f"Loaded {len(self.station_keys)} stations successfully!")
        else:
            self.status_label.setText("No matching stations found.")
    
    def on_error(self, error_message):
        """Handle processing errors"""
        # Re-enable UI
        self.select_rsk_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Show error message
        QMessageBox.critical(self, "Processing Error", error_message)
        self.status_label.setText("Error occurred during processing.")
    
    # [Map creation methods remain the same...]
    def create_matplotlib_map(self):
        """Create a matplotlib map with coastlines showing all station locations"""
        if not CARTOPY_AVAILABLE:
            # Fallback to simple scatter plot without coastlines
            self.create_simple_map()
            return
        
        # Collect all station coordinates
        all_lats = []
        all_lons = []
        
        for station_key, data in self.station_matches.items():
            station_info = data['station_info']
            all_lats.append(station_info['startLat'])
            all_lons.append(station_info['startLon'])
        
        if not all_lats:
            self.map_figure.clear()
            ax = self.map_figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No stations to display', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            self.map_canvas.draw()
            return
        
        # Calculate map bounds with padding
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)
        
        # Add 10% padding
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        padding = max(lat_range, lon_range) * 0.1 + 0.1  # At least 0.1 degree padding
        
        lat_min -= padding
        lat_max += padding
        lon_min -= padding
        lon_max += padding
        
        # Clear figure and create map
        self.map_figure.clear()
        ax = self.map_figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.5)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                         alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot all stations
        for i, (station_key, data) in enumerate(self.station_matches.items()):
            station_info = data['station_info']
            lat = station_info['startLat']
            lon = station_info['startLon']
            
            # Check if time was corrected
            time_corrected = station_info.get('time_corrected', False)
            
            if i == self.current_index:
                # Current station - large red marker with star
                ax.plot(lon, lat, marker='*', markersize=20, color='red', 
                       transform=ccrs.PlateCarree(), zorder=5,
                       markeredgecolor='darkred', markeredgewidth=1.5)
            else:
                # Other stations - smaller markers
                if time_corrected:
                    # Time-corrected stations in orange
                    ax.plot(lon, lat, marker='o', markersize=8, color='orange', 
                           transform=ccrs.PlateCarree(), zorder=4,
                           markeredgecolor='darkorange', markeredgewidth=1)
                else:
                    # Normal stations in blue
                    ax.plot(lon, lat, marker='o', markersize=8, color='blue', 
                           transform=ccrs.PlateCarree(), zorder=4,
                           markeredgecolor='darkblue', markeredgewidth=1)
        
        # Add title with current station info
        if self.current_index < len(self.station_keys):
            station_key = self.station_keys[self.current_index]
            data = self.station_matches[station_key]
            station_info = data['station_info']
            title = f"Station: {station_info['name']} (Activity {station_info['activityNumber']})"
            ax.set_title(title, fontsize=10, fontweight='bold')
        
        self.map_figure.tight_layout()
        self.map_canvas.draw()
    
    def create_simple_map(self):
        """Fallback simple scatter plot map without coastlines (if cartopy not available)"""
        # Collect all station coordinates
        all_lats = []
        all_lons = []
        
        for station_key, data in self.station_matches.items():
            station_info = data['station_info']
            all_lats.append(station_info['startLat'])
            all_lons.append(station_info['startLon'])
        
        if not all_lats:
            return
        
        # Clear figure and create simple plot
        self.map_figure.clear()
        ax = self.map_figure.add_subplot(1, 1, 1)
        
        # Plot all stations
        other_lats = []
        other_lons = []
        corrected_lats = []
        corrected_lons = []
        
        for i, (station_key, data) in enumerate(self.station_matches.items()):
            station_info = data['station_info']
            lat = station_info['startLat']
            lon = station_info['startLon']
            
            if i == self.current_index:
                # Current station
                ax.plot(lon, lat, marker='*', markersize=20, color='red', 
                       zorder=5, markeredgecolor='darkred', markeredgewidth=1.5,
                       label='Current Station')
            else:
                time_corrected = station_info.get('time_corrected', False)
                if time_corrected:
                    corrected_lats.append(lat)
                    corrected_lons.append(lon)
                else:
                    other_lats.append(lat)
                    other_lons.append(lon)
        
        # Plot other stations
        if other_lons:
            ax.plot(other_lons, other_lats, 'o', markersize=8, color='blue',
                   markeredgecolor='darkblue', markeredgewidth=1, label='Stations')
        if corrected_lons:
            ax.plot(corrected_lons, corrected_lats, 'o', markersize=8, color='orange',
                   markeredgecolor='darkorange', markeredgewidth=1, label='Time Corrected')
        
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Add title
        if self.current_index < len(self.station_keys):
            station_key = self.station_keys[self.current_index]
            data = self.station_matches[station_key]
            station_info = data['station_info']
            title = f"Station: {station_info['name']} (Activity {station_info['activityNumber']})"
            ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Set aspect ratio to be roughly geographic
        ax.set_aspect('equal', adjustable='box')
        
        self.map_figure.tight_layout()
        self.map_canvas.draw()
    
    def update_station_map(self):
        """Update the map to highlight the current station"""
        if CARTOPY_AVAILABLE:
            self.create_matplotlib_map()
        else:
            self.create_simple_map()
    
    def calculate_df_npc(self, df_profile, selected_indices):
        """Calculate binned/averaged data (df_npc) from selected indices, excluding marked points"""
        if len(selected_indices) == 0:
            return pd.DataFrame(), {}
        
        # Get data for selected indices, excluding marked points
        valid_indices = [idx for idx in selected_indices if idx not in self.excluded_indices]
        
        if len(valid_indices) == 0:
            return pd.DataFrame(), {}
        
        selected_data = df_profile.loc[valid_indices]
        
        # Calculate bins based on depth range of selected data
        depth_min = selected_data['depth'].min()
        depth_max = selected_data['depth'].max()
        
        if depth_max <= depth_min:
            return pd.DataFrame(), {}
        
        # Create bins
        interval = 1  # bin size in meters
        bin_max = np.round(depth_max)
        bin_min = np.round(depth_min)
        bins_calc = np.arange(bin_min-interval/2, bin_max + interval, interval)
        
        if len(bins_calc) < 2:
            return pd.DataFrame(), {}
        
        bins = bins_calc[:-1] + interval/2
        
        # Channels to export (base channels always included)
        channels_to_export = ['timestamp', 'conductivity', 'temperature', 'pressure', 'salinity']
        
        # Check if optional parameters should be included based on checkbox states
        if self.oxygen_checkbox.isChecked() and self.oxygen_checkbox.isEnabled():
            channels_to_export.append('dissolved_o2_concentration')
        
        if self.chlorophyll_checkbox.isChecked() and self.chlorophyll_checkbox.isEnabled():
            channels_to_export.append('chlorophyll')
        
        # Filter channels that exist in the dataframe
        available_channels = [c for c in channels_to_export if c in df_profile.columns]
                
        #######
        meta={}
        year = self.cruise_start_time.year
            
        meta['mission.missionNumber']=self.missionnumber_name_input.text().strip()
        meta['mission.missionStartDate']= self.cruise_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        meta['mission.missionStopDate']=self.cruise_end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        meta['mission.missionType']='14'
        meta['mission.platform']=self.platform_number_input.text().strip()
        meta['mission.startYear']= year
        meta['mission.platformName']=self.vessel_name_input.text().strip()
        meta['mission.missionTypeName']= 'Cruise'
        meta['mission.purpose']= 'Cruise'
        meta['mission.missionName']= self.cruise_number_input.text().strip()
        meta['mission.cruise']= self.cruise_number_input.text().strip()
        meta['mission.responsibleLaboratory']= 3
        meta['operation.operationType']= 'CTD'
        meta['operation.operationNumber']= self.current_index+1
        # meta['operation.callSignal']=self.callsign_name_input.text().strip()
        meta['operation.timeStart']= pd.Timestamp( df_profile["timestamp"].min() ).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta['operation.timeEnd']= pd.Timestamp( df_profile["timestamp"].max() ).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta['operation.timeStartQuality']=0
        meta['operation.timeEndQuality']=0
        meta['operation.featureType']=4
        
        station_key = self.station_keys[self.current_index]
        data = self.station_matches[station_key]
        station_info = data['station_info']
        
        meta['operation.latitudeStart']=station_info['startLat']
        meta['operation.longitudeStart']=station_info['startLon']
        meta['operation.positionStartQuality']=0
        meta['operation.stationType']=1000
        meta['operation.localCdiId']= str(uuid.uuid1())
        meta['operation.operationComment']= station_info['comment']
        meta['operation.operationPlatform']= meta['mission.platform']

        meta['instrument.instrumentNumber']=1
        meta['instrument.instrumentType']='CTD'
        meta['instrument.instrumentSerialNumber']= self.rsk_meta['instrument.instrumentSerialNumber']
        meta['instrument.instrumentModel']=self.rsk_meta['instrument.instrumentModel']
        meta['instrument.instrumentDataOwner']=3
        meta['instrument.instrumentProperty.profileDirection']='D'
        
        #########
        channels_to_export=[]
        channels_NPC_codes=[]
        channels_NPC_suppliedParameterName=[]
        channels_NPC_units=[ ]
        channels_NPC_suppliedUnits=[ ]
        channels_NPC_acquirementMethod=[ ]
        channels_NPC_parameterName=[]
        channels_NPC_processingLevel=[]

        channels_to_export.append('timestamp' )
        channels_NPC_codes.append('DATETIME' )
        channels_NPC_suppliedParameterName.append('DateTime' )
        channels_NPC_units.append('ISO8601 UTC' )
        channels_NPC_suppliedUnits.append('ISO8601 UTC' )
        channels_NPC_parameterName.append('Date and time' )
        channels_NPC_processingLevel.append('L0' )
        channels_NPC_acquirementMethod.append('1019900')

        channels_to_export.append('conductivity' )
        channels_NPC_codes.append('COND' )
        channels_NPC_suppliedParameterName.append('Conductivity' )
        channels_NPC_units.append('mS/cm' )
        channels_NPC_suppliedUnits.append('mS/cm' )
        channels_NPC_parameterName.append('Electrical conductivity of water' )
        channels_NPC_processingLevel.append('L0' )
        channels_NPC_acquirementMethod.append('1019900')
        
        channels_to_export.append('temperature' )
        channels_NPC_codes.append('TEMP' )
        channels_NPC_suppliedParameterName.append('Temperature' )
        channels_NPC_units.append('degC' )
        channels_NPC_suppliedUnits.append('degC' )
        channels_NPC_parameterName.append('Temperature of water' )
        channels_NPC_processingLevel.append('L0' )
        channels_NPC_acquirementMethod.append('1019900')

        # channels_to_export.append('pressure' )
        # channels_NPC_codes.append('TOTPRES' )
        # channels_NPC_suppliedParameterName.append('Pressure' )
        # channels_NPC_units.append('dbar' )
        # channels_NPC_suppliedUnits.append('dbar' )
        # channels_NPC_parameterName.append('Total Pressure' )
        # channels_NPC_processingLevel.append('L0' )
        # channels_NPC_acquirementMethod.append('1019900')
        
        channels_to_export.append('pressure' )
        channels_NPC_codes.append('PRES' )
        channels_NPC_suppliedParameterName.append('Pressure' )
        channels_NPC_units.append('dbar' )
        channels_NPC_suppliedUnits.append('dbar' )
        channels_NPC_parameterName.append('Sea Pressure' )
        channels_NPC_processingLevel.append('L0' )
        channels_NPC_acquirementMethod.append('1019900')
        

        channels_to_export.append('salinity' )
        channels_NPC_codes.append('PSAL' )
        channels_NPC_suppliedParameterName.append('Salinity' )
        channels_NPC_units.append('PSU' )
        channels_NPC_suppliedUnits.append('PSU' )
        channels_NPC_parameterName.append('Practical salinity' )
        channels_NPC_processingLevel.append('L0' )
        channels_NPC_acquirementMethod.append('1019900')

        # Only add oxygen if checkbox is checked and data is available
        if ('dissolved_o2_concentration' in available_channels and 
            self.oxygen_checkbox.isChecked() and self.oxygen_checkbox.isEnabled()): 
            channels_to_export.append('dissolved_o2_concentration' )
            channels_NPC_codes.append('DOX' )
            channels_NPC_suppliedParameterName.append('Dissolved O2' )
            channels_NPC_units.append('umol/l' )
            channels_NPC_suppliedUnits.append('umol/l' )
            channels_NPC_parameterName.append('Dissolved oxygen from in-situ sensor' )
            channels_NPC_processingLevel.append('L0' )
            channels_NPC_acquirementMethod.append('1019900')  
        
        # Only add chlorophyll if checkbox is checked and data is available
        if ('chlorophyll' in available_channels and 
            self.chlorophyll_checkbox.isChecked() and self.chlorophyll_checkbox.isEnabled()): 
            channels_to_export.append('chlorophyll' )
            channels_NPC_codes.append('ChlA_SENS' )
            channels_NPC_suppliedParameterName.append('Chlorophyll' )
            channels_NPC_units.append('ug/l' )
            channels_NPC_suppliedUnits.append('ug/l' )
            channels_NPC_parameterName.append('Chlorophyll-a fluorescence from in-situ sensor' )
            channels_NPC_processingLevel.append('L0' )
            channels_NPC_acquirementMethod.append('1019900')

        parameter_id=np.arange(len(channels_to_export))+1
        cols = []
        for i in parameter_id:
            if channels_NPC_codes[i-1]=='DATETIME':
                cols.append( channels_NPC_codes[i-1] + '.value')
                cols.append(  channels_NPC_codes[i-1] + '.sampleSize')
            else:    
                cols.append( channels_NPC_codes[i-1] + '.value')
                cols.append(  channels_NPC_codes[i-1] + '.std')
                cols.append(  channels_NPC_codes[i-1] + '.sampleSize')
            
            meta['parameter{' + str(i) + '}.parameterCode'] = channels_NPC_codes[i-1]
            meta['parameter{' + str(i) + '}.units'] = channels_NPC_units[i-1]
            meta['parameter{' + str(i) + '}.suppliedUnits'] = channels_NPC_suppliedUnits[i-1]
            meta['parameter{' + str(i) + '}.parameterName'] = channels_NPC_parameterName[i-1]
            meta['parameter{' + str(i) + '}.suppliedParameterName'] = channels_NPC_suppliedParameterName[i-1]
            meta['parameter{' + str(i) + '}.acquirementMethod'] = channels_NPC_acquirementMethod[i-1]
            meta['parameter{' + str(i) + '}.processingLevel'] = channels_NPC_processingLevel[i-1]
            
        # add depth     
        cols.append( 'DEPTH.value')
        meta['parameter{' + str(i+1) + '}.parameterCode'] = 'DEPTH'
        meta['parameter{' + str(i+1) + '}.units'] = 'm'
        meta['parameter{' + str(i+1) + '}.suppliedUnits'] = 'm'
        meta['parameter{' + str(i+1) + '}.parameterName'] = 'Depth below sea level'
        meta['parameter{' + str(i+1) + '}.suppliedParameterName'] = 'Sea Pressure' 
        meta['parameter{' + str(i+1) + '}.acquirementMethod'] = '1019900'
        meta['parameter{' + str(i+1) + '}.processingLevel'] = 'L0'        
 
        # # add depth     
        # cols.append( 'PRES.value')
        # meta['parameter{' + str(i+1) + '}.parameterCode'] = 'PRES'
        # meta['parameter{' + str(i+1) + '}.units'] = 'dbar'
        # meta['parameter{' + str(i+1) + '}.suppliedUnits'] = 'dbar'
        # meta['parameter{' + str(i+1) + '}.parameterName'] = 'Sea Pressure'
        # meta['parameter{' + str(i+1) + '}.suppliedParameterName'] = 'Sea Pressure' 
        # meta['parameter{' + str(i+1) + '}.acquirementMethod'] = '1019900'
        # meta['parameter{' + str(i+1) + '}.processingLevel'] = 'L0'  
        
        df_npc=pd.DataFrame(columns=cols)
        ii=0
        for i in range(len(bins)):
            ix_bin = (selected_data['depth'] >= bins_calc[i]) & (selected_data['depth'] < bins_calc[i+1])
            line =[]
            
            if sum(ix_bin)>0:
                for c in channels_to_export:
                    y_bin = selected_data.loc[ix_bin, c].values
                    if c == 'timestamp':
                        y_bin= pd.to_datetime(y_bin)  
                        line.append( y_bin.mean().strftime("%Y-%m-%dT%H:%M:%SZ") )
                        line.append( len(y_bin) )       
                    else:                  
                        line.append( y_bin.mean() )
                        line.append( y_bin.std() )
                        line.append( len(y_bin) )       
                line.append( int(bins[i])  )
                # line.append( int(bins[i])  )
                df_npc.loc[ii,:] = line
                ii=ii+1        
                
        # breakpoint()        
        dfff=pd.DataFrame()
        dfff['sampleNumber']=np.arange(len(df_npc))+1     
        df_npc = pd.concat([dfff,df_npc],axis=1)
        
        # print(df_npc)
        

        return df_npc , meta
    
    # Modified display_current_profile method to check parameter availability
    def display_current_profile(self):
        """Display the current profile data and update UI elements"""
        if not self.station_keys:
            return
            
        station_key = self.station_keys[self.current_index]
        data = self.station_matches[station_key]
        
        # Update profile label
        self.profile_label.setText(f"Profile {self.current_index + 1} of {len(self.station_keys)}\n{station_key}")
        
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.station_keys) - 1)
        
        # Update combined station information and statistics
        station_info = data['station_info']
        
        combined_info = f"""Station Name: {station_info['name']}
Activity Number: {station_info['activityNumber']}
Start Time: {station_info['startTime']}
End Time: {station_info['endTime']}
Location: {station_info['startLat']:.5f}°N, {station_info['startLon']:.5f}°E
Comment: {station_info['comment']}
Data Points: {data['n_datapoints']}
"""
        
        # Add time correction info if applicable
        if station_info.get('time_corrected', False):
            combined_info += f"\n⚠️ TIME CORRECTED\nOriginal Start: {station_info['original_startTime']}\n{station_info.get('correction_info', '')}"
        
        self.info_text.setText(combined_info)
        
        self.df_npc = pd.DataFrame()

        # Extract and display profile data
        if data['n_datapoints'] > 0:
            df_profile = self.df_rsk_all.loc[data['df_rsk_indices'], :].copy()
            df_profile = df_profile.reset_index(drop=True)
            
            # Check parameter availability and update checkboxes
            self.check_parameter_availability(df_profile)
            
            # get downcast
            df_profile = df_profile.reset_index(drop=True)
            peaks, p = find_peaks(df_profile['depth'], height=df_profile['depth'].max()*0.25, width=500)
             
            lb = np.zeros(len(df_profile))
            k = 1
            pp_before = 0
            for pp in peaks:
                dmax = df_profile.loc[int(p['left_ips'][k-1]):int(p['right_ips'][k-1]), 'depth'].max()
                ix_down = (df_profile['depth']>0.5) & (df_profile['depth']<dmax) & (df_profile['salinity']>0) & (df_profile.index<pp) & (df_profile.index>pp_before) & (df_profile['velocity'].rolling(100,center=True).mean()>0.1)
                lb[ix_down] = k
                k = k + 1
                pp_before = pp
            
            ix_down = lb > 0
            
            lb_up = np.zeros(len(df_profile))
            k = 1
            ppp = np.append(p['left_bases'], len(df_profile)-1)
            for i in range(len(peaks)):
                ix_up = (df_profile['depth']>0.5) & (df_profile['salinity']>0) & (df_profile.index>peaks[i]) & (df_profile.index<ppp[i+1]) & (df_profile['velocity'].rolling(100,center=True).mean()<-0.1)
                lb_up[ix_up] = k
                k = k + 1
            
            d_old = 0 
            k = 1
            lb_updated = np.zeros(len(lb))
            for j in np.unique(lb)[1:]:
               ix1 = np.where(lb==j)[0][0] 
               ix2 = np.where(lb==j)[0][-1] 
               d_new = df_profile.loc[ix1, 'depth']
               
               if (d_old - d_new) < df_profile['depth'].max()*0.5:
                   lb_updated[lb==j] = k
               else:
                   k = k + 1
                   lb_updated[lb==j] = k
               d_old = df_profile.loc[ix2, 'depth']
               
            for j in np.unique(lb_updated)[1:]:
                lbleng = np.where(lb_updated==j)[0][-1] - np.where(lb_updated==j)[0][0]
                if lbleng < 500:
                    lb_updated[lb_updated==j] = 0
                   
            lb = lb_updated
            profile_labels = np.unique(lb)[1:]
            ix_down_profile = lb == 1
            ix_up_profile = lb_up == 1   
            
            # Calculate initial df_npc using downcast profile
            if ix_down_profile.any():
                down_indices = df_profile.index[ix_down_profile]
                self.df_npc,self.meta = self.calculate_df_npc(df_profile, down_indices)
                
            # cut off after upcast
            up_indices = df_profile.index[ix_up_profile]
            if len(up_indices)>0:
                df_profile=df_profile.iloc[:up_indices[-1],:]
                ix_down_profile=ix_down_profile[:up_indices[-1]]
                
            # Plot the profile
            self.plot_profile(df_profile, station_key, ix_down_profile)
            
            # Update map to highlight current station
            self.update_station_map()
        else:
            # Clear checkboxes if no data
            self.oxygen_checkbox.setEnabled(False)
            self.oxygen_checkbox.setChecked(False)
            self.chlorophyll_checkbox.setEnabled(False)
            self.chlorophyll_checkbox.setChecked(False)
            self.param_status_label.setText("No profile data available")
            
            self.figure.clear()
            self.canvas.draw()
            self.update_station_map()
    
    # [All other methods remain the same...]
    def update_profile_plots(self):
        """Update only the profile plots with new df_npc data"""
        if not hasattr(self, 'plot_axes') or not self.plot_axes:
            return
        
        # Clear and replot binned data on each axis
        for ax_name in ['temp', 'sal', 'o2', 'chl']:
            if ax_name in self.plot_axes:
                ax = self.plot_axes[ax_name]
                
                # Remove existing binned data line if present
                for line in ax.lines[:]:
                    if line.get_label() == 'binned data':
                        line.remove()
                
                # Add new binned data if available
                if len(self.df_npc) > 0:
                    if ax_name == 'temp' and 'TEMP.value' in self.df_npc.columns:
                        ax.plot(self.df_npc['TEMP.value'], -self.df_npc['DEPTH.value'], '-r', label='binned data')
                    elif ax_name == 'sal' and 'PSAL.value' in self.df_npc.columns:
                        ax.plot(self.df_npc['PSAL.value'], -self.df_npc['DEPTH.value'], '-r', label='binned data')
                    elif ax_name == 'o2' and 'DOX.value' in self.df_npc.columns:
                        ax.plot(self.df_npc['DOX.value'], -self.df_npc['DEPTH.value'], '-r', label='binned data')
                    elif ax_name == 'chl' and 'ChlA_SENS.value' in self.df_npc.columns:
                        ax.plot(self.df_npc['ChlA_SENS.value'], -self.df_npc['DEPTH.value'], '-r', label='binned data')
                
                # Update legend for temperature plot
                if ax_name == 'temp':
                    ax.legend(loc=3)
        
        self.canvas.draw_idle()
    
    def plot_profile(self, df_profile, station_key, ix_down_profile):
        """Plot the CTD profile using subplot2grid layout with span selector"""
        self.figure.clear()
        
        # Store df_profile as instance variable for access in callback
        self.current_df_profile = df_profile
        
        # Create subplot layout and store axes references
        ax1 = plt.subplot2grid((3,4), (0,0), rowspan=2, fig=self.figure)  # Temperature
        ax2 = plt.subplot2grid((3,4), (0,1), rowspan=2, fig=self.figure)  # Salinity  
        ax3 = plt.subplot2grid((3,4), (0,2), rowspan=2, fig=self.figure)  # Oxygen
        ax4 = plt.subplot2grid((3,4), (0,3), rowspan=2, fig=self.figure)  # Chlorophyll
        ax5 = plt.subplot2grid((3,4), (2,0), colspan=4, fig=self.figure)  # Time series
        
        # Store axes for updating
        self.plot_axes = {'temp': ax1, 'sal': ax2, 'o2': ax3, 'chl': ax4, 'time': ax5}
        
        # Temperature profile
        # Plot all data points with color
        temp_scatter = ax1.scatter(df_profile['temperature'], -df_profile['depth'], 
                   s=10, c=df_profile.index.values, label='raw data', picker=True)
        
        # Overlay excluded points in red
        if len(self.excluded_indices) > 0:
            excluded_in_profile = [idx for idx in self.excluded_indices if idx in df_profile.index]
            if excluded_in_profile:
                ax1.scatter(df_profile.loc[excluded_in_profile, 'temperature'], 
                          -df_profile.loc[excluded_in_profile, 'depth'],
                          s=20, c='red', marker='x', label='excluded', zorder=10)
        
        if len(self.df_npc) > 0:
            ax1.plot(self.df_npc['TEMP.value'], -self.df_npc['DEPTH.value'], '-r', label='binned data')           
        
        ax1.set_title('Temperature (Lasso to exclude)')
        ax1.set_xlabel('Degree Celsius')
        ax1.set_ylabel('Depth in m')
        ax1.grid()
        ax1.legend(loc=3)
        
        # Salinity profile
        sal_scatter = ax2.scatter(df_profile['salinity'], -df_profile['depth'], 
                   s=10, c=df_profile.index.values, picker=True)
        
        # Overlay excluded points in red
        if len(self.excluded_indices) > 0:
            excluded_in_profile = [idx for idx in self.excluded_indices if idx in df_profile.index]
            if excluded_in_profile:
                ax2.scatter(df_profile.loc[excluded_in_profile, 'salinity'], 
                          -df_profile.loc[excluded_in_profile, 'depth'],
                          s=20, c='red', marker='x', label='excluded', zorder=10)
        
        if len(self.df_npc) > 0:
            ax2.plot(self.df_npc['PSAL.value'], -self.df_npc['DEPTH.value'], '-r', label='binned data')      
        ax2.set_title('Salinity (Lasso to exclude)')
        ax2.set_xlabel('PSU')
        ax2.set_ylabel('Depth in m')
        ax2.grid()
        
        # Create lasso selector callbacks
        def onselect_temp(verts):
            """Callback for temperature lasso selection"""
            if not self.lasso_mode_button.isChecked():
                return
            
            path = Path(verts)
            # Get x, y coordinates of all points
            xys = np.column_stack([df_profile['temperature'].values, -df_profile['depth'].values])
            # Find points inside the lasso
            ind = path.contains_points(xys)
            # Add these indices to excluded set
            selected_indices = df_profile.index[ind].tolist()
            self.excluded_indices.update(selected_indices)
            
            # Update label
            self.exclusion_label.setText(f"Excluded points: {len(self.excluded_indices)}")
            
            # Recalculate df_npc
            if hasattr(self, 'selected_span_indices') and len(self.selected_span_indices) > 0:
                self.df_npc, self.meta = self.calculate_df_npc(self.current_df_profile, self.selected_span_indices)
                self.update_profile_plots()
            
            # Redraw with exclusions marked
            station_key = self.station_keys[self.current_index]
            data = self.station_matches[station_key]
            df_profile_full = self.df_rsk_all.loc[data['df_rsk_indices'], :].copy()
            df_profile_full = df_profile_full.reset_index(drop=True)
            
            # Recalculate downcast indices
            peaks, p = find_peaks(df_profile_full['depth'], height=df_profile_full['depth'].max()*0.25, width=500)
            lb = np.zeros(len(df_profile_full))
            k = 1
            pp_before = 0
            for pp in peaks:
                dmax = df_profile_full.loc[int(p['left_ips'][k-1]):int(p['right_ips'][k-1]), 'depth'].max()
                ix_down = (df_profile_full['depth']>0.5) & (df_profile_full['depth']<dmax) & (df_profile_full['salinity']>0) & (df_profile_full.index<pp) & (df_profile_full.index>pp_before) & (df_profile_full['velocity'].rolling(100,center=True).mean()>0.1)
                lb[ix_down] = k
                k = k + 1
                pp_before = pp
            
            lb_up = np.zeros(len(df_profile_full))
            k = 1
            ppp = np.append(p['left_bases'], len(df_profile_full)-1)
            for i in range(len(peaks)):
                ix_up = (df_profile_full['depth']>0.5) & (df_profile_full['salinity']>0) & (df_profile_full.index>peaks[i]) & (df_profile_full.index<ppp[i+1]) & (df_profile_full['velocity'].rolling(100,center=True).mean()<-0.1)
                lb_up[ix_up] = k
                k = k + 1
            
            d_old = 0 
            k = 1
            lb_updated = np.zeros(len(lb))
            for j in np.unique(lb)[1:]:
               ix1 = np.where(lb==j)[0][0] 
               ix2 = np.where(lb==j)[0][-1] 
               d_new = df_profile_full.loc[ix1, 'depth']
               
               if (d_old - d_new) < df_profile_full['depth'].max()*0.5:
                   lb_updated[lb==j] = k
               else:
                   k = k + 1
                   lb_updated[lb==j] = k
               d_old = df_profile_full.loc[ix2, 'depth']
               
            for j in np.unique(lb_updated)[1:]:
                lbleng = np.where(lb_updated==j)[0][-1] - np.where(lb_updated==j)[0][0]
                if lbleng < 500:
                    lb_updated[lb_updated==j] = 0
                   
            lb = lb_updated
            ix_down_profile = lb == 1
            ix_up_profile = lb_up == 1   
            
            up_indices = df_profile_full.index[ix_up_profile]
            if len(up_indices)>0:
                df_profile_plot = df_profile_full.iloc[:up_indices[-1],:]
                ix_down_plot = ix_down_profile[:up_indices[-1]]
            else:
                df_profile_plot = df_profile_full
                ix_down_plot = ix_down_profile
            
            self.plot_profile(df_profile_plot, station_key, ix_down_plot)
        
        def onselect_sal(verts):
            """Callback for salinity lasso selection"""
            if not self.lasso_mode_button.isChecked():
                return
            
            path = Path(verts)
            # Get x, y coordinates of all points
            xys = np.column_stack([df_profile['salinity'].values, -df_profile['depth'].values])
            # Find points inside the lasso
            ind = path.contains_points(xys)
            # Add these indices to excluded set
            selected_indices = df_profile.index[ind].tolist()
            self.excluded_indices.update(selected_indices)
            
            # Update label
            self.exclusion_label.setText(f"Excluded points: {len(self.excluded_indices)}")
            
            # Recalculate df_npc
            if hasattr(self, 'selected_span_indices') and len(self.selected_span_indices) > 0:
                self.df_npc, self.meta = self.calculate_df_npc(self.current_df_profile, self.selected_span_indices)
                self.update_profile_plots()
            
            # Redraw with exclusions marked (same code as above)
            station_key = self.station_keys[self.current_index]
            data = self.station_matches[station_key]
            df_profile_full = self.df_rsk_all.loc[data['df_rsk_indices'], :].copy()
            df_profile_full = df_profile_full.reset_index(drop=True)
            
            peaks, p = find_peaks(df_profile_full['depth'], height=df_profile_full['depth'].max()*0.25, width=500)
            lb = np.zeros(len(df_profile_full))
            k = 1
            pp_before = 0
            for pp in peaks:
                dmax = df_profile_full.loc[int(p['left_ips'][k-1]):int(p['right_ips'][k-1]), 'depth'].max()
                ix_down = (df_profile_full['depth']>0.5) & (df_profile_full['depth']<dmax) & (df_profile_full['salinity']>0) & (df_profile_full.index<pp) & (df_profile_full.index>pp_before) & (df_profile_full['velocity'].rolling(100,center=True).mean()>0.1)
                lb[ix_down] = k
                k = k + 1
                pp_before = pp
            
            lb_up = np.zeros(len(df_profile_full))
            k = 1
            ppp = np.append(p['left_bases'], len(df_profile_full)-1)
            for i in range(len(peaks)):
                ix_up = (df_profile_full['depth']>0.5) & (df_profile_full['salinity']>0) & (df_profile_full.index>peaks[i]) & (df_profile_full.index<ppp[i+1]) & (df_profile_full['velocity'].rolling(100,center=True).mean()<-0.1)
                lb_up[ix_up] = k
                k = k + 1
            
            d_old = 0 
            k = 1
            lb_updated = np.zeros(len(lb))
            for j in np.unique(lb)[1:]:
               ix1 = np.where(lb==j)[0][0] 
               ix2 = np.where(lb==j)[0][-1] 
               d_new = df_profile_full.loc[ix1, 'depth']
               
               if (d_old - d_new) < df_profile_full['depth'].max()*0.5:
                   lb_updated[lb==j] = k
               else:
                   k = k + 1
                   lb_updated[lb==j] = k
               d_old = df_profile_full.loc[ix2, 'depth']
               
            for j in np.unique(lb_updated)[1:]:
                lbleng = np.where(lb_updated==j)[0][-1] - np.where(lb_updated==j)[0][0]
                if lbleng < 500:
                    lb_updated[lb_updated==j] = 0
                   
            lb = lb_updated
            ix_down_profile = lb == 1
            ix_up_profile = lb_up == 1   
            
            up_indices = df_profile_full.index[ix_up_profile]
            if len(up_indices)>0:
                df_profile_plot = df_profile_full.iloc[:up_indices[-1],:]
                ix_down_plot = ix_down_profile[:up_indices[-1]]
            else:
                df_profile_plot = df_profile_full
                ix_down_plot = ix_down_profile
            
            self.plot_profile(df_profile_plot, station_key, ix_down_plot)
        
        # Create lasso selectors (initially inactive)
        self.lasso_temp = LassoSelector(ax1, onselect_temp, button=1)
        self.lasso_sal = LassoSelector(ax2, onselect_sal, button=1)
        
        # Set initial state based on button
        if hasattr(self, 'lasso_mode_button'):
            if self.lasso_mode_button.isChecked():
                self.lasso_temp.set_active(True)
                self.lasso_sal.set_active(True)
            else:
                self.lasso_temp.set_active(False)
                self.lasso_sal.set_active(False)
        
        # Oxygen profile (if available)
        try:
            ax3.set_title('Dissolved oxygen concentration')
            ax3.set_xlabel('umol/l')
            ax3.set_ylabel('Depth in m')
            
            if 'dissolved_o2_concentration' in df_profile.columns:
                o2_data = df_profile['dissolved_o2_concentration'].dropna()
                if len(o2_data) > 0:
                    ax3.scatter(df_profile['dissolved_o2_concentration'], -df_profile['depth'], 
                               s=10, c=df_profile.index.values)
                    if len(self.df_npc) > 0 and 'DOX.value' in self.df_npc.columns:
                        ax3.plot(self.df_npc['DOX.value'], -self.df_npc['DEPTH.value'], '-r', label='binned data')     
                else:
                    ax3.text(0.5, 0.5, 'No O₂ Data', ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, 'No O₂ Data', ha='center', va='center', transform=ax3.transAxes)
            
            ax3.grid()
        except:
            pass
        
        # Chlorophyll profile (if available)
        try:
            ax4.set_title('Chlorophyll concentration')
            ax4.set_xlabel('ug/l')
            ax4.set_ylabel('Depth in m')
            ax4.grid()
            
            if 'chlorophyll' in df_profile.columns:
                chl_data = df_profile['chlorophyll'].dropna()
                if len(chl_data) > 0:
                    ax4.scatter(df_profile['chlorophyll'], -df_profile['depth'], 
                               s=10, c=df_profile.index.values)
                    if len(self.df_npc) > 0 and 'ChlA_SENS.value' in self.df_npc.columns:
                        ax4.plot(self.df_npc['ChlA_SENS.value'], -self.df_npc['DEPTH.value'], '-r', label='binned data')   
                else:
                    ax4.text(0.5, 0.5, 'No Chl Data', ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'No Chl Data', ha='center', va='center', transform=ax4.transAxes)
        except:
            pass
        
        # Time series plot at bottom (spanning full width)
        ax5.plot(df_profile['timestamp'], -df_profile['depth'], '-k')
        ax5.plot(df_profile.loc[ix_down_profile, 'timestamp'], -df_profile.loc[ix_down_profile, 'depth'], '.r')
        
        ax5.grid()
        ax5.set_ylabel('Depth in m')
        ax5.set_title(f'{station_key} - Profile (Click and drag to select span)')
        ax5.set_xlim([df_profile['timestamp'].min(), df_profile['timestamp'].max()])
        ax5.tick_params(axis='x', rotation=45)
        
        # Define the callback function for span selection
        def onselect(xmin, xmax):
            """Callback for span selector - finds indices in df_profile and recalculates df_npc"""
            # Convert matplotlib dates back to pandas timestamps
            xmin_time = mdates.num2date(xmin)
            xmax_time = mdates.num2date(xmax)
            
            # Convert to pandas timestamps
            xmin_time = pd.Timestamp(xmin_time)
            xmax_time = pd.Timestamp(xmax_time)
            
            # Handle timezone - make both timestamps comparable
            df_timestamps = self.current_df_profile['timestamp']
            
            # Check if dataframe timestamps are timezone-aware
            if df_timestamps.dt.tz is not None:
                # If df has timezone, localize our timestamps to match
                if xmin_time.tz is None:
                    xmin_time = xmin_time.tz_localize(df_timestamps.dt.tz)
                    xmax_time = xmax_time.tz_localize(df_timestamps.dt.tz)
                else:
                    xmin_time = xmin_time.tz_convert(df_timestamps.dt.tz)
                    xmax_time = xmax_time.tz_convert(df_timestamps.dt.tz)
            else:
                # If df doesn't have timezone, remove timezone from our timestamps
                if xmin_time.tz is not None:
                    xmin_time = xmin_time.tz_localize(None)
                    xmax_time = xmax_time.tz_localize(None)
            
            # Find indices within the selected time range
            mask = (df_timestamps >= xmin_time) & (df_timestamps <= xmax_time)
            selected_indices = self.current_df_profile.index[mask]
            
            if len(selected_indices) > 0:
                start_idx = selected_indices[0]
                stop_idx = selected_indices[-1]
                
                # Update the status label with selection info
                selection_text = f"Selected span: indices {start_idx} to {stop_idx} ({len(selected_indices)} points)"
                self.status_label.setText(selection_text)
                
                # Store the selected indices for potential export
                self.selected_span_indices = selected_indices
                
                # Recalculate df_npc with selected indices
                self.df_npc,self.meta = self.calculate_df_npc(self.current_df_profile, selected_indices)
                
                # Update the plots with new binned data
                self.update_profile_plots()
                
                self.canvas.draw_idle()
                
                # Print to console for debugging
                print(f"Span selected: Start index = {start_idx}, Stop index = {stop_idx}")
                print(f"Time range: {xmin_time} to {xmax_time}")
                print(f"Depth range: {self.current_df_profile.loc[start_idx, 'depth']:.1f}m to "
                      f"{self.current_df_profile.loc[stop_idx, 'depth']:.1f}m")
                print(f"Recalculated df_npc with {len(self.df_npc)} binned points")
        
        # Create the SpanSelector
        self.span_selector = SpanSelector(
            ax5, 
            onselect, 
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='yellow'),
            interactive=True,
            drag_from_anywhere=True,
            button=1  # Left mouse button
        )
        
        # Initialize span selector position with ix_down_profile
        if ix_down_profile.any():
            down_indices = df_profile.index[ix_down_profile]
            if len(down_indices) > 0:
                # Set initial span position
                start_time = df_profile.loc[down_indices[0], 'timestamp']
                end_time = df_profile.loc[down_indices[-1], 'timestamp']
                
                # Convert to matplotlib date format
                start_mpl = mdates.date2num(start_time)
                end_mpl = mdates.date2num(end_time)
                
                # Set the span selector's initial position
                self.span_selector.extents = (start_mpl, end_mpl)
                
                # Store initial selection
                self.selected_span_indices = down_indices
                self.status_label.setText(f"Initial span: indices {down_indices[0]} to {down_indices[-1]} ({len(down_indices)} points)")
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    # Navigation methods
    def previous_profile(self):
        """Navigate to previous profile"""
        if self.current_index > 0:
            self.current_index -= 1
            self.excluded_indices.clear()  # Clear exclusions when changing profiles
            self.display_current_profile()
    
    def next_profile(self):
        """Navigate to next profile"""
        if self.current_index < len(self.station_keys) - 1:
            self.current_index += 1
            self.excluded_indices.clear()  # Clear exclusions when changing profiles
            self.display_current_profile()
    
    def toggle_lasso_mode(self):
        """Toggle lasso selection mode on/off"""
        if self.lasso_mode_button.isChecked():
            self.lasso_mode_button.setText('Disable Lasso Selection')
            self.status_label.setText("Lasso mode: Draw around points to EXCLUDE them from NPC calculation")
            # Enable lasso selectors
            if hasattr(self, 'lasso_temp') and self.lasso_temp:
                self.lasso_temp.set_active(True)
            if hasattr(self, 'lasso_sal') and self.lasso_sal:
                self.lasso_sal.set_active(True)
        else:
            self.lasso_mode_button.setText('Enable Lasso Selection')
            self.status_label.setText("Lasso mode disabled")
            # Disable lasso selectors
            if hasattr(self, 'lasso_temp') and self.lasso_temp:
                self.lasso_temp.set_active(False)
            if hasattr(self, 'lasso_sal') and self.lasso_sal:
                self.lasso_sal.set_active(False)
    
    def clear_exclusions(self):
        """Clear all excluded data points"""
        self.excluded_indices.clear()
        self.exclusion_label.setText("Excluded points: 0")
        self.status_label.setText("All exclusions cleared")
        
        # Recalculate df_npc if we have a selection
        if hasattr(self, 'selected_span_indices') and len(self.selected_span_indices) > 0:
            self.df_npc, self.meta = self.calculate_df_npc(self.current_df_profile, self.selected_span_indices)
            self.update_profile_plots()
        
        # Redraw plots to remove exclusion markers
        if hasattr(self, 'current_df_profile'):
            station_key = self.station_keys[self.current_index]
            data = self.station_matches[station_key]
            df_profile = self.df_rsk_all.loc[data['df_rsk_indices'], :].copy()
            df_profile = df_profile.reset_index(drop=True)
            
            # Recalculate downcast
            peaks, p = find_peaks(df_profile['depth'], height=df_profile['depth'].max()*0.25, width=500)
            lb = np.zeros(len(df_profile))
            k = 1
            pp_before = 0
            for pp in peaks:
                dmax = df_profile.loc[int(p['left_ips'][k-1]):int(p['right_ips'][k-1]), 'depth'].max()
                ix_down = (df_profile['depth']>0.5) & (df_profile['depth']<dmax) & (df_profile['salinity']>0) & (df_profile.index<pp) & (df_profile.index>pp_before) & (df_profile['velocity'].rolling(100,center=True).mean()>0.1)
                lb[ix_down] = k
                k = k + 1
                pp_before = pp
            
            lb_up = np.zeros(len(df_profile))
            k = 1
            ppp = np.append(p['left_bases'], len(df_profile)-1)
            for i in range(len(peaks)):
                ix_up = (df_profile['depth']>0.5) & (df_profile['salinity']>0) & (df_profile.index>peaks[i]) & (df_profile.index<ppp[i+1]) & (df_profile['velocity'].rolling(100,center=True).mean()<-0.1)
                lb_up[ix_up] = k
                k = k + 1
            
            d_old = 0 
            k = 1
            lb_updated = np.zeros(len(lb))
            for j in np.unique(lb)[1:]:
               ix1 = np.where(lb==j)[0][0] 
               ix2 = np.where(lb==j)[0][-1] 
               d_new = df_profile.loc[ix1, 'depth']
               
               if (d_old - d_new) < df_profile['depth'].max()*0.5:
                   lb_updated[lb==j] = k
               else:
                   k = k + 1
                   lb_updated[lb==j] = k
               d_old = df_profile.loc[ix2, 'depth']
               
            for j in np.unique(lb_updated)[1:]:
                lbleng = np.where(lb_updated==j)[0][-1] - np.where(lb_updated==j)[0][0]
                if lbleng < 500:
                    lb_updated[lb_updated==j] = 0
                   
            lb = lb_updated
            ix_down_profile = lb == 1
            ix_up_profile = lb_up == 1   
            
            up_indices = df_profile.index[ix_up_profile]
            if len(up_indices)>0:
                df_profile=df_profile.iloc[:up_indices[-1],:]
                ix_down_profile=ix_down_profile[:up_indices[-1]]
            
            self.plot_profile(df_profile, station_key, ix_down_profile)
    
    # Export methods
    def export_selected_span(self):
        """Export the currently selected span data to CSV"""
        if hasattr(self, 'selected_span_indices') and len(self.selected_span_indices) > 0:
            station_key = self.station_keys[self.current_index]
            selected_data = self.current_df_profile.loc[self.selected_span_indices]
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Selected Span Data", 
                f"{station_key}_selected_span.csv",
                "CSV Files (*.csv)"
            )
            
            if filename:
                selected_data.to_csv(filename, index=True)
                QMessageBox.information(self, "Export Complete", 
                                        f"Selected span data exported to:\n{filename}")
                print(f"Exported selected span to {filename}")
        else:
            QMessageBox.warning(self, "No Selection", 
                                 "Please select a span in the time series plot first.")
    
    def save_npc_file(self):
        """Save current df_npc and meta as NPC file using npc.write function"""
        if len(self.df_npc) == 0 or len(self.meta) == 0:
            QMessageBox.warning(
                self, "No Data",
                "No binned data available to save. Please select a data span first."
            )
            return
        
        self.df_npc, self.meta = self.calculate_df_npc(self.current_df_profile, self.selected_span_indices)

        station_key = self.station_keys[self.current_index]
        default_filename = f"{station_key}_binned_profile_"  +  self.meta['operation.operationComment'].strip().replace(" ", "_")  + ".npc"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save NPC File", 
            default_filename,
            "NPC Files (*.npc);;All Files (*)"
        )
        
        if filename:
            try:
                # Use the npc.write function to save the file
                npc.write(self.meta, self.df_npc, filename)
                
                QMessageBox.information(
                    self, "Save Successful",
                    f"NPC file saved successfully:\n{filename}\n\n"
                    f"Metadata fields: {len(self.meta)}\n"
                    f"Data rows: {len(self.df_npc)}"
                )
                
                print(f"Saved NPC file: {filename}")
                print(f"  - Metadata fields: {len(self.meta)}")
                print(f"  - Data rows: {len(self.df_npc)}")
                print(f"  - Data columns: {list(self.df_npc.columns)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Error saving NPC file:\n{str(e)}")
                print(f"Error saving NPC file: {e}")
        
    def upload_npc_to_aws(self):
        """Create NPC file and upload to AWS S3"""
        if len(self.df_npc) == 0 or len(self.meta) == 0:
            QMessageBox.warning(
                self, "No Data", 
                "No binned data available to upload. Please select a data span first."
            )
            return
        
        self.df_npc, self.meta = self.calculate_df_npc(self.current_df_profile, self.selected_span_indices)

        os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
        os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"
        
        aws_access_key_id = '6lpqTL2pz42cRefC1R4c'
        aws_secret_access_key = 'W0eUlbebUidLGiKXM0iQdS8WL0slhMDdirZ6kICj'
        environmentstring = 'test'
        
        s3 = boto3.resource(service_name='s3',endpoint_url='https://s3.hi.no',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
        
        try:
            # Generate filename
            station_key = self.station_keys[self.current_index]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            npc_filename = f"{station_key}_{timestamp}_binned_profile.npc"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.npc') as temp_file:
                temp_filepath = temp_file.name
                
                # Write NPC file using existing npc.write function
                npc.write(self.meta, self.df_npc, temp_filepath)
            
            # Update status
            self.status_label.setText("Uploading NPC file to AWS S3...")
            QApplication.processEvents()  # Update UI
            
            # Perform upload
            destination = 'physchem/incoming/regular_stations/' +environmentstring+ '//' + os.path.basename( temp_filepath )
             
            with open(temp_filepath, 'rb') as data:
                 data.seek(0)
                 s3.Bucket('transient-data').put_object(Key=destination, Body=data)
            
            # Clean up temporary file
            os.unlink(temp_filepath)
            
            # Success message
            success_msg = (
                f"NPC file successfully uploaded to S3!\n\n"
                f"File details:\n"
                f"- Station: {station_key}\n"
                f"- Data rows: {len(self.df_npc)}\n"
                f"- Data columns: {len(self.df_npc.columns)}\n"
                f"- Metadata fields: {len(self.meta)}"
            )
            
            QMessageBox.information(self, "Upload Successful", success_msg)
            
            self.status_label.setText(f"Successfully uploaded: {npc_filename}")
            
            # Log to console
            print(f"Uploaded NPC file to S3")
            print(f"  - Station: {station_key}")
            print(f"  - Data rows: {len(self.df_npc)}")
            print(f"  - Metadata fields: {len(self.meta)}")
            
        except ClientError as e:
            error_msg = f"AWS S3 upload failed:\n{e.response['Error']['Message']}"
            QMessageBox.critical(self, "Upload Failed", error_msg)
            self.status_label.setText("Upload failed")
            print(f"S3 upload error: {e}")
            
        except Exception as e:
            error_msg = f"Upload failed due to unexpected error:\n{str(e)}"
            QMessageBox.critical(self, "Upload Failed", error_msg)
            self.status_label.setText("Upload failed")
            print(f"Upload error: {e}")
        


def launch_profile_browser(station_matches=None, df_rsk_all=None):
    """
    Launch the CTD Profile Browser GUI - Fully Automated Version with Parameter Selection
    
    Parameters (optional):
    station_matches: dict - Pre-loaded station matches (optional)
    df_rsk_all: pandas.DataFrame - Pre-loaded CTD data (optional)
    
    If no parameters provided, the GUI starts with file selection interface.
    
    Automated Features:
    - ✓ Automatically extracts date range from RSK files
    - ✓ Automatically fetches cruise data from toktlogger API
    - ✓ Automatically matches cruise based on date overlap
    - ✓ Automatically fills cruise parameters (vessel, platform, etc.)
    - ✓ Automatically processes data after file selection (no button needed!)
    - ✓ Interactive zoomable map with coastlines (cartopy)
    - ✓ Automatically detects and corrects profile start times
    - ✓ Lasso selection tool for excluding bad data points
    - ✓ Auto-detect O₂ and Chl parameters with selective export
    
    Workflow (Just One Click!):
    1. Select RSK files → Everything happens automatically!
       - Dates extracted
       - Cruise matched
       - Parameters filled
       - Data processed
       - Stations loaded
    2. Browse profiles with interactive zoomable map
    3. Enable lasso mode → Draw around outliers to exclude them
    4. Select which parameters to include in export (O₂/Chl auto-detected)
    5. Export or upload to PhysChem
    
    Parameter Selection:
    - O₂ and Chl availability auto-detected from profile data
    - Checkboxes auto-enabled/disabled based on availability
    - Auto-checked when available, can be unchecked to exclude
    - df_npc calculation respects checkbox states
    
    Requirements:
    - PyQt5
    - matplotlib  
    - pandas, numpy
    - pyrsktools (for processing RSK files)
    - requests (for API calls)
    - boto3 (for AWS S3 uploads)
    - cartopy (for coastlines - optional, will fallback to simple map)
    
    Install packages with:
    pip install PyQt5 matplotlib pandas numpy pyrsktools requests boto3 cartopy
    """
    app = QApplication(sys.argv)
    browser = CTDProfileBrowser(station_matches, df_rsk_all)
    browser.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    # Launch the browser without pre-loaded data
    # User will select files through the GUI
    launch_profile_browser()