#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

try:
    import yaml
except ImportError as e:
    print("Missing dependency 'PyYAML'. Install with: pip install pyyaml", file=sys.stderr)
    raise

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Missing dependency 'matplotlib'. Install with: pip install matplotlib", file=sys.stderr)
    raise

try:
    from scipy.signal import butter, filtfilt, savgol_filter
except ImportError as e:
    print("Missing dependency 'scipy'. Install with: pip install scipy", file=sys.stderr)
    raise

# Enable milliseconds in hover/status bar
def _format_ms(x, pos=None):
    # Format matplotlib date number to HH:MM:SS.mmm
    try:
        dt = mdates.num2date(x)
        return dt.strftime('%H:%M:%S.%f')[:-3]
    except Exception:
        return str(x)

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot + (optionally) export calibrated MCCDAQ data using YAML configuration."
    )
    # IO
    p.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    p.add_argument("--config", dest="config", required=True, help="YAML config path")
    p.add_argument("--export", dest="export", default=None,
                   help="Optional path to write a calibrated CSV (if omitted, no CSV is written unless --export is provided)")
    p.add_argument("--keep-raw", dest="keep_raw", action="store_true",
                   help="If set, keep original raw channel columns in the exported CSV")

    # Time
    p.add_argument("--timestamp-unit", dest="ts_unit", default="ms",
                   choices=["s", "ms", "us", "ns", "none"],
                   help="Unit of 'timestamp' column (default: ms). Use 'none' to skip conversion.")
    p.add_argument("--time-col", dest="time_col", default="time",
                   help="Name of the ISO8601 column to add (default: time)")

    # Calibration
    p.add_argument("--prefix", dest="prefix", default="cal_",
                   help="Prefix for calibrated column names (default: cal_)")
    p.add_argument("--drop-uncalibrated", dest="drop_uncal", action="store_true",
                   help="If set, drop calibrated columns with missing calibration in config")
    p.add_argument("--strict", dest="strict", action="store_true",
                   help="If set, raise if a configured channel column is missing from CSV")

    # Plotting options
    p.add_argument("--no-plot", dest="no_plot", action="store_true",
                   help="Disable plotting (useful if only exporting).")
    p.add_argument("--group-by-unit", dest="group_by_unit", action="store_true",
                   help="Group plots by engineering unit (default on).")
    p.add_argument("--no-group-by-unit", dest="no_group_by_unit", action="store_true",
                   help="Do not group by unit; plot all calibrated channels on one figure.")
    p.add_argument("--figdir", dest="figdir", default=None,
                   help="If provided, save figures into this directory instead of showing.")
    p.add_argument("--fig-dpi", dest="fig_dpi", type=int, default=120,
                   help="Figure DPI for saved figures (default: 120).")
    p.add_argument("--fig-size", dest="fig_size", default="12,6",
                   help="Figure size as 'W,H' in inches (default: 12,6).")
    p.add_argument("--show", dest="show", action="store_true",
                   help="Show figures interactively (ignored if --figdir is used without a display).")

    # Data reduction (optional)
    p.add_argument("--decimate", dest="decimate", type=int, default=1,
                   help="Keep every Nth row (simple decimation) before plotting/exporting (default: 1 = no decimation).")
    p.add_argument("--resample", dest="resample", default=None,
                   help="Optional pandas resample rule (e.g., '100ms', '1S'). Requires usable timestamps. Aggregation = mean.")
    
    # Low-pass filter for kg units
    p.add_argument("--kg-filter-freq", dest="kg_filter_freq", type=float, default=5.0,
                   help="Low-pass filter cutoff frequency (Hz) for derivative data with 'kg/s' units (default: 5.0).")
    p.add_argument("--kg-filter-order", dest="kg_filter_order", type=int, default=4,
                   help="Low-pass filter order for derivative data with 'kg/s' units (default: 4).")
    p.add_argument("--sample-rate", dest="sample_rate", type=float, default=100.0,
                   help="Sample rate (Hz) for filter design (default: 100.0).")
    
    # Savitzky-Golay filter for kg units (mass data)
    p.add_argument("--kg-savgol-window", dest="kg_savgol_window", type=int, default=51,
                   help="Savitzky-Golay filter window length for mass data with 'kg' units (default: 51, must be odd).")
    p.add_argument("--kg-savgol-polyorder", dest="kg_savgol_polyorder", type=int, default=3,
                   help="Savitzky-Golay filter polynomial order for mass data with 'kg' units (default: 3).")

    return p.parse_args()


def load_config(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mccdaq = cfg.get("MCCDAQ", []) or []
    sensors = []
    for entry in mccdaq:
        if entry is None:
            continue
        sensors.append({
            "hatID": entry.get("hatID"),
            "channelID": entry.get("channelID"),
            "name": entry.get("name"),
            "unit": entry.get("unit", ""),
            "type": entry.get("type", ""),
            "calibration": entry.get("calibration", None),
        })
    return sensors


def build_column_key(hat_id, ch_id):
    return f"hat{hat_id}_ch{ch_id}"


def validate_calibration_table(cal_table):
    arr = np.asarray(cal_table, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Calibration must be a list of [raw, value] pairs")
    idx = np.argsort(arr[:, 0], kind="mergesort")
    arr = arr[idx]
    xs, first_idx = np.unique(arr[:, 0], return_index=True)
    ys = arr[first_idx, 1]
    return xs, ys


def pwl_map(raw_values: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(xs) == 0:
        return np.full_like(raw_values, np.nan, dtype=float)
    if len(xs) == 1:
        return np.full_like(raw_values, ys[0], dtype=float)
    interp_vals = np.interp(raw_values, xs, ys, left=np.nan, right=np.nan)
    left_mask = raw_values < xs[0]
    if np.any(left_mask):
        m_left = (ys[1] - ys[0]) / (xs[1] - xs[0])
        interp_vals[left_mask] = ys[0] + m_left * (raw_values[left_mask] - xs[0])
    right_mask = raw_values > xs[-1]
    if np.any(right_mask):
        m_right = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        interp_vals[right_mask] = ys[-1] + m_right * (raw_values[right_mask] - xs[-1])
    return interp_vals


def add_iso_time(df: pd.DataFrame, ts_unit: str, out_col: str) -> pd.DataFrame:
    if "timestamp" not in df.columns or ts_unit == "none":
        return df
    unit_map = {"s": "s", "ms": "ms", "us": "us", "ns": "ns"}
    unit = unit_map.get(ts_unit, "ms")
    try:
        dt = pd.to_datetime(df["timestamp"].astype("int64"), unit=unit, utc=True)
        df[out_col] = dt.dt.tz_convert("UTC").dt.tz_localize(None)  # naive UTC datetime for plotting
    except Exception as e:
        df[out_col] = pd.NaT
        print(f"Warning: timestamp to datetime conversion failed: {e}", file=sys.stderr)
    return df


def apply_savgol_filter(data: np.ndarray, window_length: int = 51, polyorder: int = 3) -> np.ndarray:
    """Apply a Savitzky-Golay filter to the data."""
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
        print(f"Warning: Window length must be odd, using {window_length}", file=sys.stderr)
    
    # Ensure window_length > polyorder
    if window_length <= polyorder:
        window_length = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
        print(f"Warning: Window length must be greater than polyorder, using {window_length}", file=sys.stderr)
    
    if len(data) < window_length:
        print(f"Warning: Data length {len(data)} too short for window {window_length}, returning unfiltered data", file=sys.stderr)
        return data
    
    # Remove NaN values for filtering
    mask = ~np.isnan(data)
    if not np.any(mask):
        return data
    
    filtered = data.copy()
    valid_data = data[mask]
    
    if len(valid_data) < window_length:
        print(f"Warning: Too few valid data points for filtering, returning unfiltered data", file=sys.stderr)
        return data
    
    try:
        filtered_valid = savgol_filter(valid_data, window_length, polyorder, mode='interp')
        filtered[mask] = filtered_valid
        
    except Exception as e:
        print(f"Warning: Savitzky-Golay filter application failed: {e}, returning unfiltered data", file=sys.stderr)
        return data
    
    return filtered


def apply_lowpass_filter(data: np.ndarray, cutoff_freq: float, sample_rate: float, order: int = 4) -> np.ndarray:
    """Apply a low-pass Butterworth filter to the data."""
    if len(data) < 2 * order:
        print(f"Warning: Data too short for filter order {order}, returning unfiltered data", file=sys.stderr)
        return data
    
    # Remove NaN values for filtering
    mask = ~np.isnan(data)
    if not np.any(mask):
        return data
    
    filtered = data.copy()
    valid_data = data[mask]
    
    if len(valid_data) < 2 * order:
        print(f"Warning: Too few valid data points for filtering, returning unfiltered data", file=sys.stderr)
        return data
    
    try:
        nyquist = sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        
        if normal_cutoff >= 1.0:
            print(f"Warning: Cutoff frequency {cutoff_freq} Hz too high for sample rate {sample_rate} Hz, returning unfiltered data", file=sys.stderr)
            return data
        
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_valid = filtfilt(b, a, valid_data)
        filtered[mask] = filtered_valid
        
    except Exception as e:
        print(f"Warning: Filter application failed: {e}, returning unfiltered data", file=sys.stderr)
        return data
    
    return filtered


def calibrate_dataframe(df: pd.DataFrame, sensors, prefix="cal_", strict=False, drop_uncal=False, 
                       kg_savgol_window=51, kg_savgol_polyorder=3):
    out_df = df.copy()
    cal_columns = []  # list of (col_name, unit)

    missing_cols = []
    missing_cal = []

    for s in sensors:
        hat = s["hatID"]
        ch = s["channelID"]
        name = s.get("name") or f"hat{hat}_ch{ch}"
        unit = s.get("unit", "")
        cal = s.get("calibration", None)
        col_key = build_column_key(hat, ch)

        if col_key not in df.columns:
            missing_cols.append(col_key)
            if strict:
                raise KeyError(f"Configured channel '{col_key}' not found in CSV.")
            else:
                continue

        out_col = f"{prefix}{name}{'[' + unit + ']' if unit else ''}"

        if not cal:
            if drop_uncal:
                continue
            else:
                data = df[col_key].astype(float).to_numpy()
                # Apply Savitzky-Golay filter for kg units
                if unit.lower() == 'kg':
                    data = apply_savgol_filter(data, kg_savgol_window, kg_savgol_polyorder)
                    print(f"Applied Savitzky-Golay filter (window={kg_savgol_window}, polyorder={kg_savgol_polyorder}) to {name} (kg units)")
                out_df[out_col] = data
                cal_columns.append((out_col, unit))
                missing_cal.append(col_key)
                continue

        try:
            xs, ys = validate_calibration_table(cal)
        except Exception as e:
            print(f"Invalid calibration for {name} ({col_key}): {e}", file=sys.stderr)
            out_df[out_col] = np.nan
            cal_columns.append((out_col, unit))
            continue

        raw_vals = df[col_key].astype(float).to_numpy()
        mapped = pwl_map(raw_vals, xs, ys)
        
        # Apply Savitzky-Golay filter for kg units after calibration
        if unit.lower() == 'kg':
            mapped = apply_savgol_filter(mapped, kg_savgol_window, kg_savgol_polyorder)
            print(f"Applied Savitzky-Golay filter (window={kg_savgol_window}, polyorder={kg_savgol_polyorder}) to {name} (kg units)")
        
        out_df[out_col] = mapped
        cal_columns.append((out_col, unit))

    if missing_cols:
        print(f"Note: Missing channel columns in CSV (skipped): {sorted(set(missing_cols))}", file=sys.stderr)
    if missing_cal and not drop_uncal:
        print(f"Note: Channels without calibration entries (passed-through raw): {sorted(set(missing_cal))}", file=sys.stderr)

    return out_df, cal_columns


def maybe_resample(df: pd.DataFrame, time_col: str, rule: str):
    if time_col not in df.columns:
        print("Resample requested but no time column available; skipping.", file=sys.stderr)
        return df
    if df[time_col].isna().all():
        print("Resample requested but time column conversion failed; skipping.", file=sys.stderr)
        return df
    # set index to time for resample
    tmp = df.set_index(time_col)
    # numeric-only means; non-numeric columns are dropped automatically
    return tmp.resample(rule).mean(numeric_only=True).reset_index()


def calculate_mass_derivatives(df: pd.DataFrame, cal_columns, time_col: str, sample_rate: float, 
                             kg_filter_freq: float = 5.0, kg_filter_order: int = 4):
    """Calculate time derivatives for mass (kg) channels using Savitzky-Golay filtered data, then apply low-pass filter to derivatives."""
    derivative_columns = []
    
    # Find mass columns
    mass_columns = [(col, unit) for col, unit in cal_columns if unit.lower() == 'kg']
    
    if not mass_columns:
        return df, derivative_columns
    
    if time_col not in df.columns or df[time_col].isna().all():
        # Fall back to sample-based derivative using sample rate
        dt = 1.0 / sample_rate
        print(f"Using sample rate {sample_rate} Hz for derivative calculation")
    else:
        dt = None  # Will use actual time differences
    
    for col, unit in mass_columns:
        if col not in df.columns:
            continue
            
        derivative_col = col.replace('[kg]', '[kg/s]') + '_derivative'
        
        # Use the filtered mass data (already filtered in calibrate_dataframe)
        filtered_mass_data = df[col]
        
        if dt is None:
            # Use actual time differences
            time_vals = pd.to_datetime(df[time_col])
            time_diff = time_vals.diff().dt.total_seconds()
            mass_diff = filtered_mass_data.diff()
            derivative = mass_diff / time_diff
        else:
            # Use constant sample rate
            derivative = filtered_mass_data.diff() / dt
        
        # Apply low-pass filter to the derivative data
        derivative_filtered = apply_lowpass_filter(derivative.to_numpy(), kg_filter_freq, sample_rate, kg_filter_order)
        print(f"Applied low-pass filter ({kg_filter_freq} Hz) to derivative of {col}")
        
        df[derivative_col] = derivative_filtered
        derivative_columns.append((derivative_col, 'kg/s'))
        
    return df, derivative_columns

def do_plots(df: pd.DataFrame, cal_columns, time_col: str, group_by_unit: bool, figdir: Path | None, fig_size=(12, 6), fig_dpi=120, show=False, derivative_columns=None):
    if not cal_columns:
        print("No calibrated columns to plot.", file=sys.stderr)
        return

    # Parse size
    if isinstance(fig_size, str):
        try:
            w, h = fig_size.split(",")
            fig_size = (float(w), float(h))
        except Exception:
            fig_size = (12, 6)

    # Combine regular and derivative columns for plotting
    all_plot_columns = cal_columns[:]
    if derivative_columns:
        all_plot_columns.extend(derivative_columns)

    # Build groups
    if group_by_unit:
        groups = {}
        for col, unit in all_plot_columns:
            key = unit if unit else "unitless"
            groups.setdefault(key, []).append(col)
        for unit, cols in groups.items():
            fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            x = df[time_col] if time_col in df.columns else df.index
            for c in cols:
                if c in df.columns:
                    ax.plot(x, df[c], label=c)
            ax.set_title(f"Calibrated signals ({unit})")
            ax.set_xlabel("Time" if time_col in df.columns else "Sample")
            ax.set_ylabel(unit if unit else "value")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best", ncols=1 if len(cols) < 8 else 2, fontsize="small")
            if figdir:
                figdir.mkdir(parents=True, exist_ok=True)
                out = figdir / f"plot_{unit.replace('/', '_')}.png"
                fig.savefig(out, bbox_inches="tight")
                print(f"Saved {out}")
                plt.close(fig)
            else:
                if show:
                    # Show milliseconds on hover
                    plt.gca().format_xdata = _format_ms
                    plt.show()
                    plt.show()
                else:
                    # If neither saving nor showing, still draw once to let caller decide
                    plt.draw()
    else:
        fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
        x = df[time_col] if time_col in df.columns else df.index
        for c, _unit in all_plot_columns:
            if c in df.columns:
                ax.plot(x, df[c], label=c)
        ax.set_title("Calibrated signals")
        ax.set_xlabel("Time" if time_col in df.columns else "Sample")
        ax.set_ylabel("value")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        if figdir:
            figdir.mkdir(parents=True, exist_ok=True)
            out = figdir / "plot_all.png"
            fig.savefig(out, bbox_inches="tight")
            print(f"Saved {out}")
            plt.close(fig)
        else:
            if show:
                # Show milliseconds on hover
                plt.gca().format_xdata = _format_ms
                plt.show()      
                plt.show()
            else:
                plt.draw()


def main():
    args = parse_args()
    in_path = Path(args.inp)
    cfg_path = Path(args.config)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")

    df = pd.read_csv(in_path)

    # Time handling
    df = add_iso_time(df, args.ts_unit, args.time_col)

    # Decimate early to reduce plotting load
    if args.decimate and args.decimate > 1:
        df = df.iloc[::args.decimate, :].reset_index(drop=True)

    sensors = load_config(cfg_path)
    # Calibrate (keeps raw columns until we decide what to export)
    df_cal, cal_columns = calibrate_dataframe(
        df, sensors,
        prefix=args.prefix,
        strict=args.strict,
        drop_uncal=args.drop_uncal,
        kg_savgol_window=args.kg_savgol_window,
        kg_savgol_polyorder=args.kg_savgol_polyorder
    )

    # Calculate derivatives for mass channels
    df_cal, derivative_columns = calculate_mass_derivatives(
        df_cal, cal_columns, args.time_col, args.sample_rate,
        kg_filter_freq=args.kg_filter_freq, kg_filter_order=args.kg_filter_order
    )

    # Resample after calibration, if requested
    if args.resample:
        df_cal = maybe_resample(df_cal, args.time_col, args.resample)

    # Plot unless disabled
    group_by_unit = True
    if args.no_group_by_unit:
        group_by_unit = False
    elif args.group_by_unit:
        group_by_unit = True

    if not args.no_plot:
        figdir = Path(args.figdir) if args.figdir else None
        do_plots(
            df_cal, cal_columns,
            time_col=args.time_col,
            group_by_unit=group_by_unit,
            figdir=figdir,
            fig_size=args.fig_size,
            fig_dpi=args.fig_dpi,
            show=args.show,
            derivative_columns=derivative_columns
        )

    # Optional export
    if args.export:
        # Decide what to export (include derivatives)
        if args.keep_raw:
            out_df = df_cal
        else:
            keep = ["timestamp"]
            if args.time_col in df_cal.columns:
                keep.append(args.time_col)
            keep += [c for c, _u in cal_columns if c in df_cal.columns]
            keep += [c for c, _u in derivative_columns if c in df_cal.columns]
            out_df = df_cal[keep]
        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Wrote calibrated CSV -> {out_path}")


if __name__ == "__main__":
    main()
