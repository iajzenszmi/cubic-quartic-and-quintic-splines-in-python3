#!/usr/bin/env python3
"""
weather_splines.py
Demo of cubic/quartic/quintic splines for weather data processing.

Features
- 1D time-series smoothing/interpolation with UnivariateSpline (k=3,4,5)
- Derivative (rate-of-change) estimate from the spline
- Resampling to regular intervals and CSV export
- Optional 2D gridded upscaling with RectBivariateSpline (k=3..5)
- Termux-friendly matplotlib plots (PNG files)

Usage examples
1) Synthetic demo (diurnal temps), cubic spline, export 10-min data and plot:
   python3 weather_splines.py --demo --order 3 --freq "10min" --out csv plot

2) Use your own CSV with 'time' and 'value' columns:
   python3 weather_splines.py --input my_temps.csv --time-col time --val-col temp \
       --order 5 --smooth 2.0 --freq "15min" --out csv plot

3) 2D gridded demo (synthetic lat-lon field) with quintic spline:
   python3 weather_splines.py --demo2d --order2d 5 --scale 3 --out plot
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Termux/headless friendly
import matplotlib.pyplot as plt

from datetime import datetime
try:
    from scipy.interpolate import UnivariateSpline, RectBivariateSpline
except ImportError as e:
    raise SystemExit("This demo requires SciPy. Try: pip install numpy scipy pandas matplotlib") from e

# ---------- Utilities ----------

def _to_float_timeindex(times: pd.Series) -> np.ndarray:
    """Convert pandas datetime Series to float seconds since first sample."""
    t = pd.to_datetime(times).astype("int64") / 1e9  # seconds since epoch
    return (t - t.iloc[0]).to_numpy(dtype=float)     # start at 0 for conditioning

def generate_demo_series(n_hours=48, step_minutes=20, noise=0.4, seed=42):
    """Synthetic diurnal temperature in °C with a front passing."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01 00:00:00", periods=int(n_hours*60/step_minutes), freq=f"{step_minutes}min")
    # base diurnal: 18 + 6*sin(2πt/24h - phase), plus a cooling front around hour 30
    hours = np.arange(len(t)) * (step_minutes/60.0)
    diurnal = 18.0 + 6.0*np.sin(2*np.pi*hours/24 - 0.8)
    front = -3.0*np.exp(-0.5*((hours-30.0)/4.0)**2)
    y = diurnal + front + rng.normal(0, noise, size=hours.shape)
    return pd.DataFrame({"time": t, "value": y})

def fit_time_spline(df, order=3, smooth=1.0, freq="10min"):
    """Fit spline to (time,value) and resample to regular grid."""
    if order not in (3,4,5):
        raise ValueError("order must be 3, 4, or 5")

    # Ensure time ascending and drop NaNs
    df = df.dropna(subset=[args.time_col, args.val_col]).sort_values(args.time_col).reset_index(drop=True)

    x = _to_float_timeindex(df[args.time_col])  # seconds since start
    y = df[args.val_col].to_numpy(dtype=float)

    # Fit smoothing spline (s controls smoothing: ~ sum((residuals)^2) target)
    spl = UnivariateSpline(x, y, k=order, s=smooth)

    # Build regular time grid at requested frequency
    t_start = pd.to_datetime(df[args.time_col].iloc[0])
    t_end   = pd.to_datetime(df[args.time_col].iloc[-1])
    grid = pd.date_range(t_start, t_end, freq=freq)
    xg = _to_float_timeindex(pd.Series(grid))

    yg = spl(xg)
    dyg_dt = spl.derivative(n=1)(xg)  # °C per second
    dyg_dthr = dyg_dt * 3600.0        # °C per hour (more intuitive)

    out = pd.DataFrame({
        "time": grid,
        "value_spline": yg,
        "rate_c_per_hour": dyg_dthr
    })
    return spl, out

def plot_time_series(original_df, resampled_df, title, png_path):
    plt.figure(figsize=(9,4.8))
    plt.plot(original_df[args.time_col], original_df[args.val_col], ".", label="observations")
    plt.plot(resampled_df["time"], resampled_df["value_spline"], "-", label="spline")
    plt.xlabel("time"); plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()

def demo_2d_field(nlat=25, nlon=40, seed=0):
    """Synthetic 2D temperature field on (lat,lon) with fronts and diurnal-like gradient."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-40, -10, nlat)       # e.g., Australia lat band
    lon = np.linspace(112, 154, nlon)       # longitudes AU
    LON, LAT = np.meshgrid(lon, lat)
    field = 22 + 8*np.cos((LAT+25)/7) - 3*np.sin((LON-133)/6)
    # add two 'fronts'
    field += -4*np.exp(-((LAT+28)**2 + (LON-138)**2)/30)
    field +=  3*np.exp(-((LAT+20)**2 + (LON-120)**2)/60)
    field += rng.normal(0, 0.2, size=field.shape)
    return lat, lon, field

def upscale_2d(lat, lon, field, k=3, scale=2):
    """Upscale gridded field with RectBivariateSpline."""
    if k not in (3,4,5):
        raise ValueError("order2d must be 3, 4, or 5")
    # RectBivariateSpline expects strictly increasing coords
    rbs = RectBivariateSpline(lat, lon, field, kx=k, ky=k)
    lat_hi = np.linspace(lat.min(), lat.max(), len(lat)*scale)
    lon_hi = np.linspace(lon.min(), lon.max(), len(lon)*scale)
    field_hi = rbs(lat_hi, lon_hi)
    return lat_hi, lon_hi, field_hi

def plot_2d(lat, lon, field, title, png_path):
    plt.figure(figsize=(6.4,4.8))
    plt.imshow(field, origin="lower",
               extent=[lon.min(), lon.max(), lat.min(), lat.max()],
               aspect="auto")
    plt.colorbar(label="value")
    plt.xlabel("lon"); plt.ylabel("lat")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()

# ---------- Main ----------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cubic/Quartic/Quintic spline demo for weather data.")
    p.add_argument("--input", help="CSV with columns time/value (or use --demo).")
    p.add_argument("--time-col", default="time", help="Time column name (default: time).")
    p.add_argument("--val-col", default="value", help="Value column name (default: value).")
    p.add_argument("--order", type=int, default=3, choices=[3,4,5], help="Spline order for 1D (3,4,5).")
    p.add_argument("--smooth", type=float, default=1.0, help="Smoothing parameter s (increase to smooth more).")
    p.add_argument("--freq", default="10min", help="Resample frequency (e.g., 10min, 1H).")
    p.add_argument("--out", nargs="+", choices=["csv","plot","none"], default=["csv","plot"],
                   help="What to output for 1D: csv and/or plot.")
    p.add_argument("--demo", action="store_true", help="Use synthetic 1D time series data.")
    # 2D options
    p.add_argument("--demo2d", action="store_true", help="Run 2D gridded example.")
    p.add_argument("--order2d", type=int, default=3, choices=[3,4,5], help="Spline order for 2D (kx=ky).")
    p.add_argument("--scale", type=int, default=2, help="Upscale factor for 2D grid.")
    args = p.parse_args()

    # ----- 1D time-series flow -----
    if args.demo or args.input:
        if args.demo:
            df = generate_demo_series()
        else:
            df = pd.read_csv(args.input)
            # try to parse time column if not already datetime
            if not np.issubdtype(df[args.time_col].dtype, np.datetime64):
                df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")

        spl, out = fit_time_spline(df, order=args.order, smooth=args.smooth, freq=args.freq)

        # Outputs
        stem = f"spline1d_k{args.order}"
        if "csv" in args.out:
            csv_path = f"{stem}_{args.freq}.csv"
            out.to_csv(csv_path, index=False)
            print(f"[1D] wrote {csv_path} with {len(out)} rows")

        if "plot" in args.out:
            png_path = f"{stem}_{args.freq}.png"
            plot_time_series(df, out, f"1D spline k={args.order}, s={args.smooth}", png_path)
            print(f"[1D] wrote {png_path}")

        # quick console preview
        print(out.head(8).to_string(index=False))

    # ----- 2D field flow -----
    if args.demo2d:
        lat, lon, field = demo_2d_field()
        lat_hi, lon_hi, field_hi = upscale_2d(lat, lon, field, k=args.order2d, scale=args.scale)

        # Save quicklook plots
        plot_2d(lat, lon, field, "2D original field", f"spline2d_original.png")
        plot_2d(lat_hi, lon_hi, field_hi, f"2D spline upscaled k={args.order2d}", f"spline2d_k{args.order2d}_x{args.scale}.png")

        # Export arrays for downstream processing
        np.save("spline2d_lat.npy", lat_hi)
        np.save("spline2d_lon.npy", lon_hi)
        np.save("spline2d_field.npy", field_hi)

        print(f"[2D] wrote spline2d_k{args.order2d}_x{args.scale}.png and .npy arrays (lat/lon/field)")
