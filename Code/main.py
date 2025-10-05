# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import math
from typing import Optional
import os

# === CONFIG: set file paths (change names/paths if you prefer) ===
DATA_DIR = "data"
DATA_FILES = {
    "temperature": os.path.join(DATA_DIR, "temperature.xlsx"),
    "rainfall": os.path.join(DATA_DIR, "rainfall.xlsx"),
    "soilph": os.path.join(DATA_DIR, "soilph.xlsx"),
    "precipitation": os.path.join(DATA_DIR, "precipitation.xlsx"),
}

app = FastAPI(title="Farm Data API (EcoFarm)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# in-memory loaded datasets
loaded = {}

# ---------------- helpers ----------------
def detect_columns(df: pd.DataFrame):
    """
    Try to find lat, lon, date, and a numeric value column automatically.
    Returns (lat_col, lon_col, date_col, value_col) or None if not found.
    """
    cols = list(df.columns)
    lowcols = [c.lower() for c in cols]

    lat_idx = next((i for i, c in enumerate(lowcols) if "lat" in c and "plate" not in c), None)
    lon_idx = next((i for i, c in enumerate(lowcols) if "lon" in c or "lng" in c or "long" in c), None)
    date_idx = next((i for i, c in enumerate(lowcols) if "date" in c or "time" in c), None)

    val_idx = None
    # prefer columns that look like measurement names
    for i, c in enumerate(lowcols):
        if i in (lat_idx, lon_idx, date_idx):
            continue
        if any(k in c for k in ["temp", "temperature", "rain", "ph", "precip", "value", "measurement", "reading"]):
            val_idx = i
            break
    # fallback: first numeric column not lat/lon/date
    if val_idx is None:
        for i, col in enumerate(cols):
            if i in (lat_idx, lon_idx, date_idx):
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val_idx = i
                    break
            except Exception:
                continue

    lat_col = cols[lat_idx] if lat_idx is not None else None
    lon_col = cols[lon_idx] if lon_idx is not None else None
    date_col = cols[date_idx] if date_idx is not None else None
    val_col = cols[val_idx] if val_idx is not None else None
    return lat_col, lon_col, date_col, val_col

def load_dataset(path: str):
    df = pd.read_excel(path)
    lat_col, lon_col, date_col, val_col = detect_columns(df)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not detect latitude/longitude columns in {os.path.basename(path)}. Columns: {list(df.columns)}")
    # keep a safe copy and canonical columns
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df["__lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["__lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["__date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df["__value"] = pd.to_numeric(df[val_col], errors="coerce") if val_col else None
    meta = {"lat_col": lat_col, "lon_col": lon_col, "date_col": date_col, "value_col": val_col}
    return df, meta

def haversine(lat1, lon1, lat2, lon2):
    # Haversine distance in kilometers
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2 * R * math.asin(math.sqrt(a))

def reload_all():
    results = {}
    for name, path in DATA_FILES.items():
        if not os.path.exists(path):
            results[name] = {"status": "missing", "path": path}
            continue
        try:
            df, meta = load_dataset(path)
            loaded[name] = {"df": df, "meta": meta, "path": path}
            results[name] = {"status": "loaded", "rows": len(df)}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
    return results

# load on startup
@app.on_event("startup")
def startup_event():
    reload_all()

# ---------------- endpoints ----------------
@app.get("/datasets")
def list_datasets():
    return {"available": list(loaded.keys())}

@app.post("/reload")
def reload_endpoint():
    """Reload all files from disk (call this if you replace/update XLSX files)."""
    return reload_all()

@app.get("/{dataset}")
def query_dataset(
    dataset: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 200,
):
    """
    Query a dataset by bounding box, date range, or nearest point.
    Examples:
      /temperature?min_lat=10&max_lat=12&limit=50
      /rainfall?lat=10.5&lon=76.2&radius_km=50&limit=10
    """
    if dataset not in loaded:
        raise HTTPException(status_code=404, detail="Dataset not found. Call /datasets to see available datasets.")
    entry = loaded[dataset]
    if "df" not in entry:
        raise HTTPException(status_code=500, detail=f"Dataset {dataset} not loaded correctly.")
    df = entry["df"]
    q = df

    if min_lat is not None:
        q = q[q["__lat"] >= min_lat]
    if max_lat is not None:
        q = q[q["__lat"] <= max_lat]
    if min_lon is not None:
        q = q[q["__lon"] >= min_lon]
    if max_lon is not None:
        q = q[q["__lon"] <= max_lon]
    if date_from is not None:
        d1 = pd.to_datetime(date_from, errors="coerce")
        if not pd.isna(d1):
            q = q[q["__date"] >= d1]
    if date_to is not None:
        d2 = pd.to_datetime(date_to, errors="coerce")
        if not pd.isna(d2):
            q = q[q["__date"] <= d2]

    if lat is not None and lon is not None:
        q = q.copy()
        q["__dist_km"] = q.apply(lambda r: haversine(lat, lon, r["__lat"], r["__lon"]), axis=1)
        if radius_km is not None:
            q = q[q["__dist_km"] <= radius_km]
        q = q.sort_values("__dist_km")
    else:
        q = q.sort_values("__date", ascending=False, na_position="last")

    q = q.head(limit)
    out = []
    for _, row in q.iterrows():
        out.append({
            "latitude": float(row["__lat"]) if not pd.isna(row["__lat"]) else None,
            "longitude": float(row["__lon"]) if not pd.isna(row["__lon"]) else None,
            "date": row["__date"].isoformat() if not pd.isna(row["__date"]) else None,
            "value": float(row["__value"]) if (row["__value"] is not None and not pd.isna(row["__value"])) else None,
            "distance_km": float(row["__dist_km"]) if ("__dist_km" in row and not pd.isna(row.get("__dist_km"))) else None
        })
    return {"dataset": dataset, "count": len(out), "results": out}

@app.get("/combined")
def combined(lat: float, lon: float, date: Optional[str] = None, max_km: float = 100.0, date_tolerance_days: int = 1):
    """
    For a given lat & lon (and optional date), returns the nearest record from each dataset.
    Useful: get temperature + rainfall + pH at a point for the game.
    """
    results = {}
    target_date = pd.to_datetime(date, errors="coerce") if date is not None else None

    for name, entry in loaded.items():
        if "df" not in entry:
            results[name] = {"error": "not loaded"}
            continue
        df = entry["df"].copy()
        df["__dist_km"] = df.apply(lambda r: haversine(lat, lon, r["__lat"], r["__lon"]), axis=1)
        df = df.sort_values("__dist_km")
        chosen = None
        if target_date is not None and "__date" in df.columns:
            window = df[(df["__date"] >= (target_date - pd.Timedelta(days=date_tolerance_days))) &
                        (df["__date"] <= (target_date + pd.Timedelta(days=date_tolerance_days)))]
            if not window.empty:
                chosen = window.iloc[0]
        if chosen is None:
            if df.empty:
                results[name] = None
                continue
            chosen = df.iloc[0]
        if chosen["__dist_km"] > max_km:
            results[name] = None
            continue
        results[name] = {
            "value": float(chosen["__value"]) if (chosen["__value"] is not None and not pd.isna(chosen["__value"])) else None,
            "date": chosen["__date"].isoformat() if not pd.isna(chosen["__date"]) else None,
            "latitude": float(chosen["__lat"]),
            "longitude": float(chosen["__lon"]),
            "distance_km": float(chosen["__dist_km"]),
            "source_path": entry.get("path")
        }

    return {"lat": lat, "lon": lon, "date": date, "results": results}
