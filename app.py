"""
Synthetix - Forest & Flood Intelligence Web Platform (Round 3)

Start (one command):   python app.py
Then open:              http://127.0.0.1:8000   (opens automatically)

Backend : FastAPI (JSON API under /api/...)   Frontend: static/index.html + app.js + Plotly (local file)
Models  : models/forest_xgb.joblib (Round 2, unchanged)  |  models/flood_ts_model.joblib (Round 1, unchanged)
Data    : paths in config.json  (forest = folder of .pkl patches, flood = CSV file or folder of CSVs)
No retraining happens anywhere in this app: the saved checkpoints are only loaded and used to predict.
"""
import base64, hashlib, io, json, pickle, sys, threading, time, webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
from forest_seg import FEATURES, district as patch_district, features, nodata_mask  # noqa: E402
from flood_model import add_features  # noqa: E402

CFG = {"forest_data": "data/forest", "flood_data": "data/flood", "port": 8000, "flood_weight": 0.6,
       "flood_rate_for_max_risk": 1.0}
if (ROOT / "config.json").exists():
    CFG.update(json.loads((ROOT / "config.json").read_text()))


def path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ models (load only, never retrain)
FOREST_FILE, FLOOD_FILE = ROOT / "models" / "forest_xgb.joblib", ROOT / "models" / "flood_ts_model.joblib"
forest_clf = joblib.load(FOREST_FILE)["model"]
FOREST_THR = float(getattr(forest_clf, "forest_threshold", 0.5))
try:                                   # Round 1 checkpoint (pickle)
    flood_ck = joblib.load(FLOOD_FILE)
    flood_clf, FLOOD_THR, FLOOD_FEATS = flood_ck["model"], float(flood_ck["threshold"]), flood_ck["features"]
except Exception as e:                 # same trees saved in XGBoost's portable JSON format (works on any xgboost version)
    from xgboost import XGBClassifier
    print(f"Flood pickle not readable with this xgboost version ({e}); loading the portable JSON copy of the same model.")
    FLOOD_FILE = ROOT / "models" / "flood_ts_model.json"
    flood_clf = XGBClassifier(); flood_clf.load_model(FLOOD_FILE)
    meta = json.loads((ROOT / "models" / "flood_ts_model_meta.json").read_text())
    FLOOD_THR, FLOOD_FEATS = float(meta["threshold"]), meta["features"]
MODEL_INFO = {"forest": {"file": "models/forest_xgb.joblib", "sha256": sha(FOREST_FILE), "threshold": FOREST_THR},
              "flood": {"file": "models/" + FLOOD_FILE.name, "sha256": sha(FLOOD_FILE), "threshold": FLOOD_THR}}

STATE = {"forest": {}, "forest_status": "waiting", "flood": None, "flood_status": "waiting",
         "flood_missing": {}, "errors": []}
LOCK = threading.Lock()


def norm_district(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


# ------------------------------------------------------------------ forest
def read_patch(raw, name):
    """Accept pickle (image, label) / pickle image / .npy image. Returns image HxWx11, label or None."""
    if name.endswith(".npy"):
        obj = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        obj = pickle.loads(raw, encoding="latin1")
    if isinstance(obj, dict):
        x = obj.get("image", obj.get("x")); y = obj.get("label", obj.get("y"))
    elif isinstance(obj, (tuple, list)):
        x, y = obj[0], (obj[1] if len(obj) > 1 else None)
    else:
        x, y = obj, None
    x = np.asarray(x, np.float32)
    if x.ndim == 3 and x.shape[0] in (10, 11, 12) and x.shape[-1] not in (10, 11, 12):
        x = np.moveaxis(x, 0, -1)
    if x.ndim != 3 or x.shape[-1] < 11:
        raise ValueError(f"expected an image of shape (H, W, 11), got {x.shape}")
    x = np.nan_to_num(x[..., :11])
    y = None if y is None else np.asarray(y).astype(np.uint8)
    return x, y


def predict_forest(x, y=None):
    nd = nodata_mask(x) if y is None else (nodata_mask(x) | (y == 0))
    prob = forest_clf.predict_proba(features(x).reshape(-1, len(FEATURES)))[:, 1].reshape(x.shape[:2])
    pred = np.where(prob >= FOREST_THR, 2, 1).astype(np.uint8)
    pred[nd] = 0
    return pred, prob


UNCERTAIN_BAND = 0.10   # pixels whose forest probability is within +/-0.10 of the threshold are "uncertain"


def patch_stats(name, dist, pred, y, prob=None):
    n = pred.size; nd = int((pred == 0).sum()); f = int((pred == 2).sum()); nf = int((pred == 1).sum())
    s = {"name": name, "district": dist, "pixels": n, "nodata_pixels": nd, "nodata_percent": round(100 * nd / n, 2),
         "forest_pixels": f, "nonforest_pixels": nf,
         "fc_predicted": round(100 * f / (f + nf), 2) if f + nf else None, "has_label": y is not None}
    if prob is not None and f + nf:
        pv = prob[pred > 0]
        s["uncertain_percent"] = round(100 * float((np.abs(pv - FOREST_THR) < UNCERTAIN_BAND).mean()), 2)
        s["mean_confidence"] = round(float(np.abs(pv - FOREST_THR).mean() / max(FOREST_THR, 1 - FOREST_THR)), 3)
    if y is not None:
        v = (y > 0) & (pred > 0)
        rf = int(((y == 2) & v).sum()); rn = int(((y == 1) & v).sum())
        s["fc_reference"] = round(100 * rf / (rf + rn), 2) if rf + rn else None
        cm = np.bincount((y[v].astype(int) - 1) * 2 + (pred[v].astype(int) - 1), minlength=4).reshape(2, 2)
        s["cm"] = cm.tolist()
        s["accuracy"] = round(float(np.trace(cm) / max(cm.sum(), 1)), 4)
    return s


def load_forest():
    src = path(CFG["forest_data"])
    files = sorted({p.name: p for p in list(src.rglob("*.pkl")) + list(src.rglob("*.npy"))}.values(),
                   key=lambda p: p.name) if src.exists() else []
    STATE["forest_status"] = f"running 0/{len(files)}" if files else f"no patches found in {src}"
    for i, p in enumerate(files, 1):
        try:
            x, y = read_patch(p.read_bytes(), p.name)
            pred, prob = predict_forest(x, y)
            with LOCK:
                STATE["forest"][p.stem] = {"stats": patch_stats(p.stem, patch_district(p), pred, y, prob),
                                           "pred": pred, "file": str(p)}
        except Exception as e:
            STATE["errors"].append(f"forest {p.name}: {e}")
        STATE["forest_status"] = f"running {i}/{len(files)}"
    if files:
        STATE["forest_status"] = f"ready ({len(STATE['forest'])} patches)"


def forest_districts():
    rows = {}
    for v in STATE["forest"].values():
        s = v["stats"]; r = rows.setdefault(s["district"], {"district": s["district"], "patches": 0, "forest": 0,
                                                           "nonforest": 0, "nodata": 0, "ref_f": 0, "ref_n": 0, "cm": np.zeros((2, 2), int)})
        r.setdefault("fcs", []); r.setdefault("unc", [])
        if s["fc_predicted"] is not None: r["fcs"].append(s["fc_predicted"])
        if s.get("uncertain_percent") is not None: r["unc"].append(s["uncertain_percent"])
        r["patches"] += 1; r["forest"] += s["forest_pixels"]; r["nonforest"] += s["nonforest_pixels"]; r["nodata"] += s["nodata_pixels"]
        if s.get("cm"):
            cm = np.array(s["cm"]); r["cm"] += cm; r["ref_f"] += int(cm[1].sum()); r["ref_n"] += int(cm[0].sum())
    out = []
    for r in sorted(rows.values(), key=lambda r: r["district"]):
        tot = r["forest"] + r["nonforest"]; ref = r["ref_f"] + r["ref_n"]
        out.append({"district": r["district"], "patches": r["patches"],
                    "fc_predicted": round(100 * r["forest"] / tot, 2) if tot else None,
                    "fc_reference": round(100 * r["ref_f"] / ref, 2) if ref else None,
                    "nodata_percent": round(100 * r["nodata"] / max(r["patches"] * 65536, 1), 2),
                    "macro_f1": macro_f1(r["cm"]) if ref else None, **spread(r.get("fcs", [])),
                    "uncertain_percent": round(float(np.mean(r["unc"])), 2) if r.get("unc") else None,
                    "fc_bias": round(100 * r["forest"] / tot - 100 * r["ref_f"] / ref, 2) if tot and ref else None})
    return out


def spread(v):
    """Patch-to-patch spread of forest cover in a district, with a 95% confidence interval of the mean."""
    if not v:
        return {"fc_patch_mean": None, "fc_patch_median": None, "fc_patch_std": None, "fc_ci95_low": None, "fc_ci95_high": None}
    a = np.asarray(v, float); m = a.mean(); sd = a.std(ddof=1) if len(a) > 1 else 0.0
    h = 1.96 * sd / np.sqrt(len(a)) if len(a) > 1 else None
    return {"fc_patch_mean": round(m, 2), "fc_patch_median": round(float(np.median(a)), 2), "fc_patch_std": round(sd, 2),
            "fc_ci95_low": round(max(m - h, 0), 2) if h is not None else None, "fc_ci95_high": round(min(m + h, 100), 2) if h is not None else None}


def wilson(k, n, z=1.96):
    """95% Wilson interval for a rate k/n (works well for rare events like floods)."""
    if n == 0:
        return None, None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - w) / d, 100 * (c + w) / d


def macro_f1(cm):
    cm = np.asarray(cm, float); f = []
    for i in range(2):
        tp = cm[i, i]; p = tp / max(cm[:, i].sum(), 1); r = tp / max(cm[i].sum(), 1)
        f.append(2 * p * r / max(p + r, 1e-12))
    return round(float(np.mean(f)), 4)


def class_metrics(cm, names):
    cm = np.asarray(cm, float); res = {}
    for i, n in enumerate(names):
        tp = cm[i, i]; p = tp / max(cm[:, i].sum(), 1); r = tp / max(cm[i].sum(), 1)
        res[n] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-12), 4),
                  "iou": round(tp / max(cm[i].sum() + cm[:, i].sum() - tp, 1), 4), "support": int(cm[i].sum())}
    return {"per_class": res, "accuracy": round(float(np.trace(cm) / max(cm.sum(), 1)), 4),
            "macro_f1": round(float(np.mean([res[n]["f1"] for n in names])), 4),
            "mIoU": round(float(np.mean([res[n]["iou"] for n in names])), 4),
            "confusion_matrix": cm.astype(int).tolist(), "labels": names}


PALETTE = np.array([[110, 118, 114], [214, 190, 140], [46, 125, 79]], np.uint8)  # NoData grey, non-forest sand, forest green


def to_png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def rgb_image(x):
    rgb = x[..., [3, 2, 1]]; valid = ~nodata_mask(x)
    hi = np.percentile(rgb[valid], 98) if valid.any() else 1
    img = (np.clip(rgb / max(hi, 1e-6), 0, 1) * 255).astype(np.uint8)
    img[~valid] = [60, 60, 60]
    return img


def patch_view(x, y, pred, prob, name, dist):
    rgb = rgb_image(x); mask = PALETTE[pred]
    over = rgb.copy(); f = pred == 2
    over[f] = (0.45 * rgb[f] + 0.55 * np.array([60, 200, 110])).astype(np.uint8)
    over[pred == 0] = [110, 118, 114]
    out = {"stats": patch_stats(name, dist, pred, y, prob), "rgb": to_png(rgb), "mask": to_png(mask), "overlay": to_png(over),
           "probability": to_png((np.stack([prob] * 3, -1) * 255).astype(np.uint8))}
    if y is not None:
        out["reference"] = to_png(PALETTE[np.clip(y, 0, 2)])
    out["threshold"] = FOREST_THR
    out["prob_grid"] = grid(np.where(pred == 0, np.nan, prob))
    out["ndvi_grid"] = grid(ndvi_of(x))
    return out


def ndvi_of(x):
    v = (x[..., 4] - x[..., 3]) / (x[..., 4] + x[..., 3] + 1e-6)
    return np.where(nodata_mask(x), np.nan, v)


def grid(a, n=64):
    """Block-average to n x n for 3D surfaces; NoData stays empty (null)."""
    h, w = a.shape; a = a[: h // (h // n) * (h // n), : w // (w // n) * (w // n)]
    b = a.reshape(n, a.shape[0] // n, n, a.shape[1] // n)
    with np.errstate(all="ignore"):
        g = np.nanmean(b, axis=(1, 3))
    return [[None if not np.isfinite(v) else round(float(v), 3) for v in row] for row in g]


# ------------------------------------------------------------------ flood
def load_flood_frame(df):
    df = df.reset_index(drop=True)
    miss = {c: int(n) for c, n in df.isna().sum().items() if n > 0}
    X = add_features(df)
    for c in FLOOD_FEATS:
        if c not in X:
            X[c] = np.nan
    df["probability"] = flood_clf.predict_proba(X[FLOOD_FEATS])[:, 1]
    df["station"] = df[next(c for c in ("station", "station_id", "cell_id") if c in df)].astype(str)
    df["district"] = df[next(c for c in ("district", "district_name", "region_id", "region") if c in df)].astype(str)
    df["missing_values"] = df.isna().sum(axis=1)
    return df, miss


def load_flood():
    src = path(CFG["flood_data"])
    files = [src] if src.is_file() else sorted(src.rglob("*.csv")) if src.exists() else []
    files = [f for f in files if not any(k in f.name.lower() for k in ("pred", "submission"))]
    if not files:
        STATE["flood_status"] = f"no CSV found in {src}"; return
    STATE["flood_status"] = "running"
    try:
        df, miss = load_flood_frame(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
        STATE["flood"], STATE["flood_missing"] = df, miss
        STATE["flood_status"] = f"ready ({len(df):,} rows, {df['station'].nunique():,} stations)"
    except Exception as e:
        STATE["flood_status"] = f"error: {e}"; STATE["errors"].append(f"flood: {e}")


def flood_view(threshold=None, district=None, station=None, start=None, end=None):
    df = STATE["flood"]
    if df is None:
        raise HTTPException(503, "Flood data not loaded yet")
    thr = FLOOD_THR if threshold is None else float(threshold)
    m = pd.Series(True, index=df.index)
    if district: m &= df["district"] == district
    if station: m &= df["station"] == station
    if start: m &= df["date"] >= start
    if end: m &= df["date"] <= end
    d = df[m].copy(); d["pred"] = (d["probability"] >= thr).astype(int)
    return d, thr


def district_flood(d):
    g = d.groupby("district")
    out = pd.DataFrame({"stations": g["station"].nunique(), "rows": g.size(),
                        "flood_predictions": g["pred"].sum(), "flood_rate_percent": 100 * g["pred"].mean(),
                        "mean_probability": g["probability"].mean(), "missing_values": g["missing_values"].sum()})
    if "target" in d:
        out["actual_floods"] = g["target"].sum()
    ci = [wilson(int(k), int(n)) for k, n in zip(out["flood_predictions"], out["rows"])]
    out["rate_ci95_low"] = [c[0] for c in ci]; out["rate_ci95_high"] = [c[1] for c in ci]
    return out.reset_index()


# ------------------------------------------------------------------ insights / risk
def insights(threshold=None, weight=None):
    fd = {norm_district(r["district"]): r for r in forest_districts()}
    fl = {}
    if STATE["flood"] is not None:
        d, _ = flood_view(threshold)
        for r in district_flood(d).to_dict("records"):
            fl[norm_district(r["district"])] = r
    ref = float(CFG["flood_rate_for_max_risk"])
    w = float(CFG["flood_weight"] if weight is None else min(max(float(weight), 0.0), 1.0)); rows = []
    for k in sorted(set(fd) | set(fl)):
        f, w_ = fd.get(k), fl.get(k)
        flood_exp = min(1.0, w_["flood_rate_percent"] / ref) if w_ else None
        deficit = (1 - f["fc_predicted"] / 100) if f and f["fc_predicted"] is not None else None
        if flood_exp is not None and deficit is not None:
            score, basis = 100 * (w * flood_exp + (1 - w) * deficit), "flood + forest"
        elif flood_exp is not None:
            score, basis = 100 * flood_exp, "flood only (no forest data)"
        else:
            score, basis = 100 * deficit, "forest only (no flood data)"
        rows.append({"district": k, "fc_predicted": f["fc_predicted"] if f else None,
                     "forest_patches": f["patches"] if f else 0,
                     "flood_rate_percent": round(w_["flood_rate_percent"], 3) if w_ else None,
                     "flood_predictions": int(w_["flood_predictions"]) if w_ else None,
                     "flood_exposure": round(flood_exp, 3) if flood_exp is not None else None,
                     "forest_deficit": round(deficit, 3) if deficit is not None else None,
                     "risk_score": round(score, 1), "basis": basis,
                     "risk_level": "High" if score >= 60 else "Medium" if score >= 35 else "Low"})
    rows.sort(key=lambda r: -r["risk_score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def risk_text(w=None):
    w = CFG["flood_weight"] if w is None else w
    return RISK_TEXT.replace("{FW}", f"{100 * w:.0f}").replace("{GW}", f"{100 - 100 * w:.0f}")


RISK_TEXT = ("Risk score (0-100) = {FW}% x Flood exposure + "
             "{GW}% x Forest deficit. Flood exposure = the district's share of "
             f"cell-days predicted as flood, divided by {CFG['flood_rate_for_max_risk']}% and capped at 1 (a district where "
             f"{CFG['flood_rate_for_max_risk']}% or more of cell-days flood gets full exposure). Forest deficit = 1 - predicted "
             "forest cover (FC%/100): less forest means more runoff and erosion. If a district has only one of the two, "
             "that part alone is used and the row is marked. Levels: High >= 60, Medium 35-60, Low < 35.")

# ------------------------------------------------------------------ API
app = FastAPI(title="Synthetix Forest & Flood Intelligence API")


def clean(o):
    """Make results JSON-safe: numpy types -> Python, NaN/inf (missing values) -> null (shown as NoData)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


@app.get("/api/status")
def status():
    return {"forest": STATE["forest_status"], "flood": STATE["flood_status"], "errors": STATE["errors"][-10:],
            "models": MODEL_INFO, "config": {k: str(v) for k, v in CFG.items()}}


@app.get("/api/overview")
def overview():
    fdist = forest_districts(); fl = STATE["flood"]
    valid = [r for r in fdist if r["fc_predicted"] is not None]
    tot_f = sum(v["stats"]["forest_pixels"] for v in STATE["forest"].values())
    tot_n = sum(v["stats"]["nonforest_pixels"] for v in STATE["forest"].values())
    cards = {"forest_cover_percent": round(100 * tot_f / (tot_f + tot_n), 2) if tot_f + tot_n else None,
             "forest_patches": len(STATE["forest"]), "forest_districts": len(valid),
             "flood_events_predicted": int((fl["probability"] >= FLOOD_THR).sum()) if fl is not None else None,
             "flood_stations": int(fl["station"].nunique()) if fl is not None else None,
             "flood_districts": int(fl["district"].nunique()) if fl is not None else None,
             "flood_dates": int(fl["date"].nunique()) if fl is not None else None}
    cards["districts_covered"] = len({norm_district(r["district"]) for r in fdist} |
                                     ({norm_district(x) for x in fl["district"].unique()} if fl is not None else set()))
    return clean({"cards": cards, "status": status(), "top_risk": insights()[:5]})


@app.get("/api/forest/patches")
def forest_patches(district: str | None = None):
    rows = [v["stats"] for v in STATE["forest"].values() if not district or v["stats"]["district"] == district]
    return clean({"patches": sorted(rows, key=lambda r: r["name"]), "status": STATE["forest_status"]})


@app.get("/api/forest/patch/{name}")
def forest_patch(name: str):
    v = STATE["forest"].get(name)
    if not v:
        raise HTTPException(404, f"patch {name} not found")
    x, y = read_patch(Path(v["file"]).read_bytes(), v["file"])
    pred, prob = predict_forest(x, y)
    return clean(patch_view(x, y, pred, prob, name, v["stats"]["district"]))


@app.post("/api/forest/upload")
async def forest_upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        x, y = read_patch(raw, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not read {file.filename}: {e}")
    pred, prob = predict_forest(x, y)
    stem = Path(file.filename).stem; dist = patch_district(Path(file.filename)) if "_" in stem else "uploaded"
    up = ROOT / "data" / "uploads"; up.mkdir(parents=True, exist_ok=True); (up / file.filename).write_bytes(raw)
    with LOCK:
        STATE["forest"][stem] = {"stats": patch_stats(stem, dist, pred, y, prob), "pred": pred, "file": str(up / file.filename)}
    return clean(patch_view(x, y, pred, prob, stem, dist))


@app.get("/api/hero")
def hero():
    """A real patch for the 3D terrain on the Home page: the patch with the most predicted forest."""
    if not STATE["forest"]:
        return {"name": None}
    name, v = max(STATE["forest"].items(), key=lambda kv: kv[1]["stats"]["forest_pixels"])
    x, y = read_patch(Path(v["file"]).read_bytes(), v["file"])
    pred = v["pred"]
    return clean({"name": name, "district": v["stats"]["district"], "ndvi": grid(ndvi_of(x)),
                  "forest": grid(np.where(pred == 0, np.nan, (pred == 2).astype(float)))})


@app.get("/api/flood/map")
def flood_map(threshold: float | None = None, district: str | None = None, date: str | None = None, limit: int = 4000):
    """Stations as 3D points (longitude, latitude, flood probability) for one date: all predicted floods + a sample of the rest."""
    df = STATE["flood"]
    if df is None:
        raise HTTPException(503, "Flood data not loaded yet")
    date = date or df["date"].max()
    d, thr = flood_view(threshold, district, None, date, date)
    fl = d[d["pred"] == 1]; rest = d[d["pred"] == 0]
    rest = rest.sample(min(len(rest), max(limit - len(fl), 0)), random_state=0)
    pts = pd.concat([rest, fl])
    return clean({"date": date, "threshold": thr, "n_total": int(len(d)), "n_flood": int(len(fl)),
                  "lon": pts["longitude"].round(4).tolist(), "lat": pts["latitude"].round(4).tolist(),
                  "prob": pts["probability"].round(4).tolist(), "pred": pts["pred"].tolist(), "station": pts["station"].tolist()})


@app.get("/api/forest/districts")
def forest_district_api():
    return clean({"districts": forest_districts(), "status": STATE["forest_status"]})


@app.get("/api/flood/meta")
def flood_meta():
    df = STATE["flood"]
    if df is None:
        return {"status": STATE["flood_status"]}
    return clean({"status": STATE["flood_status"], "districts": sorted(df["district"].unique().tolist()),
                  "dates": sorted(df["date"].unique().tolist()), "stations": int(df["station"].nunique()),
                  "default_threshold": FLOOD_THR, "missing_by_column": STATE["flood_missing"],
                  "rows_with_missing": int((df["missing_values"] > 0).sum()), "has_target": "target" in df,
                  "station_examples": df.groupby("station")["probability"].max().nlargest(30).index.tolist()})


@app.get("/api/flood/timeseries")
def flood_timeseries(threshold: float | None = None, district: str | None = None, station: str | None = None,
                     start: str | None = None, end: str | None = None):
    d, thr = flood_view(threshold, district, station, start, end)
    if d.empty:
        return {"threshold": thr, "series": [], "message": "No rows match these filters"}
    g = d.groupby("date")
    s = pd.DataFrame({"mean_probability": g["probability"].mean(), "max_probability": g["probability"].max(),
                      "predicted_floods": g["pred"].sum(), "rows": g.size(), "missing_values": g["missing_values"].sum()})
    if "target" in d:
        s["actual_floods"] = g["target"].sum()
    rain = next((c for c in ("rain_1d", "rain_7d", "rainfall") if c in d), None)
    if rain:
        s["rain_mean"] = g[rain].mean()
    return clean({"threshold": thr, "rain_column": rain, "series": s.reset_index().round(5).to_dict("records")})


@app.get("/api/flood/stations")
def flood_stations(threshold: float | None = None, district: str | None = None, start: str | None = None,
                   end: str | None = None, top: int = 25):
    d, thr = flood_view(threshold, district, None, start, end)
    g = d.groupby(["station", "district"])
    s = pd.DataFrame({"days": g.size(), "flood_days_predicted": g["pred"].sum(), "max_probability": g["probability"].max(),
                      "mean_probability": g["probability"].mean(), "missing_values": g["missing_values"].sum()})
    if "target" in d:
        s["flood_days_actual"] = g["target"].sum()
    s = s.reset_index().sort_values(["flood_days_predicted", "max_probability"], ascending=False)
    return clean({"threshold": thr, "total_stations": int(len(s)), "stations_with_flood": int((s["flood_days_predicted"] > 0).sum()),
                  "top": s.head(top).round(4).to_dict("records")})


@app.get("/api/flood/districts")
def flood_districts(threshold: float | None = None, start: str | None = None, end: str | None = None):
    d, thr = flood_view(threshold, None, None, start, end)
    return clean({"threshold": thr, "districts": district_flood(d).round(4).to_dict("records")})


@app.post("/api/flood/upload")
async def flood_upload(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        need = {"date", "latitude", "longitude", "water_occurrence", "hand", "rain_7d"}
        if need - set(df.columns):
            raise ValueError(f"missing columns: {sorted(need - set(df.columns))}")
        df, miss = load_flood_frame(df)
    except Exception as e:
        raise HTTPException(400, f"Could not use {file.filename}: {e}")
    STATE["flood"], STATE["flood_missing"] = df, miss
    STATE["flood_status"] = f"ready ({len(df):,} rows from upload {file.filename})"
    return flood_meta()


@app.get("/api/insights")
def insights_api(threshold: float | None = None, weight: float | None = None):
    rows = insights(threshold, weight)
    both = sum(r["basis"] == "flood + forest" for r in rows)
    return clean({"rows": rows, "risk_definition": risk_text(weight), "weight": CFG["flood_weight"] if weight is None else weight,
                  "coverage": {"districts": len(rows), "with_both": both,
                               "forest_only": sum(r["basis"].startswith("forest only") for r in rows),
                               "flood_only": sum(r["basis"].startswith("flood only") for r in rows)}})


@app.get("/api/performance")
def performance():
    out = {"models": MODEL_INFO}
    fm = ROOT / "metrics" / "forest_metrics.json"
    if fm.exists():
        m = json.loads(fm.read_text())
        out["forest_saved"] = {"source": "Round 2 held-out test set (187 patches, 20%)",
                               **class_metrics(m["confusion_matrix"]["matrix"], ["Non-forest", "Forest"]),
                               "fc_reference": m.get("FC_percent_reference"), "fc_predicted": m.get("FC_percent_predicted"),
                               "per_district": m.get("per_district"), "feature_importance": dict(list(m.get("feature_importance", {}).items())[:10])}
    lm = ROOT / "metrics" / "flood_metrics.json"
    if lm.exists():
        m = json.loads(lm.read_text())
        out["flood_saved"] = {"source": "Round 1 held-out future period (latest 20% of dates)",
                              **class_metrics(m["confusion_matrix"]["matrix"], ["No flood", "Flood"]),
                              "pr_auc": m.get("pr_auc"), "pr_auc_random": m.get("pr_auc_random"),
                              "feature_importance": dict(list(m.get("feature_importance", {}).items())[:10])}
    cm = np.zeros((2, 2), int)
    for v in STATE["forest"].values():
        if v["stats"].get("cm"):
            cm += np.array(v["stats"]["cm"])
    if cm.sum():
        out["forest_live"] = {"source": "labels in the currently loaded forest data", **class_metrics(cm, ["Non-forest", "Forest"])}
    df = STATE["flood"]
    if df is not None and "target" in df:
        p = (df["probability"] >= FLOOD_THR).astype(int)
        c = np.bincount(df["target"].astype(int) * 2 + p, minlength=4).reshape(2, 2)
        out["flood_live"] = {"source": "labels in the currently loaded flood data", **class_metrics(c, ["No flood", "Flood"])}
    return clean(out)


@app.get("/api/export/patch/{name}.npy")
def export_patch(name: str):
    v = STATE["forest"].get(name)
    if not v:
        raise HTTPException(404, f"patch {name} not found")
    b = io.BytesIO(); np.save(b, v["pred"])
    return Response(b.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename={name}.npy"})


@app.get("/api/export/{kind}")
def export(kind: str, threshold: float | None = None):
    if kind == "forest_patches.csv":
        df = pd.DataFrame([{k: v for k, v in x["stats"].items() if k != "cm"} for x in STATE["forest"].values()])
    elif kind == "forest_districts.csv":
        df = pd.DataFrame(forest_districts())
    elif kind == "flood_predictions.csv":
        d, thr = flood_view(threshold)
        cols = [c for c in ("district", "station", "latitude", "longitude", "date", "probability", "pred", "target", "missing_values") if c in d]
        df = d[cols].rename(columns={"pred": "prediction"})
    elif kind == "flood_stations.csv":
        df = pd.DataFrame(flood_stations(threshold, top=10**9)["top"])
    elif kind == "flood_districts.csv":
        df = pd.DataFrame(flood_districts(threshold)["districts"])
    elif kind == "district_risk.csv":
        df = pd.DataFrame(insights(threshold))
    elif kind == "forest_masks.zip":
        import zipfile
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
            for name, v in STATE["forest"].items():
                nb = io.BytesIO(); np.save(nb, v["pred"]); z.writestr(f"{v['stats']['district']}/{name}.npy", nb.getvalue())
        return Response(b.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": "attachment; filename=forest_masks.zip"})
    elif kind == "summary_report.html":
        return HTMLResponse(report_html(threshold), headers={"Content-Disposition": "attachment; filename=summary_report.html"})
    else:
        raise HTTPException(404, "unknown export")
    return Response(df.to_csv(index=False).replace(",nan,", ",NoData,"), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={kind}"})


def report_html(threshold=None):
    ov = overview()["cards"]; risk = insights(threshold); perf = performance()
    t = lambda rows, cols: "<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{'NoData' if r.get(c) is None else r.get(c)}</td>" for c in cols) + "</tr>" for r in rows) + "</table>"
    fs, ls = perf.get("forest_saved", {}), perf.get("flood_saved", {})
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Synthetix summary report</title>
<style>body{{font-family:Segoe UI,Arial;margin:32px;color:#1d2a24}}table{{border-collapse:collapse;margin:8px 0 20px}}
td,th{{border:1px solid #c9d3cc;padding:4px 8px;font-size:13px}}th{{background:#e8efe9}}h1{{color:#1b6e34}}</style></head><body>
<h1>Synthetix - Forest &amp; Flood Intelligence summary</h1><p>Generated {time.strftime('%Y-%m-%d %H:%M')}</p>
<h2>Key indicators</h2>{t([ov], list(ov))}
<h2>District risk ranking</h2><p>{risk_text()}</p>
{t(risk, ['rank', 'district', 'risk_score', 'risk_level', 'fc_predicted', 'flood_rate_percent', 'basis'])}
<h2>Forest districts</h2>{t(forest_districts(), ['district', 'patches', 'fc_predicted', 'fc_reference', 'nodata_percent', 'macro_f1'])}
<h2>Model performance</h2><p>Forest (Round 2 test): accuracy {fs.get('accuracy')}, macro-F1 {fs.get('macro_f1')}, mIoU {fs.get('mIoU')}.
Flood (Round 1 held-out period): accuracy {ls.get('accuracy')}, macro-F1 {ls.get('macro_f1')}, PR-AUC {ls.get('pr_auc')}.</p>
<p>Models used unchanged: forest SHA256 {MODEL_INFO['forest']['sha256'][:16]}..., flood SHA256 {MODEL_INFO['flood']['sha256'][:16]}...</p>
</body></html>"""


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=load_forest, daemon=True).start()
    threading.Thread(target=load_flood, daemon=True).start()
    url = f"http://127.0.0.1:{CFG['port']}"
    print(f"\n  Synthetix platform starting at {url}  (Ctrl+C to stop)\n")
    threading.Timer(2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=int(CFG["port"]), log_level="info")
