# -*- coding: utf-8 -*-
"""
RiWEx Sentinel-2 — working logic + simple batching CLI
(Revised: fixes Qt thread crash + RAM allocation errors without changing method logic)

Key changes (implementation only, not science):
- Force matplotlib "Agg" backend (avoid Qt/QThreadStorage issues).
- Use float32 where possible (same math, lower memory).
- Replace np.stack(...) on huge 3D arrays with disk-backed np.memmap.
- Aggressive cleanup (gc.collect) between big steps.
"""

import os
# ---- must be BEFORE importing matplotlib.pyplot ----
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
import random
import warnings
import urllib.request
import argparse
import gc

import ee
import geemap
import xarray as xr
import xee  # registers the 'ee' backend for xarray.open_dataset
import numpy as np
import pandas as pd
import maxflow
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import rioxarray
from pyproj import Geod
from rasterio import features  # IMPORTANT: needed for reach rasterization

warnings.filterwarnings("ignore")

# ----------------- CONFIG (defaults; override via CLI) -----------------
HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Edit these values to process a different location, or override them
    # with the command-line options documented in README.md.
    "site": "example_site",
    "latitude": -3.312,
    "longitude": -60.598,
    "output_path": os.path.join(HERE, "results_sentinel2"),

    "water_indices": ["ndwi"],
    "reach_type": ["single_reach"],
    "segment_snic": "Y",
    "lambda_values": [0.01],

    "buffer_coef": 1,
    "energy_method": "basic",
    "energy_alpha": 1.0,
    "energy_beta": 0.5,

    "jrc_threshold": 0,
    "dem_percentile": 70,
    "resolution": 10,

    # Use --project-id or set GOOGLE_CLOUD_PROJECT. Never commit credentials.
    "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
    "ee_url": "https://earthengine-highvolume.googleapis.com",
    "debug_plots": False,
}

# ----------------- UTIL -----------------
def log_time(start_time, section_name):
    elapsed = time.time() - start_time
    print(f"Section '{section_name}' took {elapsed:.2f} seconds")
    return time.time()

def initialize_earth_engine():
    if not CONFIG["project_id"]:
        raise ValueError(
            "An Earth Engine-enabled Google Cloud project is required. "
            "Pass --project-id YOUR_PROJECT_ID or set GOOGLE_CLOUD_PROJECT."
        )
    try:
        if CONFIG.get("project_id"):
            ee.Initialize(project=CONFIG["project_id"], opt_url=CONFIG.get("ee_url"))
        else:
            ee.Initialize(opt_url=CONFIG.get("ee_url")) if CONFIG.get("ee_url") else ee.Initialize()
    except Exception:
        print("[GEE] Authenticating…")
        ee.Authenticate()
        if CONFIG.get("project_id"):
            ee.Initialize(project=CONFIG["project_id"], opt_url=CONFIG.get("ee_url"))
        else:
            ee.Initialize(opt_url=CONFIG.get("ee_url")) if CONFIG.get("ee_url") else ee.Initialize()
    print("[GEE] Initialized.")

def _safe_close_all_figures():
    try:
        plt.close("all")
    except Exception:
        pass

# ----------------- REACH HELPERS -----------------
def find_reach(reach_collection, reach_id=None, lon=None, lat=None):
    start_time = time.time()
    if reach_id:
        reach = reach_collection.filter(ee.Filter.eq("reach_id", reach_id)).first()
        if reach.getInfo() is None:
            raise ValueError(f"No reach found for reach_id {reach_id}")
        return reach, start_time

    if lon is None or lat is None:
        raise ValueError("Must provide either reach_id or coordinates")

    point = ee.Geometry.Point([lon, lat])
    buffer_size, max_buffer = 1000, 10000
    while buffer_size <= max_buffer:
        print(f"Searching for reach with buffer size: {buffer_size}m")
        filtered = reach_collection.filterBounds(point.buffer(buffer_size))
        if filtered.size().getInfo() > 0:
            nearest = filtered.map(lambda f: f.set("distance_to_point", f.geometry().distance(point))) \
                              .sort("distance_to_point").first()
            reach_id_test = nearest.get("reach_id").getInfo()
            if reach_id_test:
                print(f"Found reach_id: {reach_id_test}, width: {nearest.get('width').getInfo()}")
                return nearest, start_time
        buffer_size += 1000
    raise ValueError(f"No reach found near ({lon}, {lat}) within {max_buffer}m")

def create_buffer_zone(feature, buffer_coef):
    width = ee.Number.parse(feature.get("width"))
    variance = ee.Number.parse(feature.get("width_var")).sqrt()
    buffer_size = ee.Number(buffer_coef).multiply(variance).add(width)
    return feature.geometry().buffer(buffer_size)

def calculate_reach_length(feature, reach_type, reaches):
    start_time = time.time()

    def _get_prop(feat, names):
        for n in names:
            try:
                v = feat.get(n)
                if v is not None and v.getInfo() is not None:
                    return ee.Number.parse(v)
            except Exception:
                pass
        return None

    if reach_type == "single_reach":
        length_num = _get_prop(feature, ["reach_len", "reach_length"])
        if length_num is None:
            raise ValueError("Missing reach length property (reach_len/reach_length).")
        length = length_num.divide(1000)
    else:
        up_id = ee.Number.parse(feature.get("rch_id_up"))
        dn_id = ee.Number.parse(feature.get("rch_id_dn"))
        up_river = reaches.filter(ee.Filter.eq("reach_id", up_id)).first()
        dn_river = reaches.filter(ee.Filter.eq("reach_id", dn_id)).first()
        current_len = _get_prop(feature, ["reach_len", "reach_length"])
        up_len = _get_prop(up_river, ["reach_len", "reach_length"])
        dn_len = _get_prop(dn_river, ["reach_len", "reach_length"]) if dn_river.getInfo() else ee.Number(0)
        if current_len is None or up_len is None:
            raise ValueError("Missing reach length on upstream/current reach.")
        length = current_len.add(up_len).add(dn_len).divide(1000)

    return length.getInfo(), start_time

# ----------------- MAPS -----------------
def _ee_s2_rgb(reach_buffer, start="2020-01-01", end="2023-12-31"):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(reach_buffer).filterDate(start, end)
          .sort("CLOUDY_PIXEL_PERCENTAGE").first()
          .select(["B4", "B3", "B2"]).multiply(0.0001))
    return s2.visualize(bands=["B4", "B3", "B2"], min=0, max=0.3)

def save_map_visualization(site, lon, lat, reach_buffer, reach_geometry, save_path):
    start_time = time.time()
    gauge_point = ee.Feature(ee.Geometry.Point([lon, lat]), {"name": site})
    try:
        gauge_buffer = gauge_point.geometry().buffer(100)
        zoomed_region = reach_buffer.buffer(5000)

        vis_image = (ee.Image().paint(reach_buffer, 1, 2).visualize(palette=["red"])
                     .blend(ee.Image().paint(reach_geometry, 1, 2).visualize(palette=["blue"]))
                     .blend(ee.Image().paint(gauge_buffer, 1).visualize(palette=["yellow"])))

        satellite = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                     .filterBounds(reach_buffer)
                     .filterDate("2020-01-01", "2023-12-31")
                     .sort("CLOUDY_PIXEL_PERCENTAGE")
                     .first()
                     .select(["B4", "B3", "B2"])
                     .multiply(0.0001))

        final_image = satellite.visualize(bands=["B4", "B3", "B2"], min=0, max=0.3).blend(vis_image)

        thumb_url = final_image.getThumbURL({
            "image": final_image,
            "dimensions": "800x600",
            "region": zoomed_region,
            "format": "png",
            "crs": "EPSG:4326"
        })
        map_path = os.path.join(save_path, f"{site}_map.png")
        urllib.request.urlretrieve(thumb_url, map_path)
        print(f"Map saved for {site} at {map_path}")
    except Exception as e:
        print(f"Error saving map for {site}: {e}")
    log_time(start_time, "Map Visualization")

def save_corrected_buffer_plot_sat(site, lon, lat, reach_buf_cor_gdf, reach_buffer, reach, save_path):
    try:
        fc_buf = geemap.gdf_to_ee(reach_buf_cor_gdf)
        buf_geom = fc_buf.geometry()

        gauge_point = ee.Feature(ee.Geometry.Point([lon, lat]), {"name": site})
        rgb = _ee_s2_rgb(reach_buffer)

        overlays = (ee.Image().paint(buf_geom, 1, 2).visualize(palette=["red"])
                    .blend(ee.Image().paint(ee.FeatureCollection([reach]), 1, 2).visualize(palette=["blue"]))
                    .blend(ee.Image().paint(gauge_point.geometry().buffer(100), 1).visualize(palette=["yellow"])))

        final_image = rgb.blend(overlays)
        thumb_url = final_image.getThumbURL({
            "image": final_image,
            "dimensions": "800x600",
            "region": reach_buffer.buffer(5000),
            "format": "png",
            "crs": "EPSG:4326"
        })
        out_png = os.path.join(save_path, f"{site}_buffer_corrected.png")
        urllib.request.urlretrieve(thumb_url, out_png)
        print(f"Corrected-buffer map saved for {site} at {out_png}")
    except Exception as e:
        print(f"[WARN] Could not draw corrected buffer (satellite) for {site}: {e}")

# ----------------- SENTINEL-2 / INDICES -----------------
def sentinel_cloud_mask(img, reach_buffer, topo_condition):
    date = img.date().format("YYYY-MM-dd")
    cloud = img.select("probability").lt(45)

    mask = cloud.Not().selfMask().multiply(ee.Image.pixelArea().divide(1e6))
    mask_area = ee.Number(mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=reach_buffer,
        scale=CONFIG["resolution"]
    ).values().get(0))

    buffer_area = ee.Number(reach_buffer.area().divide(1e6))
    cloud_buffer_percent = mask_area.divide(buffer_area).multiply(100.0)

    ms = img.select("B.*").multiply(0.0001)

    results = {}
    for w_index in CONFIG["water_indices"]:
        if w_index == "ndwi":
            result = (ms.normalizedDifference(["B3", "B8"])
                      .rename(w_index).updateMask(cloud).updateMask(topo_condition)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", cloud_buffer_percent))

        elif w_index == "ndmi":
            result = (ms.normalizedDifference(["B8", "B11"])
                      .rename(w_index).updateMask(cloud).updateMask(topo_condition)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", cloud_buffer_percent))

        elif w_index == "awei_nsh":
            result = (ms.expression("4 * (G - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)", {
                "G": ms.select("B3"), "NIR": ms.select("B8"),
                "SWIR1": ms.select("B11"), "SWIR2": ms.select("B12")
            }).rename(w_index).updateMask(cloud).updateMask(topo_condition)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", cloud_buffer_percent))

        elif w_index == "awei_sh":
            result = (ms.expression("B + 2.5 * G - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2", {
                "B": ms.select("B2"), "G": ms.select("B3"), "NIR": ms.select("B8"),
                "SWIR1": ms.select("B11"), "SWIR2": ms.select("B12")
            }).rename(w_index).updateMask(cloud).updateMask(topo_condition)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", cloud_buffer_percent))

        elif w_index == "andwi":
            result = (ms.expression("(B + G + R - NIR - SWIR1 - SWIR2)/(B + G + R + NIR + SWIR1 + SWIR2)", {
                "B": ms.select("B2"), "G": ms.select("B3"), "R": ms.select("B4"),
                "NIR": ms.select("B8"), "SWIR1": ms.select("B11"), "SWIR2": ms.select("B12")
            }).rename(w_index).updateMask(cloud).updateMask(topo_condition)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", cloud_buffer_percent))
        else:
            continue

        results[w_index] = result

    return results

def get_dem_and_jrc(reach_buffer, feature):
    start_time = time.time()
    try:
        fabdem = (ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
                  .filterBounds(reach_buffer).mosaic()
                  .setDefaultProjection("EPSG:4326", None, 30))

        fabdem_per = ee.Number(fabdem.reduceRegion(
            reducer=ee.Reducer.percentile([CONFIG["dem_percentile"]]),
            geometry=reach_buffer,
            scale=CONFIG["resolution"]
        ).values().get(0))

        slope = ee.Terrain.slope(fabdem)

        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        jrc = jrc.where(jrc.gt(CONFIG["jrc_threshold"]), 1).where(jrc.lte(CONFIG["jrc_threshold"]), 2)
        jrc = jrc.reproject(crs="EPSG:4326", scale=CONFIG["resolution"])
        jrc = jrc.eq(1).selfMask()

        topo_condition = fabdem.lt(fabdem_per).Or(slope.lt(.1).And(jrc))
        jrc_condition = jrc

        feature_width = ee.Number.parse(feature.get("width"))
        print(f"feather_width:{feature_width.getInfo()}")

        result = ee.Algorithms.If(
            feature_width.lt(70),
            jrc_condition,
            topo_condition
        )
        return ee.Image(result), start_time

    except Exception as e:
        print(f"Error in get_dem_and_jrc: {e}")
        raise

def process_sentinel_data(reach_buffer, topo_condition):
    start_time = time.time()
    try:
        sentinel = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select("B.*")
                    .linkCollection(ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY"), "probability")
                    .filterBounds(reach_buffer)
                    .sort("system:time_start"))

        results_by_index = {}
        for w_index in CONFIG["water_indices"]:
            def map_cloud_mask(img):
                results = sentinel_cloud_mask(img, reach_buffer, topo_condition)
                return results[w_index]

            sentinel_mapped = sentinel.map(map_cloud_mask)

            if sentinel_mapped.size().getInfo() == 0:
                print(f"No Sentinel images available for index {w_index} after cloud filtering")
                results_by_index[w_index] = None
                continue

            sentinel_mapped = sentinel_mapped.filter(ee.Filter.lt("cloud_percent_clipped_region", 10))

            def segmentation(img):
                snic = ee.Algorithms.Image.Segmentation.SNIC(
                    image=img, size=15, compactness=0.5, connectivity=8
                )
                return snic.select(".*_mean").rename(w_index).copyProperties(img, img.propertyNames())

            sentinel_segmentation = sentinel_mapped.map(segmentation)
            results_by_index[w_index] = sentinel_segmentation

        log_time(start_time, "Sentinel Data Processing")
        return results_by_index, start_time

    except Exception as e:
        print(f"Error in process_sentinel_data: {e}")
        raise

def process_jrc_monthly(reach_buffer):
    start_time = time.time()
    sentinel = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").select("B4")
                .filterBounds(reach_buffer))
    sen_mean = sentinel.mean()
    jrc_img = ee.ImageCollection("JRC/GSW1_4/MonthlyRecurrence").select(['monthly_recurrence'], ['occ'])
    jrc_sen = jrc_img.map(lambda img: img.addBands(sen_mean).copyProperties(img, img.propertyNames()))
    return jrc_sen.select('occ'), start_time

def _open_segmentation_result(result_ref):
    if isinstance(result_ref, str):
        return xr.open_dataarray(result_ref), True
    return result_ref, False

# ----------------- ENERGY / SEGMENTATION -----------------
def add_temporal_dependency(target_map, temporal_map, method, alpha=CONFIG["energy_alpha"], beta=CONFIG["energy_alpha"]):
    if (target_map.max() > 1.0 or target_map.min() < 0.0 or
            temporal_map.max() > 1.0 or temporal_map.min() < 0.0):
        raise ValueError("Input maps must be normalized between 0 and 1")
    if target_map.shape != temporal_map.shape:
        raise ValueError("Input maps must have the same shape")
    if method == 'basic':
        combined_map = target_map * temporal_map
    else:
        modified_temporal_map = 1.0 / (1.0 + np.exp(-alpha * (temporal_map - beta)))
        combined_map = target_map * modified_temporal_map
    return combined_map

def binary_segmentation_with_maxflow(
    normalized_image, foreground_seeds=None, background_seeds=None, lambda_param=CONFIG["lambda_values"],
    site=None, w_index=None, time_index=None, time_coord=None
):
    # Keep logic same; just ensure float32 arrays to reduce memory
    normalized_image = np.asarray(normalized_image, dtype=np.float32)

    if normalized_image.max() > 1.0 or normalized_image.min() < 0.0:
        raise ValueError("Input image must be normalized between 0 and 1")

    sigma = float(np.std(normalized_image))
    if sigma == 0.0:
        sigma = 1e-6

    height, width = normalized_image.shape
    foreground_seeds = foreground_seeds or []
    background_seeds = background_seeds or []

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((height, width))

    smooth_image = gaussian_filter(normalized_image, sigma=sigma)

    # lambda_param is float; keep logic identical
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                weight = lambda_param * np.exp(-abs(smooth_image[y, x] - smooth_image[y, x + 1]) / sigma)
                g.add_edge(nodeids[y, x], nodeids[y, x + 1], weight, weight)
            if y < height - 1:
                weight = lambda_param * np.exp(-abs(smooth_image[y, x] - smooth_image[y + 1, x]) / sigma)
                g.add_edge(nodeids[y, x], nodeids[y + 1, x], weight, weight)

    for y, x in foreground_seeds:
        g.add_tedge(nodeids[y, x], 0, 1000)
    for y, x in background_seeds:
        g.add_tedge(nodeids[y, x], 1000, 0)

    fg_weight = normalized_image
    bg_weight = 1 - fg_weight

    for y in range(height):
        for x in range(width):
            if (y, x) not in foreground_seeds and (y, x) not in background_seeds:
                g.add_tedge(nodeids[y, x], float(bg_weight[y, x]), float(fg_weight[y, x]))

    flow = g.maxflow()

    segmentation = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if g.get_segment(nodeids[y, x]) == 0:
                segmentation[y, x] = 1  # land = 1
            else:
                segmentation[y, x] = 0  # river = 0

    river_val = 0
    land_val = 1
    river = (1 - segmentation) * fg_weight
    land = segmentation * bg_weight
    e_data = np.sum(river[segmentation == river_val]) + np.sum(land[segmentation == land_val])
    e_smooth = flow - e_data

    energy_terms = {
        "time_index": time_index,
        "time": time_coord,
        "lambda": lambda_param,
        "e_data": float(e_data),
        "e_smooth": float(e_smooth),
        "flow": float(flow)
    }
    return segmentation, energy_terms

# ----------------- BLUNDER DETECTION -----------------
def movAve_new(values, window_size):
    half_window = window_size // 2
    padded = np.pad(values, (half_window, half_window), mode='edge')
    moving_avg = np.array([np.mean(padded[i:i + window_size]) for i in range(len(values))])
    return moving_avg

def blunder_detection(df, window_size):
    df_clean = df.copy()
    value_columns = df_clean.select_dtypes(include=[np.number]).columns

    if len(value_columns) == 0:
        raise ValueError("No numeric column found for blunder detection.")
    elif len(value_columns) > 1:
        raise ValueError(f"More than one numeric column found: {list(value_columns)}. Please keep only one.")

    value_col = value_columns[0]

    pass_flag = True
    while pass_flag:
        moving_avg = movAve_new(df_clean[value_col].values, window_size)
        residuals = df_clean[value_col].values - moving_avg
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)

        if std_res == 0 or np.isnan(std_res):
            break

        mean_subtracted = residuals - mean_res
        abs_res = np.abs(mean_subtracted)
        max_idx = np.argmax(abs_res)
        max_val = abs_res[max_idx]

        if max_val > 3 * std_res:
            df_clean = df_clean.drop(df_clean.index[max_idx]).reset_index(drop=True)
        else:
            pass_flag = False

    return df_clean

# ----------------- LOCAL PROCESSING -----------------
def process_local_data(sentinel_segmentation, jrc_monthly_col, reach_buffer, reach_buffer_cor_gdf, site, site_save_path):
    start_time = time.time()
    results_by_index_lambda = {}

    try:
        for w_index in CONFIG["water_indices"]:
            results_by_index_lambda[w_index] = {}
            energy_data_list = []

            sentinel_data = sentinel_segmentation.get(w_index)
            if sentinel_data is None:
                for lambda_val in CONFIG["lambda_values"]:
                    results_by_index_lambda[w_index][lambda_val] = None
                print(f"No data for {w_index}, skipping")
                continue

            # ---- Sentinel segmentation cube (float32 to reduce memory)
            ds = xr.open_dataset(sentinel_data, engine="ee", crs="EPSG:4326", scale=0.0002, geometry=reach_buffer)
            ds = ds.astype("float32")

            # ---- JRC Global occurrence
            jrc_img = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
            jrc_arr = xr.open_dataset(jrc_img, engine="ee", crs="EPSG:4326", scale=0.0002, geometry=reach_buffer)
            # keep small
            try:
                jrc_arr["occurrence"] = jrc_arr["occurrence"].astype("float32")
            except Exception:
                pass

            # For plotting / grid alignment
            ds_sub = ds.sel(time=slice('2023', '2024')).mean(dim='time')
            jrc_arr = jrc_arr.interp_like(ds_sub, method='nearest').isel(time=0).drop_vars('time')

            # --- save diagnostic 01
            if CONFIG.get("debug_plots"):
                try:
                    fig0, ax0 = plt.subplots(figsize=(5, 5))
                    jrc_arr.occurrence.plot(x='lon', y='lat', robust=True, ax=ax0)
                    plt.savefig(os.path.join(site_save_path, '01_jrc_occ.png'), dpi=360, bbox_inches='tight')
                    plt.close(fig0)
                except Exception:
                    _safe_close_all_figures()

            river_mask_occ = (jrc_arr.occurrence >= 90).astype(int).values.T
            non_river_mask_occ = (jrc_arr.occurrence < 10).astype(int).values.T

            # 02
            if CONFIG.get("debug_plots"):
                try:
                    fig1, ax1 = plt.subplots(figsize=(5, 5))
                    (jrc_arr.occurrence >= 90).astype(int).plot(x="lon", y="lat", robust=True, ax=ax1)
                    plt.savefig(os.path.join(site_save_path, "02_river_mask.png"), dpi=360, bbox_inches="tight")
                    plt.close(fig1)
                except Exception:
                    _safe_close_all_figures()

            # 03
            if CONFIG.get("debug_plots"):
                try:
                    fig2, ax2 = plt.subplots(figsize=(5, 5))
                    (jrc_arr.occurrence < 10).astype(int).plot(x="lon", y="lat", ax=ax2)
                    plt.savefig(os.path.join(site_save_path, "03_river_nan_mask.png"), dpi=360, bbox_inches="tight")
                    plt.close(fig2)
                except Exception:
                    _safe_close_all_figures()

            # ---- JRC monthly occurrence cube (float32)
            ds_occ = xr.open_dataset(jrc_monthly_col, engine="ee", crs="EPSG:4326", scale=0.0002, geometry=reach_buffer)
            ds_occ = ds_occ.astype("float32")

            ds_occ_normal = (ds_occ / np.float32(100.0))
            ds_occ_mean = ds_occ_normal.median('time')
            ds_occ_fill = ds_occ_normal.fillna(np.float32(0.0))
            ds_occ_fill = xr.where(ds_occ_fill == 0, ds_occ_mean, ds_occ_fill).fillna(np.float32(0.0))

            print(f"Dataset dimensions for {w_index}: {ds.dims}")
            time_coords = pd.to_datetime(ds.time.values)

            ds_index = ds[w_index]
            ds_rolling = ds_index.rolling(time=8, center=True, min_periods=1).mean("time")
            ds_filled = ds_index.fillna(ds_rolling)

            # normalization (same logic, but float32 + eps to avoid 0/0)
            def normalize(time_slice):
                ts = time_slice.astype("float32")
                denom = (ts.max() - ts.min())
                return (ts - ts.min()) / (denom + np.float32(1e-6))

            normalized_ds = ds_filled.groupby("time").apply(normalize)
            ds_norm_filled = normalized_ds.fillna(np.float32(0.0))
            ds_norm_filled2 = ds_norm_filled.assign_coords(month=ds_norm_filled.time.dt.month - 1)

            # forcing by JRC occurrence
            ds_norm_filled2 = xr.where(jrc_arr.occurrence >= 90, np.float32(1.0), ds_norm_filled2)

            # 04
            if CONFIG.get("debug_plots"):
                try:
                    fig3 = ds_norm_filled2.sel(time="2024").plot(x="lon", y="lat", col="time", col_wrap=10)
                    plt.savefig(os.path.join(site_save_path, "04_index_norm.png"), dpi=360, bbox_inches="tight")
                    plt.close(fig3.fig)
                except Exception:
                    _safe_close_all_figures()

            # ---- Reach raster fill
            reach_shp_path = os.path.join(site_save_path, "reach.shp")
            if not os.path.exists(reach_shp_path):
                raise FileNotFoundError(f"Missing reach shapefile in site folder: {reach_shp_path}")

            reach_load = gpd.read_file(reach_shp_path)
            reach_geom = reach_load.geometry

            ds_sub_dim = ds_sub.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
            reach_raster = features.rasterize(
                reach_geom,
                out_shape=ds_sub_dim.rio.shape,
                transform=ds_sub_dim.rio.transform()
            )
            reach_raster_arr = xr.DataArray(
                reach_raster,
                coords={'lat': ds_sub_dim.lat, 'lon': ds_sub_dim.lon},
                dims=('lat', 'lon')
            )
            reach_raster_arr = reach_raster_arr.transpose('lon', 'lat')

            # 05
            if CONFIG.get("debug_plots"):
                try:
                    fig4, ax4 = plt.subplots(figsize=(5, 5))
                    reach_raster_arr.plot(x="lon", y="lat", ax=ax4)
                    plt.savefig(os.path.join(site_save_path, "05_reach_raster.png"), dpi=360, bbox_inches="tight")
                    plt.close(fig4)
                except Exception:
                    _safe_close_all_figures()

            # forcing by reach raster
            ds_norm_filled2 = xr.where(reach_raster_arr == 1, np.float32(1.0), ds_norm_filled2)

            # 06
            if CONFIG.get("debug_plots"):
                try:
                    fig5 = ds_norm_filled2.sel(time="2024").plot(x="lon", y="lat", col="time", col_wrap=10)
                    plt.savefig(os.path.join(site_save_path, "06_index_norm_filled_by_reach.png"), dpi=360, bbox_inches="tight")
                    plt.close(fig5.fig)
                except Exception:
                    _safe_close_all_figures()

            # ---- Maxflow per lambda, per time
            T = ds_norm_filled2.sizes["time"]
            ny = ds_norm_filled2.sizes["lat"]
            nx = ds_norm_filled2.sizes["lon"]

            foreground_indices = np.argwhere(river_mask_occ)
            background_indices = np.argwhere(non_river_mask_occ)
            foreground_seeds = [tuple(idx) for idx in foreground_indices]
            background_seeds = [tuple(idx) for idx in background_indices]

            month_indices = ds_norm_filled2.month.values.astype(int)
            seed_samples_by_time = []
            td_base_mmap_path = os.path.join(site_save_path, f"._{site}_{w_index}_td_base.dat")
            td_base_mm = np.memmap(td_base_mmap_path, dtype=np.float32, mode="w+", shape=(T, ny, nx))

            for t in range(T):
                index_slice = np.asarray(ds_norm_filled2.isel(time=t).data, dtype=np.float32).T

                if index_slice.shape != (ny, nx):
                    raise ValueError(f"Time slice shape mismatch: {index_slice.shape}")

                month_index = int(month_indices[t])
                occ_slice = np.asarray(ds_occ_fill.isel(time=month_index).occ.data, dtype=np.float32).T

                num_fg = min(20, len(foreground_seeds)) if foreground_seeds else 0
                num_bg = min(20, len(background_seeds)) if background_seeds else 0
                foreground_seeds_sample = random.sample(foreground_seeds, num_fg) if num_fg > 0 else []
                background_seeds_sample = random.sample(background_seeds, num_bg) if num_bg > 0 else []
                seed_samples_by_time.append((foreground_seeds_sample, background_seeds_sample))

                td_map = add_temporal_dependency(index_slice, occ_slice, method=CONFIG['energy_method'])
                td_base_mm[t, :, :] = td_map.astype(np.float32, copy=False)

            td_base_mm.flush()

            if CONFIG.get("debug_plots"):
                try:
                    td_preview_arr = xr.DataArray(
                        data=td_base_mm,
                        dims=["time", "lat", "lon"],
                        coords={"time": time_coords, "lat": ds.lat, "lon": ds.lon},
                        name=f"temporal_dependency_{w_index}"
                    )
                    fig6 = td_preview_arr.sel(time="2024").plot(x="lon", y="lat", col="time", col_wrap=10)
                    plt.savefig(os.path.join(site_save_path, "07_temporal_dependency.png"), dpi=360, bbox_inches="tight")
                    plt.close(fig6.fig)
                    del td_preview_arr
                except Exception:
                    _safe_close_all_figures()

            for lambda_val in CONFIG["lambda_values"]:
                seg_mmap_path = os.path.join(site_save_path, f"._{site}_{w_index}_lambda{lambda_val}_seg.dat")
                seg_nc_path = os.path.join(site_save_path, f"{site}_{w_index}_lambda_{lambda_val}_segmentation.nc")

                seg_mm = np.memmap(seg_mmap_path, dtype=np.uint8, mode="w+", shape=(T, ny, nx))

                for t in range(T):
                    td_map = np.asarray(td_base_mm[t, :, :], dtype=np.float32)
                    foreground_seeds_sample, background_seeds_sample = seed_samples_by_time[t]

                    bn_seg, energy_terms = binary_segmentation_with_maxflow(
                        td_map,
                        foreground_seeds_sample,
                        background_seeds_sample,
                        lambda_val,
                        site=site,
                        w_index=w_index,
                        time_index=t,
                        time_coord=str(time_coords[t])
                    )

                    seg_mm[t, :, :] = bn_seg.astype(np.uint8, copy=False)
                    energy_data_list.append(energy_terms)

                seg_mm.flush()

                results_arr = xr.DataArray(
                    data=seg_mm,
                    dims=["time", "lat", "lon"],
                    coords={"time": time_coords, "lat": ds.lat, "lon": ds.lon},
                    name=f"segmentation_{w_index}_lambda_{lambda_val}"
                )

                if CONFIG.get("debug_plots"):
                    try:
                        fig7 = results_arr.sel(time="2024").plot(x="lon", y="lat", col="time", col_wrap=10)
                        plt.savefig(os.path.join(site_save_path, "08_maxflow_result.png"), dpi=360, bbox_inches="tight")
                        plt.close(fig7.fig)
                    except Exception:
                        _safe_close_all_figures()

                seg_arr = results_arr
                seg_arr = xr.where(seg_arr == 1, np.nan, seg_arr)

                seg_arr = seg_arr.rio.write_crs('EPSG:4326', inplace=True)
                seg_arr = seg_arr.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
                seg_arr.rio.write_coordinate_system(inplace=True)

                seg_arr_clip = seg_arr.rio.clip(reach_buffer_cor_gdf.geometry, reach_buffer_cor_gdf.crs)
                seg_arr_clip = seg_arr_clip.astype("float32")
                seg_arr_clip.to_netcdf(seg_nc_path)
                results_by_index_lambda[w_index][lambda_val] = seg_nc_path

                del seg_mm, results_arr, seg_arr, seg_arr_clip
                try:
                    if os.path.exists(seg_mmap_path):
                        os.remove(seg_mmap_path)
                except OSError:
                    pass
                gc.collect()

            del td_base_mm, seed_samples_by_time, foreground_seeds, background_seeds
            try:
                if os.path.exists(td_base_mmap_path):
                    os.remove(td_base_mmap_path)
            except OSError:
                pass
            gc.collect()

            # ---- Save energy terms
            if energy_data_list:
                energy_df = pd.DataFrame(energy_data_list)
                pivoted_data = {"time": pd.to_datetime(time_coords)}

                for lambda_val in CONFIG["lambda_values"]:
                    lambda_data = energy_df[energy_df["lambda"] == lambda_val]
                    pivoted_data[f"e_data_lambda_{lambda_val}"] = lambda_data["e_data"].values
                    pivoted_data[f"e_smooth_lambda_{lambda_val}"] = lambda_data["e_smooth"].values
                    pivoted_data[f"flow_lambda_{lambda_val}"] = lambda_data["flow"].values

                energy_df_pivoted = pd.DataFrame(pivoted_data)
                energy_csv_path = os.path.join(site_save_path, f"{site}_energy_{w_index}.csv")
                energy_df_pivoted.to_csv(energy_csv_path, index=False)
                print(f"Saved energy terms for {w_index} to {energy_csv_path}")

            # cleanup big datasets per index
            del ds, jrc_arr, ds_occ, ds_occ_normal, ds_occ_mean, ds_occ_fill
            del ds_sub, ds_index, ds_rolling, ds_filled, normalized_ds, ds_norm_filled, ds_norm_filled2
            gc.collect()

        log_time(start_time, "Local Data Processing")
        return results_by_index_lambda, start_time

    except Exception as e:
        print(f"Error in process_local_data: {e}")
        raise

# ----------------- WIDTHS -----------------
def calculate_river_width(results_by_index_lambda, river_length, site, site_save_path, lat, lon, reach, buffer_area):
    start_time = time.time()
    results_by_index = {}

    try:
        geod = Geod(ellps="WGS84")

        sample_index = next(
            (w for w in CONFIG["water_indices"]
             if results_by_index_lambda.get(w, {}).get(CONFIG["lambda_values"][0]) is not None),
            None
        )
        if sample_index is None:
            print(f"No valid data for site {site}, skipping river width calculation")
            return {}, start_time

        seg_sample, sample_from_disk = _open_segmentation_result(
            results_by_index_lambda[sample_index][CONFIG["lambda_values"][0]]
        )
        lon_coords = seg_sample.coords["lon"].values
        lat_coords = seg_sample.coords["lat"].values
        if sample_from_disk and hasattr(seg_sample, "close"):
            seg_sample.close()

        scale = 0.0002
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        lon_left, lon_right = lon_grid - scale / 2, lon_grid + scale / 2
        lat_bottom, lat_top = lat_grid - scale / 2, lat_grid + scale / 2

        cell_areas = np.zeros_like(lon_grid, dtype=float)
        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                lons = [lon_left[i, j], lon_right[i, j], lon_right[i, j], lon_left[i, j]]
                lats = [lat_bottom[i, j], lat_bottom[i, j], lat_top[i, j], lat_top[i, j]]
                cell_areas[i, j] = abs(geod.polygon_area_perimeter(lons, lats)[0])

        cell_areas_da = xr.DataArray(
            cell_areas,
            dims=["lat", "lon"],
            coords={"lat": lat_coords, "lon": lon_coords}
        )

        metadata = [
            f"site: {site} (lat: {lat}, lon: {lon})",
            f"river reach_id: {reach.get('reach_id').getInfo()}",
            f"river width: {reach.get('width').getInfo()}",
            f"buffer area (km²): {buffer_area}",
            f"river length (km): {river_length}",
            ""
        ]

        for w_index in CONFIG["water_indices"]:
            width_dict = {"time": None}

            for lambda_val in CONFIG["lambda_values"]:
                results_ref = results_by_index_lambda.get(w_index, {}).get(lambda_val)
                col = f"width_lambda_{lambda_val}"

                if results_ref is None:
                    if width_dict["time"] is not None:
                        width_dict[col] = [np.nan] * len(width_dict["time"])
                    continue

                results_arr, results_from_disk = _open_segmentation_result(results_ref)
                try:
                    times = results_arr.time.values
                    cluster_area = ((results_arr == 0) * cell_areas_da).sum(dim=["lon", "lat"])
                    river_width = cluster_area / (river_length * 1000.0)

                    width_df_lambda = river_width.to_dataframe(name=col)
                    width_dict[col] = width_df_lambda[col].values

                    if width_dict["time"] is None:
                        width_dict["time"] = times
                finally:
                    if results_from_disk and hasattr(results_arr, "close"):
                        results_arr.close()

            if width_dict["time"] is None:
                print(f"No valid data for {w_index}, skipping CSV save")
                results_by_index[w_index] = None
                continue

            width_df = pd.DataFrame(width_dict)

            width_df_smooth = width_df[["time"]].copy()
            lambda_cols = [c for c in width_df.columns if c != "time" and pd.api.types.is_numeric_dtype(width_df[c])]
            if not lambda_cols:
                results_by_index[w_index] = None
                continue

            for col in lambda_cols:
                tmp = width_df[["time", col]].dropna(subset=[col]).reset_index(drop=True)
                if len(tmp) == 0:
                    width_df_smooth[col] = np.nan
                    continue
                tmp_clean = blunder_detection(tmp, 3)
                width_df_smooth = width_df_smooth.merge(tmp_clean[["time", col]], on="time", how="left")

            ok_cols = [f"width_lambda_{lv}" for lv in CONFIG["lambda_values"]
                       if f"width_lambda_{lv}" in width_df_smooth.columns]
            if not ok_cols:
                results_by_index[w_index] = None
                continue

            width_da = xr.DataArray(
                data=width_df_smooth[ok_cols].values,
                dims=["time", "lambda"],
                coords={
                    "time": width_df_smooth["time"],
                    "lambda": [float(c.split("_")[-1]) for c in ok_cols]
                },
                name=f"width_{w_index}"
            )
            results_by_index[w_index] = width_da

            width_csv_path = os.path.join(site_save_path, f"{site}_widths_{w_index}.csv")
            with open(width_csv_path, "w", encoding="utf-8") as f:
                f.write("\n".join(metadata))
            width_df_smooth.to_csv(width_csv_path, mode="a", index=False)
            print(f"Saved widths to {width_csv_path}")

        log_time(start_time, "River Width Calculation")
        return results_by_index, start_time

    except Exception as e:
        print(f"Error in calculate_river_width: {e}")
        raise

# ----------------- SINGLE-SITE PROCESS -----------------
def process_site(site, latitude, longitude):
    os.makedirs(CONFIG["output_path"], exist_ok=True)
    reaches = ee.FeatureCollection("projects/sat-io/open-datasets/SWORD/reaches_merged")
    site = str(site)
    lon, lat = float(longitude), float(latitude)
    print(f"\nProcessing site: {site} (lat: {lat}, lon: {lon})")
    site_dir = os.path.join(
        CONFIG["output_path"], site.replace(" ", "_").replace("/", "_")
    )
    os.makedirs(site_dir, exist_ok=True)

    try:
        reach, start_time = find_reach(reaches, lon=lon, lat=lat)
        reach_buffer = create_buffer_zone(reach, CONFIG["buffer_coef"])
        reach_buffer = ee.Geometry(reach_buffer).simplify(100)
        buffer_area = reach_buffer.area().divide(1e6).getInfo()
        log_time(start_time, "Reach Identification")
        print(f"Buffer area (km²): {buffer_area}")

        reach_fc = ee.FeatureCollection([reach])
        shp_path = os.path.join(site_dir, "reach.shp")
        geemap.ee_to_shp(reach_fc, shp_path)
        reach_gpd = gpd.read_file(shp_path)
        reach_buf_cor = reach_gpd.buffer(0.008, cap_style='flat')
        reach_buf_cor_gdf = gpd.GeoDataFrame(geometry=reach_buf_cor, crs=reach_gpd.crs)

        river_length, start_time = calculate_reach_length(reach, "single_reach", reaches)
        log_time(start_time, "River Length Calculation")
        print(f"River length (km): {river_length}")

        save_map_visualization(site, lon, lat, reach_buffer, reach.geometry(), site_dir)
        save_corrected_buffer_plot_sat(site, lon, lat, reach_buf_cor_gdf, reach_buffer, reach, site_dir)
        topo_condition, start_time = get_dem_and_jrc(reach_buffer, reach)
        log_time(start_time, "DEM and JRC Data Loading")
        sentinel_segmentation, start_time = process_sentinel_data(reach_buffer, topo_condition)
        log_time(start_time, "Sentinel Data Processing")
        jrc_months, start_time = process_jrc_monthly(reach_buffer)
        log_time(start_time, "JRC monthly collection")

        if all(v is None for v in sentinel_segmentation.values()):
            raise RuntimeError(f"No valid Sentinel-2 data found for {site}")

        results_by_index_lambda, start_time = process_local_data(
            sentinel_segmentation, jrc_months, reach_buffer, reach_buf_cor_gdf, site, site_dir
        )
        log_time(start_time, "Local Data Processing")
        calculate_river_width(
            results_by_index_lambda, river_length, site, site_dir, lat, lon, reach, buffer_area
        )
        log_time(start_time, "River Width Calculation")
    finally:
        _safe_close_all_figures()
        gc.collect()

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Run RiWEx Sentinel-2 for one location")
    p.add_argument("--site", default=CONFIG["site"], help="A label for the site")
    p.add_argument("--latitude", type=float, default=CONFIG["latitude"])
    p.add_argument("--longitude", type=float, default=CONFIG["longitude"])
    p.add_argument(
        "--project-id",
        default=CONFIG["project_id"],
        help="Your Earth Engine-enabled Google Cloud project ID",
    )
    p.add_argument("--indices", nargs="+", default=CONFIG["water_indices"],
                   help="Water indices to run: ndwi ndmi awei_nsh awei_sh andwi")
    p.add_argument("--lambdas", nargs="+", type=float, default=CONFIG["lambda_values"],
                   help="Lambda values. Example: --lambdas 0.01 0.1 1")
    p.add_argument("--debug-plots", action="store_true",
                   help="Save diagnostic plots. Off by default for faster runs.")
    return p.parse_args()

def main():
    args = parse_args()
    CONFIG["project_id"] = args.project_id
    CONFIG["water_indices"] = args.indices
    CONFIG["lambda_values"] = args.lambdas
    CONFIG["debug_plots"] = args.debug_plots

    initialize_earth_engine()
    process_site(args.site, args.latitude, args.longitude)

if __name__ == "__main__":
    main()
