import os
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import rasterio
import urllib.request
import rioxarray
import ee
import geemap
import time
import argparse
import glob
import re
from shapely.geometry import shape as shapely_shape, mapping
import matplotlib.pyplot as plt
from datetime import datetime
from calendar import monthrange
import stackstac
from pystac_client import Client
import xarray as xr
from skimage.transform import resize as skimage_resize
from scipy.ndimage import gaussian_filter
from maxflow import Graph
import math
from pyproj import Geod, CRS
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ==========================================================
# CONFIGURATION
# ==========================================================

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Edit these values to process a different location, or override them
    # with the command-line options documented in README.md.
    "site": "example_site",
    "latitude": -3.312,
    "longitude": -60.598,
    "output_path": os.path.join(HERE, "results_modis"),
    # Use --project-id or set GOOGLE_CLOUD_PROJECT. Never commit credentials.
    "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
    "ee_url":      "https://earthengine.googleapis.com",
    "buffer_coef": 1,
}

# ==========================================================
# UTILITIES
# ==========================================================

def log_time(start, name):
    print(f"{name} took {time.time() - start:.2f} s")
    return time.time()

def initialize_earth_engine():
    if not CONFIG["project_id"]:
        raise ValueError(
            "An Earth Engine-enabled Google Cloud project is required. "
            "Pass --project-id YOUR_PROJECT_ID or set GOOGLE_CLOUD_PROJECT."
        )
    try:
        ee.Authenticate()
        ee.Initialize(project=CONFIG["project_id"], opt_url=CONFIG["ee_url"])
    except ee.EEException as e:
        raise RuntimeError("Earth Engine init failed") from e

def save_ee_feature_to_shapefile(ee_feature, save_dir, filename):
    feature_info = ee_feature.getInfo()
    geometry = shapely_shape(feature_info["geometry"])
    properties = feature_info.get("properties", {})

    gdf = gpd.GeoDataFrame([properties], geometry=[geometry], crs="EPSG:4326")

    shp_dir = os.path.join(save_dir, "shapefile")
    os.makedirs(shp_dir, exist_ok=True)

    shp_path = os.path.join(shp_dir, filename)
    gdf.to_file(shp_path)
    print(f"✔ Shapefile saved: {shp_path}")

def find_reach(reach_collection, lon, lat):
    point = ee.Geometry.Point([lon, lat])
    buff = 1000
    max_buff = 10000
    while buff <= max_buff:
        print(f"Searching reach within {buff} m …")
        filtered = reach_collection.filterBounds(point.buffer(buff))
        if filtered.size().getInfo() > 0:
            nearest = filtered.map(
                lambda f: f.set("dist", f.geometry().distance(point))
            ).sort("dist").first()
            rid = nearest.get("reach_id").getInfo()
            if rid:
                print(f"  → Found reach_id {rid}")
                return nearest, time.time()
        buff += 1000
    raise ValueError(f"No reach near {lon},{lat} within {max_buff} m")

def create_buffer_zone(feature, coef):
    width = ee.Number.parse(feature.get("width"))
    var = ee.Number.parse(feature.get("width_var")).sqrt()
    buff = ee.Number(coef).multiply(var).add(width)
    return feature.geometry().buffer(buff)

def save_map_visualization(site, lon, lat, reach_buffer, reach_geom, save_dir):
    gauge = ee.Feature(ee.Geometry.Point([lon, lat]), {"name": site})
    zoom_region = reach_buffer.buffer(4000)

    try:
        vis = (
            ee.Image().paint(reach_buffer, 1, 2).visualize(palette=["red"])
            .blend(ee.Image().paint(reach_geom, 1, 2).visualize(palette=["blue"]))
            .blend(ee.Image().paint(gauge.geometry().buffer(100), 1).visualize(palette=["yellow"]))
        )

        url = vis.getThumbURL({
            "dimensions": "800x600",
            "region": zoom_region,
        })
        map_path = os.path.join(save_dir, f"{str(site)}_map.png")
        urllib.request.urlretrieve(url, map_path)
        print(f"✔ Map saved: {map_path}")
    except Exception as e:
        print(f"⚠️ Failed to save map: {e}")

# ==========================================================
# MAIN: one configured location → site folder
# ==========================================================

def main():
    start = time.time()

    parser = argparse.ArgumentParser(description="Run RiWEx MODIS for one location")
    parser.add_argument("--site", default=CONFIG["site"], help="A label for the site")
    parser.add_argument("--latitude", type=float, default=CONFIG["latitude"])
    parser.add_argument("--longitude", type=float, default=CONFIG["longitude"])
    parser.add_argument(
        "--project-id",
        default=CONFIG["project_id"],
        help="Your Earth Engine-enabled Google Cloud project ID",
    )
    args = parser.parse_args()

    CONFIG["project_id"] = args.project_id
    os.makedirs(CONFIG["output_path"], exist_ok=True)

    initialize_earth_engine()
    reaches = ee.FeatureCollection("projects/sat-io/open-datasets/SWORD/reaches_merged")

    site = str(args.site).replace(" ", "_").replace("/", "_")
    lat, lon = float(args.latitude), float(args.longitude)
    dataset_dir = os.path.join(CONFIG["output_path"], "sites")
    site_dir = os.path.join(dataset_dir, site)
    os.makedirs(site_dir, exist_ok=True)

    print(f"\n📍 Site: {site}  (lat={lat}, lon={lon})")
    reach, _ = find_reach(reaches, lon=lon, lat=lat)
    reach_id = str(reach.get("reach_id").getInfo())
    save_ee_feature_to_shapefile(reach, site_dir, f"{reach_id}.shp")
    buff = create_buffer_zone(reach, CONFIG["buffer_coef"])
    buff = ee.Geometry(buff).simplify(100)
    save_map_visualization(reach_id, lon, lat, buff, reach.geometry(), site_dir)

    log_time(start, "MAIN")
    print("✔ Site preparation complete")

# ==========================================================
# RUN MAIN FIRST
# ==========================================================

if __name__ == "__main__":
    main()

# ==========================================================
# DOWNSTREAM PROCESSING NOW PER SITE
# ==========================================================

base_dir = CONFIG["output_path"]

# Process the generated site collection.
dataset_dirs = [
    os.path.join(base_dir, d)
    for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
]

# STAC settings
stac_endpoint = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
collections = ["GWP_P1D"]
dates_from_to = ["2003-01-01", "2022-12-31"]
target_band = "gwpp1d"
margin = 1
m = 1
n = 1
p = 0.5
energy_method = 'basic'  # or 'advanced'

# ==========================================================
# FUNCTIONS FOR ANALYSIS (kept as in your logic)
# ==========================================================

def stack_geotiffs(input_folder, output_path, order_path, stack_name, gdf, m, n):
    tif_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".tif") and f != os.path.basename(output_path)
    ]
    tif_files.sort()
    if not tif_files:
        print(f"⚠ No TIFFs to stack in {input_folder}")
        return

    order_file = os.path.join(order_path, f"tif_order_{stack_name}.txt")
    with open(order_file, "w") as f:
        for tif in tif_files:
            f.write(os.path.basename(tif) + "\n")

    try:
        reach_buf = gdf.buffer(0.008, cap_style='flat')
        reach_buff_union = reach_buf.unary_union
        reach_buff_series = gpd.GeoSeries([reach_buff_union], crs=gdf.crs)
        mask_geom = [reach_buff_series.iloc[0].__geo_interface__]
    except Exception:
        mask_geom = None

    stacked_image = []
    dates = []

    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            if mask_geom:
                cropped, tr = mask(src, mask_geom, crop=True, nodata=0)
            else:
                cropped = src.read()
                tr = src.transform

            stacked_image.append(cropped[0])
            filename = os.path.basename(tif_file)
            try:
                # filename like "2003-01-01_daily_gwpp1d.tif" → "200301"
                dt_raw = filename.split("_")[0].replace("-", "")  # 20030101
                dt = dt_raw[:6]  # keep YYYYMM for monthly logic
                dates.append(dt)
            except Exception:
                dates.append("unknown")

    stacked_image = np.stack(stacked_image, axis=0)

    with rasterio.open(tif_files[0]) as src:
        meta = src.meta.copy()
    meta.update({
        "count": len(stacked_image),
        "height": stacked_image.shape[1],
        "width": stacked_image.shape[2],
        "transform": tr,
        "nodata": 0
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        for i, (band, dt) in enumerate(zip(stacked_image, dates), start=1):
            dst.write(band, i)
            if dt != "unknown":
                dst.update_tags(i, date=dt)

    for tif in tif_files:
        os.remove(tif)

    print(f"✔ Stacked GeoTIFF saved: {output_path}")


def normalize_stacked_geotiff(input_path, output_path, start_date, end_date):
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        normalized_bands = []
        tags = []

        for band_idx in range(1, src.count + 1):
            band_tag = src.tags(band_idx).get('date', None)
            if band_tag and re.match(r'^\d{6}$', band_tag):
                band_date = datetime.strptime(band_tag, "%Y%m")
                if start_date <= band_date <= end_date:
                    days = monthrange(band_date.year, band_date.month)[1]
                    band_data = src.read(band_idx).astype('float32')
                    normalized_bands.append(band_data / days)
                    tags.append(band_tag)

        if not normalized_bands:
            print(f"⚠ No valid bands in {input_path}")
            return

        profile.update(dtype='float32', count=len(normalized_bands))

        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, (nb, tg) in enumerate(zip(normalized_bands, tags), start=1):
                dst.write(nb, i)
                dst.update_tags(i, date=tg)

        print(f"✔ Normalized stacked saved: {output_path}")


def calculate_water_occurrence(normalized_geotiff_path, output_folder, water_occurrence_filename="water_occurrence.tif"):
    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(normalized_geotiff_path) as src:
        data = src.read()
        water_occ = np.mean(data, axis=0)
        mn, mx = np.min(water_occ), np.max(water_occ)
        if mx > mn:
            water_occ = (water_occ - mn) / (mx - mn)
        else:
            water_occ[:] = 0.0

        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)

        outp = os.path.join(output_folder, water_occurrence_filename)
        with rasterio.open(outp, 'w', **profile) as dst:
            dst.write(water_occ.astype(rasterio.float32), 1)

        print(f"✔ Water occurrence: {outp}")


def normalize_stacked_geotiff_for_month(stacked_geotiff_file, normalized_geotiff_file, month, start_year, end_year):
    start_date = datetime(start_year, month, 1)
    end_date = datetime(end_year, month, monthrange(end_year, month)[1])

    with rasterio.open(stacked_geotiff_file) as src:
        profile = src.profile.copy()
        normalized_bands = []

        for band in range(1, src.count + 1):
            tag = src.tags(band).get('date', None)
            if tag and re.match(r'^\d{6}$', tag):
                bd = datetime.strptime(tag, '%Y%m')
                if start_date <= bd <= end_date and bd.month == month:
                    days = monthrange(bd.year, bd.month)[1]
                    normalized_bands.append(src.read(band).astype('float32') / days)

        if not normalized_bands:
            print(f"⚠ No monthly bands for {month} in {stacked_geotiff_file}")
            return

        profile.update(dtype='float32', count=len(normalized_bands))
        with rasterio.open(normalized_geotiff_file, 'w', **profile) as dst:
            for i, nb in enumerate(normalized_bands, start=1):
                dst.write(nb, i)

        print(f"✔ Monthly normalized saved: {normalized_geotiff_file}")


def load_normalized_monthly_data(normalized_tiff_path):
    normalized = {}
    with rasterio.open(normalized_tiff_path) as src:
        for i in range(1, src.count + 1):
            tag = src.tags(i).get('date', None)
            if tag and re.match(r'^\d{6}$', tag):
                normalized[tag] = src.read(i)
    return normalized


def add_temporal_dependency(target_map, temporal_map, method='basic', alpha=1.0, beta=0.5):
    if method == 'basic':
        return target_map * temporal_map
    else:
        modified = 1.0 / (1.0 + np.exp(-alpha * (temporal_map - beta)))
        return target_map * modified


def load_monthly_occurrence(monthly_dir, reach_id):
    occ_dir = os.path.join(monthly_dir, "water_occurrence")
    occ = {}

    for mth in [f"{i:02d}" for i in range(1, 13)]:
        fn = f"water_occurrence_stacked_GWP1_month_{mth}.tif"
        path = os.path.join(occ_dir, fn)
        if os.path.exists(path):
            with rasterio.open(path) as src:
                occ[mth] = src.read(1)
    return occ


def process_daily_maps(daily_dir, monthly_occ, normalized_monthly_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tif_files = sorted(glob.glob(os.path.join(daily_dir, "*.tif")))

    for tif in tif_files:
        date_str = os.path.basename(tif)[:10]
        yr = date_str[:4]
        mo = date_str[5:7]
        y_m = f"{yr}{mo}"

        out_file = os.path.join(output_dir, os.path.basename(tif).replace(".tif", "_temporal.tif"))
        if os.path.exists(out_file):
            continue

        if mo not in monthly_occ or y_m not in normalized_monthly_data:
            continue

        with rasterio.open(tif) as src:
            daily = src.read(1)
            profile = src.profile

        occ = monthly_occ[mo]
        norm = normalized_monthly_data[y_m]

        if occ.shape != daily.shape:
            occ = skimage_resize(occ, daily.shape, order=1, mode='reflect', anti_aliasing=True)
        if norm.shape != daily.shape:
            norm = skimage_resize(norm, daily.shape, order=1, mode='reflect', anti_aliasing=True)

        occ = np.where(occ == 0, 0.1, occ)
        norm = np.where(norm == 0, 0.1, norm)
        daily = np.where(daily == 0, 0.1, daily)

        temporal = occ * p * norm

        enhanced = add_temporal_dependency(daily, temporal, energy_method)

        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(enhanced, 1)

        print(f"✔ Temporal processed: {out_file}")


def get_utm_epsg_from_lonlat(lon, lat):
    zone = math.floor((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def calculate_white_pixel_area(tiff_path):
    with rasterio.open(tiff_path) as src:
        image = src.read(1)
        transform = src.transform
        height, width = image.shape

        geod = Geod(ellps="WGS84")

        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        lon_grid = np.array(xs).reshape(height, width)
        lat_grid = np.array(ys).reshape(height, width)

        dlon = (transform.a / 2.0)
        dlat = (abs(transform.e) / 2.0)

        lon_left = lon_grid - dlon
        lon_right = lon_grid + dlon
        lat_bottom = lat_grid - dlat
        lat_top = lat_grid + dlat

        cell_areas = np.zeros((height, width), dtype=np.float64)
        for i in range(height):
            for j in range(width):
                lons = [lon_left[i,j], lon_right[i,j], lon_right[i,j], lon_left[i,j]]
                lats = [lat_bottom[i,j], lat_bottom[i,j], lat_top[i,j], lat_top[i,j]]
                cell_areas[i,j] = abs(geod.polygon_area_perimeter(lons, lats)[0])

        mask_black = (image == 0)
        total_area = cell_areas[mask_black].sum()

        return total_area, image.shape, transform, src.crs


def get_reach_length(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    if 'reach_len' in gdf.columns:
        return gdf['reach_len'].sum()
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3857)
    return gdf.length.sum()


def binary_segmentation_with_maxflow(
        normalized_image,
        foreground_seeds=None,
        background_seeds=None,
        lambda_param=1.0,
        sigma=1.0):
    if normalized_image.max() > 1.0 or normalized_image.min() < 0.0:
        raise ValueError("Image must be normalized [0,1]")

    H, W = normalized_image.shape
    g = Graph[float]()
    nodeids = g.add_grid_nodes((H, W))
    smooth = gaussian_filter(normalized_image, sigma=sigma)

    for y in range(H):
        for x in range(W):
            if x < W - 1:
                diff = smooth[y,x] - smooth[y,x+1]
                w = lambda_param * np.exp(-(diff**2)/(2*sigma**2))
                g.add_edge(nodeids[y,x], nodeids[y,x+1], w, w)
            if y < H - 1:
                diff = smooth[y,x] - smooth[y+1,x]
                w = lambda_param * np.exp(-(diff**2)/(2*sigma**2))
                g.add_edge(nodeids[y,x], nodeids[y+1,x], w, w)

    for y in range(H):
        for x in range(W):
            fg = normalized_image[y,x]
            bg = 1.0 - fg
            g.add_tedge(nodeids[y,x], bg, fg)

    g.maxflow()
    seg = np.zeros((H,W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            if g.get_segment(nodeids[y,x]) == 0:
                seg[y,x] = 1
    return seg

# ==========================================================
# MAIN DOWNSTREAM LOOP (per dataset → per site)
# ==========================================================

for dataset_dir in dataset_dirs:

    site_dirs = [
        os.path.join(dataset_dir, s)
        for s in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, s))
    ]

    for site_dir in site_dirs:
        station_name = os.path.basename(site_dir)

        shp_files = glob.glob(os.path.join(site_dir, "shapefile", "*.shp"))
        if not shp_files:
            print(f"⚠ Missing shapefile in site: {station_name}")
            continue

        shapefile_path = shp_files[0]
        reach_id = os.path.splitext(os.path.basename(shapefile_path))[0]

        daily_dir = os.path.join(site_dir, "data_Daily_buffer")
        data_monthly_root = os.path.join(site_dir, "data_Monthly")
        normalized_tiff = os.path.join(data_monthly_root, "normalized_stacked_GWP1.tif")
        monthly_dir = os.path.join(data_monthly_root, "monthly_normalized")
        output_dir = os.path.join(site_dir, "data_Daily_temporal")
        segmented_dir = os.path.join(site_dir, "data_Daily_segmented")

        os.makedirs(daily_dir, exist_ok=True)
        os.makedirs(data_monthly_root, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(segmented_dir, exist_ok=True)

        # ==================================================
        # STEP 1 — DAILY STAC download + buffer masking
        # ==================================================

        catalog = Client.open(stac_endpoint)
        sample_daily_file = os.path.join(daily_dir, "2003-01-01_daily_gwpp1d.tif")

        if not os.path.exists(sample_daily_file):
            try:
                gdf = gpd.read_file(shapefile_path).to_crs(4326)
                minx, miny, maxx, maxy = gdf.total_bounds
                bbox = [minx-margin, miny-margin, maxx+margin, maxy+margin]

                stac_items = catalog.search(
                    collections=collections,
                    datetime=dates_from_to,
                    bbox=bbox
                ).item_collection()

                if len(stac_items)==0:
                    print(f"⚠ No STAC items for {reach_id}")
                    continue

                gwp = stackstac.stack(stac_items, bounds_latlon=bbox, epsg=4326)
                band_idx = [i for i,b in enumerate(gwp.band.values) if b==target_band][0]

                try:
                    reach_buf = gdf.buffer(0.008, cap_style='flat')
                    merged = reach_buf.unary_union
                    mask_geom = [merged.__geo_interface__]
                except Exception:
                    mask_geom = None

                for it in range(len(gwp.time.values)):
                    b = gwp[it][band_idx]
                    xt = b.time.values
                    date_str = np.datetime_as_string(xt, unit='D')
                    out = os.path.join(daily_dir, f"{date_str}_daily_{target_band}.tif")

                    with rasterio.Env():
                        with rasterio.MemoryFile() as mem:
                            prof = {
                                "driver": "GTiff",
                                "height": b.shape[0],
                                "width": b.shape[1],
                                "count": 1,
                                "dtype": b.dtype,
                                "crs": b.rio.crs,
                                "transform": b.rio.transform(),
                                "nodata": 0
                            }
                            with mem.open(**prof) as tmp:
                                tmp.write(b.values,1)

                            with mem.open() as tmp2:
                                if mask_geom:
                                    cropped, tr = mask(tmp2, mask_geom, crop=True, nodata=0)
                                else:
                                    cropped = tmp2.read()
                                    tr = tmp2.transform

                                meta = tmp2.meta.copy()
                                meta.update({
                                    "height": cropped.shape[1],
                                    "width": cropped.shape[2],
                                    "transform": tr,
                                    "nodata": 0
                                })
                                with rasterio.open(out,"w",**meta) as dst:
                                    dst.write(cropped[0],1)

            except Exception as e:
                print(f"⚠ STAC error for reach {reach_id}: {e}")
                continue

        # ==================================================
        # STEP 2 — Monthly stack + normalization + occurrence
        # ==================================================

        monthly_output_folder = os.path.join(data_monthly_root, "monthly_normalized")
        os.makedirs(monthly_output_folder, exist_ok=True)

        # STACK
        stacked_path = os.path.join(data_monthly_root, f"{reach_id}_stacked_GWP1.tif")
        if not os.path.exists(stacked_path):
            gdf = gpd.read_file(shapefile_path).to_crs(4326)
            stack_geotiffs(daily_dir, stacked_path, data_monthly_root, reach_id, gdf, m, n)

        # Normalize stacked (full period)
        if not os.path.exists(normalized_tiff):
            normalize_stacked_geotiff(stacked_path, normalized_tiff,
                                      datetime(2003,1,1), datetime(2022,12,31))

        # Full water occurrence
        calculate_water_occurrence(normalized_tiff, data_monthly_root)

        # ==================================================
        # STEP 3 — Monthly normalized & unfilled/fill
        # ==================================================

        normalized_folder = os.path.join(monthly_output_folder, "normalized")
        occ_folder = os.path.join(monthly_output_folder, "water_occurrence")
        os.makedirs(normalized_folder, exist_ok=True)
        os.makedirs(occ_folder, exist_ok=True)

        for mo in range(1, 13):
            norm_file = os.path.join(normalized_folder,
                                     f"normalized_stacked_GWP1_month_{mo:02d}.tif")
            normalize_stacked_geotiff_for_month(
                stacked_geotiff_file=stacked_path,
                normalized_geotiff_file=norm_file,
                month=mo,
                start_year=2003,
                end_year=2022
            )

            unfilled = os.path.join(occ_folder,
                                    f"water_occurrence_stacked_GWP1_month_{mo:02d}_unfilled.tif")
            calculate_water_occurrence(norm_file, occ_folder,
                                       f"water_occurrence_stacked_GWP1_month_{mo:02d}_unfilled.tif")

            # Fill unfilled using overall full occurrence
            full_occ = os.path.join(data_monthly_root, "water_occurrence.tif")
            if not os.path.exists(full_occ):
                continue

            try:
                with rioxarray.open_rasterio(unfilled) as mds:
                    mocc = mds.squeeze()
                with rioxarray.open_rasterio(full_occ) as ods:
                    occ_full = ods.squeeze()

                filled = xr.where(
                    mocc.isnull() | (mocc == 0),
                    occ_full,
                    mocc
                )
                filled_out = os.path.join(occ_folder,
                                          f"water_occurrence_stacked_GWP1_month_{mo:02d}.tif")
                filled.rio.to_raster(filled_out)

                os.remove(unfilled)
            except Exception:
                pass

        # ==================================================
        # STEP 4 — Temporal daily enhancement
        # ==================================================

        monthly_occ = load_monthly_occurrence(monthly_output_folder, reach_id)
        normalized_monthly_data = load_normalized_monthly_data(normalized_tiff)

        sample_temporal = os.path.join(output_dir, "2003-01-01_daily_gwpp1d_temporal.tif")
        if not os.path.exists(sample_temporal):
            process_daily_maps(daily_dir, monthly_occ, normalized_monthly_data, output_dir)

        # ==================================================
        # STEP 5 — Segmentation + Area analysis
        # ==================================================

        reach_len = get_reach_length(shapefile_path)
        if reach_len == 0:
            continue

        daily_temp_files = sorted(glob.glob(os.path.join(output_dir, "*_temporal.tif")))
        lambda_values = [1]
        area_by_lambda = {lam: [] for lam in lambda_values}

        for tf in daily_temp_files:

            date_str = os.path.basename(tf)[:10]

            with rasterio.open(tf) as src:
                image = src.read(1)
                profile = src.profile

            norm_img = image.astype(np.float32)
            norm_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min() + 1e-6)
            sigma = float(np.std(norm_img))

            for lam in lambda_values:
                region = binary_segmentation_with_maxflow(norm_img, lambda_param=lam, sigma=sigma)
                seg = (region * 255).astype(np.uint8)
                profile.update(dtype=rasterio.uint8)

                seg_file = os.path.join(segmented_dir, f"{date_str}_segmented_lambda{lam}.tif")
                with rasterio.open(seg_file, 'w', **profile) as dst:
                    dst.write(seg, 1)

                area, shape_wh, tr, crs = calculate_white_pixel_area(seg_file)
                area_ratio = (area/1e6) / (reach_len/1e3)
                area_by_lambda[lam].append({"Date": date_str, "Area/Length": area_ratio})

        # ==================================================
        # STEP 6 — Plot + Excel
        # ==================================================

        for lam in lambda_values:
            recs = area_by_lambda[lam]
            if not recs:
                continue

            df = pd.DataFrame(recs)
            df["Date"] = pd.to_datetime(df["Date"])

            plt.figure(figsize=(14,7))
            plt.plot(df["Date"], df["Area/Length"], label=f"λ={lam}", marker='o')
            plt.title(f"Area/Length vs Time — Reach {reach_id}")
            plt.xlabel("Date")
            plt.ylabel("Area/Length (km)")
            plt.grid(True)
            plt.tight_layout()

            plot_path = os.path.join(site_dir, f"{reach_id}_lambda_plot.png")
            plt.savefig(plot_path)
            plt.close()

            excel_path = os.path.join(site_dir, f"{reach_id}_area_per_length.xlsx")
            df.to_excel(excel_path, index=False)

print("\n✔ All datasets and sites fully processed.")
