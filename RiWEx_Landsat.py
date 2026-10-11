import ee
import geemap
import xarray as xr
import xee
import netCDF4
import numpy as np
import pandas as pd
import maxflow
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
import geopandas as gpd
import rioxarray
import os
import time
import urllib.request
from pyproj import Geod
import geopandas as gpd
import rasterio
from rasterio import features
import argparse
import warnings
from datetime import datetime


warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))

# Configuration
CONFIG = {
    # Edit these values to process a different location, or override them
    # with the command-line options documented in README.md.
    "site": "example_site",
    "latitude": -3.312,
    "longitude": -60.598,
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "output_path": os.path.join(HERE, "results_landsat"),
    "water_indices": ["ndwi", "ndmi", "awei_sh", "awei_nsh", "andwi"],
    "reach_type": ["single_reach"],
    "segment_snic": 'Y', # 'Y'/'N'
    "lambda_values": [1.0],
    "buffer_coef": 1,
    "energy_method": "basic",
    "energy_alpha": 1.0,
    "energy_beta": 0.5,
    "jrc_threshold": 0,
    "dem_percentile": 70,
    "resolution": 30,
    # Use --project-id or set GOOGLE_CLOUD_PROJECT. Never commit credentials.
    "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
    "ee_url": "https://earthengine-highvolume.googleapis.com",
}

def log_time(start_time, section_name):
    """Log elapsed time for a section and return current time."""
    elapsed = time.time() - start_time
    print(f"Section '{section_name}' took {elapsed:.2f} seconds")
    return time.time()


def validate_date_range():
    """Validate the ISO study dates; end_date is exclusive."""
    try:
        start = datetime.fromisoformat(CONFIG["start_date"])
        end = datetime.fromisoformat(CONFIG["end_date"])
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD format.") from exc
    if start >= end:
        raise ValueError("start_date must be earlier than end_date.")


def initialize_earth_engine():
    """Initialize Google Earth Engine with authentication."""
    if not CONFIG["project_id"]:
        raise ValueError(
            "An Earth Engine-enabled Google Cloud project is required. "
            "Pass --project-id YOUR_PROJECT_ID or set GOOGLE_CLOUD_PROJECT."
        )
    try:
        ee.Authenticate()
        ee.Initialize(project=CONFIG["project_id"], opt_url=CONFIG["ee_url"])
    except ee.EEException as e:
        raise RuntimeError(f"Failed to initialize Earth Engine: {e}") from e

def find_reach(reach_collection, reach_id=None, lon=None, lat=None):
    """Find a river reach by ID or coordinates with incremental buffering."""
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
            nearest = filtered.map(
                lambda f: f.set("distance_to_point", f.geometry().distance(point))
            ).sort("distance_to_point").first()
            reach_id_test = nearest.get("reach_id").getInfo()
            if reach_id_test:
                print(f"Found reach_id: {reach_id_test}, width: {nearest.get('width').getInfo()}")
                return nearest, start_time
        buffer_size += 1000
    raise ValueError(f"No reach found near ({lon}, {lat}) within {max_buffer}m")

def create_buffer_zone(feature, buffer_coef):
    """Create a buffer zone around a river reach based on width and variance."""
    width = ee.Number.parse(feature.get("width"))
    variance = ee.Number.parse(feature.get("width_var")).sqrt()
    buffer_size = ee.Number(buffer_coef).multiply(variance).add(width)
    return feature.geometry().buffer(buffer_size)

def calculate_reach_length(feature, reach_type, reaches):
    """Calculate river length for single or multiple reaches."""
    start_time = time.time()
    if reach_type == "single_reach":
        length = ee.Number.parse(feature.get("reach_len")).divide(1000)
    elif reach_type == "multiple_reach":
        up_id = ee.Number.parse(feature.get("rch_id_up"))
        dn_id = ee.Number.parse(feature.get("rch_id_dn"))
        up_river = reaches.filter(ee.Filter.eq("reach_id", up_id)).first()
        dn_river = reaches.filter(ee.Filter.eq("reach_id", dn_id)).first()
        current_len = ee.Number.parse(feature.get("reach_len"))
        up_len = ee.Number.parse(up_river.get("reach_len"))
        dn_len = ee.Number(0 if dn_river.getInfo() is None else dn_river.get("reach_len"))
        length = current_len.add(up_len).add(dn_len).divide(1000)
    return length.getInfo(), start_time

def save_map_visualization(site, lon, lat, reach_buffer, reach_geometry, save_path):
    """Save a map visualization as a PNG file."""
    start_time = time.time()
    gauge_point = ee.Feature(ee.Geometry.Point([lon, lat]), {"name": site})
    try:
        gauge_buffer = gauge_point.geometry().buffer(100)
        zoomed_region = reach_buffer.buffer(5000)
        vis_image = (ee.Image().paint(reach_buffer, 1, 2).visualize(palette=["red"])
                     .blend(ee.Image().paint(reach_geometry, 1, 2).visualize(palette=["blue"]))
                     .blend(ee.Image().paint(gauge_buffer, 1).visualize(palette=["yellow"])))
        satellite = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                    .filterBounds(reach_buffer)
                    .filterDate(CONFIG["start_date"], CONFIG["end_date"])
                    .sort("CLOUD_COVER")
                    .first()
                    .select(["SR_B4", "SR_B3", "SR_B2"])
                    .multiply(2.75e-05).add(-0.2))
        final_image = satellite.visualize(bands=["SR_B4", "SR_B3", "SR_B2"], min=0, max=0.3).blend(vis_image)
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

def bitwise_extract(value, from_bit, to_bit=None):
    """Extract bits from an Earth Engine image for cloud masking."""
    to_bit = from_bit if to_bit is None else to_bit
    mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
    mask = ee.Number(1).leftShift(mask_size).subtract(1)
    return value.rightShift(from_bit).bitwiseAnd(mask)

def landsat_cloud_mask(img, reach_buffer, buffer_area, topo_condition):
    """Apply cloud masking and compute water indices for a Landsat image."""
    date = img.date().format("YYYY-MM-dd")
    sr = img.select("SR_B.*").multiply(2.75e-05).add(-0.2)
    qa = img.select("QA_PIXEL")
    cirrus = bitwise_extract(qa, 2)
    cloud = bitwise_extract(qa, 3)
    shadow = bitwise_extract(qa, 4)
    mask = cirrus.eq(1).Or(cloud.eq(1)).Or(shadow.eq(1))
    mask_area = mask.multiply(ee.Image.pixelArea().divide(1e6))
    mask_area_in_buffer = ee.Number(
        mask_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=reach_buffer,
            scale=CONFIG["resolution"]
        ).values().get(0)
    )
    buffer_cloud_percent = mask_area_in_buffer.divide(buffer_area).multiply(100.0)
    results = {}
    for w_index in CONFIG["water_indices"]:
        if w_index == "ndwi":
            result = (sr.normalizedDifference(["SR_B3", "SR_B5"])
                      .updateMask(mask.Not()).updateMask(topo_condition).rename(w_index)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", buffer_cloud_percent))
        elif w_index == "ndmi":
            result = (sr.normalizedDifference(["SR_B5", "SR_B6"])
                      .updateMask(mask.Not()).updateMask(topo_condition).rename(w_index)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", buffer_cloud_percent))
        elif w_index == "awei_nsh":
            result = (sr.expression("4 * (G - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)", {
                "G": sr.select("SR_B3"), "NIR": sr.select("SR_B5"),
                "SWIR1": sr.select("SR_B6"), "SWIR2": sr.select("SR_B7")
            }).updateMask(mask.Not()).updateMask(topo_condition).rename(w_index)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", buffer_cloud_percent))
        elif w_index == "awei_sh":
            result = (sr.expression("B + 2.5 * G - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2", {
                "B": sr.select("SR_B2"), "G": sr.select("SR_B3"), "NIR": sr.select("SR_B5"),
                "SWIR1": sr.select("SR_B6"), "SWIR2": sr.select("SR_B7")
            }).updateMask(mask.Not()).updateMask(topo_condition).rename(w_index)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", buffer_cloud_percent))
        elif w_index == "andwi":
            result = (sr.expression("(B + G + R - NIR - SWIR1 - SWIR2)/(B + G + R + NIR + SWIR1 + SWIR2)", {
                "B": sr.select("SR_B2"), "G": sr.select("SR_B3"), "R": sr.select("SR_B4"),
                "NIR": sr.select("SR_B5"), "SWIR1": sr.select("SR_B6"), "SWIR2": sr.select("SR_B7")
            }).updateMask(mask.Not()).updateMask(topo_condition).rename(w_index)
                      .set("system:time_start", ee.Date(date).millis())
                      .set("cloud_percent_clipped_region", buffer_cloud_percent))
        results[w_index] = result
    return results

def get_dem_and_jrc(reach_buffer, feature):
    """Load DEM and JRC data for topographic filtering."""
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
        ######## remove JRC, because we need non water pixel as well.
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        jrc = jrc.where(jrc.gt(CONFIG["jrc_threshold"]), 1).where(jrc.lte(CONFIG["jrc_threshold"]), 2)
        jrc = jrc.eq(1).selfMask()
        ######### topo condition: if width less than 70m,  then only JRC

        topo_condition = fabdem.lt(fabdem_per).Or(slope.lt(.1)).And(jrc)
        jrc_condition = jrc

        feature_width = ee.Number.parse(feature.get("width"))

        print(f'feather_width:{feature_width.getInfo()}')

        result = ee.Algorithms.If(
            feature_width.lt(70),
            jrc_condition,
            topo_condition
        )

        return result, start_time


    except Exception as e:
        print(f"Error in get_dem_and_jrc: {e}")
        raise


def nodata_percent(img, reach_buffer, buffer_area):
  nodata = img.eq(0).selfMask()
  nodata_area = nodata.multiply(ee.Image.pixelArea().divide(1e6))
  nodata_sum = ee.Number(nodata_area.reduceRegion(
      reducer = ee.Reducer.sum(),
      geometry = reach_buffer,
      scale = 30,
      maxPixels = 1e13
  ).values().get(0))
  nodata_percent = (nodata_sum.divide(buffer_area)).multiply(100.0).round()
  return img.selfMask().copyProperties(img, img.propertyNames()).set('nodata_per', ee.Number(nodata_percent))



def process_jrc_collection(reach_buffer, buffer_area):
    start_time = time.time()
    jrc = (
    ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
    .filterDate('1984','2013')
    .filterBounds(reach_buffer)
    .map(lambda img: img.clip(reach_buffer).copyProperties(img, img.propertyNames()))
)
    jrc_nodata = jrc.map(
        lambda img: nodata_percent(img, reach_buffer, buffer_area).copyProperties(img, img.propertyNames())
    ).filter(ee.Filter.lt('nodata_per',50))

    return jrc_nodata, start_time

###############################################################################

def process_landsat_data(reach_buffer, buffer_area, topo_condition):
    """Process Landsat imagery with cloud masking and segmentation."""
    start_time = time.time()
    try:
        landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        landsat = (landsat8.merge(landsat9)
                   .filterBounds(reach_buffer)
                   .filterDate(CONFIG["start_date"], CONFIG["end_date"])
                   .sort("system:time_start"))
        results_by_index = {}
        for w_index in CONFIG["water_indices"]:
            def map_cloud_mask(img):
                results = landsat_cloud_mask(img, reach_buffer, buffer_area, topo_condition)
                return results[w_index]
            landsat_mapped = landsat.map(map_cloud_mask)
            if landsat_mapped.size().getInfo() == 0:
                print(f"No Landsat images available for index {w_index} after cloud filtering")
                results_by_index[w_index] = None
                continue
            landsat_mapped = landsat_mapped.filter(ee.Filter.lt("cloud_percent_clipped_region", 10))
            def segmentation(img):
              # make snic as a optional parameter


              if CONFIG['segment_snic'] == 'Y':
                seg = ee.Algorithms.Image.Segmentation.SNIC(image=img, size=15, compactness=0.5, connectivity=8).select('.*_mean')
              else:
                seg = img

              return seg.rename(w_index).copyProperties(img, img.propertyNames())
            landsat_segmentation = landsat_mapped.map(segmentation)
            results_by_index[w_index] = landsat_segmentation
        return results_by_index, start_time
    except Exception as e:
        print(f"Error in process_landsat_data: {e}")
        raise

def add_temporal_dependency(target_map, temporal_map, method, alpha = CONFIG["energy_alpha"], beta=CONFIG["energy_alpha"]):
    """Add temporal dependency to normalized image data."""
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
    normalized_image, foreground_seeds=None, background_seeds=None, lambda_param=1.0,
    site=None, w_index=None, time_index=None, time_coord=None
):
    """Perform binary segmentation using max-flow algorithm and return energy terms."""
    if normalized_image.max() > 1.0 or normalized_image.min() < 0.0:
        raise ValueError("Input image must be normalized between 0 and 1")
    sigma = np.std(normalized_image)
    height, width = normalized_image.shape
    foreground_seeds = foreground_seeds or []
    background_seeds = background_seeds or []
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((height, width))
    smooth_image = gaussian_filter(normalized_image, sigma=sigma)
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
                g.add_tedge(nodeids[y, x], bg_weight[y, x], fg_weight[y, x])
    flow = g.maxflow()
    segmentation = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if g.get_segment(nodeids[y, x]) == 0:
                segmentation[y, x] = 1  # River pixels = 1
            else:
                segmentation[y, x] = 0  # Land pixels = 0

    # Calculate e_data and e_smooth
    river_val = 1  # River pixels in segmentation
    land_val = 0   # Land pixels in segmentation
    river = segmentation * fg_weight  # fg_weight = normalized_image for river
    land = (1 - segmentation) * bg_weight  # bg_weight = 1 - normalized_image for land
    e_data = np.sum(river[segmentation == river_val]) + np.sum(land[segmentation == land_val])
    e_smooth = flow - e_data

    # Return energy terms
    energy_terms = {
        "time_index": time_index,
        "time": time_coord,
        "lambda": lambda_param,
        "e_data": float(e_data),
        "e_smooth": float(e_smooth),
        "flow": float(flow)
    }
    return segmentation, energy_terms

def normalize_spatial_dims(dataset):
    """Use the lon/lat names expected by the processing code across xee versions."""
    rename_map = {}
    if "x" in dataset.dims and "lon" not in dataset.dims:
        rename_map["x"] = "lon"
    if "y" in dataset.dims and "lat" not in dataset.dims:
        rename_map["y"] = "lat"
    return dataset.rename(rename_map) if rename_map else dataset


def process_local_data(landsat_segmentation, jrc_col, reach_buffer, reach_buffer_cor, site, site_save_path):
    """Process local data with normalization and segmentation, saving energy terms in one CSV per index."""
    start_time = time.time()
    results_by_index_lambda = {}
    try:
        for w_index in CONFIG["water_indices"]:
            results_by_index_lambda[w_index] = {}
            energy_data_list = []  # Collect energy terms for this index
            jrc_energy_data_list = []
            landsat_data = landsat_segmentation.get(w_index)
            if landsat_data is None:
                for lambda_val in CONFIG["lambda_values"]:
                    results_by_index_lambda[w_index][lambda_val] = None
                continue
            ds = geemap.ee_to_xarray(
                landsat_data, crs="EPSG:4326", scale=0.0003, geometry=reach_buffer
            )
            ds = normalize_spatial_dims(ds)

            # add single jrc occurrence as a xarray
            ###############################################

            jrc_img = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
            jrc_arr = geemap.ee_to_xarray(
                jrc_img,
                crs = 'EPSG:4326',
                scale = 0.0003,
                geometry = reach_buffer
            )
            jrc_arr = normalize_spatial_dims(jrc_arr)

            ds_sub = ds.sel(
                time=slice(CONFIG["start_date"], CONFIG["end_date"])
            ).mean(dim='time')
            jrc_arr = jrc_arr.interp_like(ds_sub, method = 'nearest').isel(time = 0).drop_vars('time')

            fig0, ax0 = plt.subplots(figsize = (5,5))
            plot0 = jrc_arr.occurrence.plot(
                x = 'lon',
                y = 'lat',
                robust = True,
                ax = ax0
            )
            print(f'{site_save_path}/01_jrc_occ.png')
            plt.savefig(f'{site_save_path}/01_jrc_occ.png', dpi = 360, bbox_inches = 'tight')
            plt.close(fig0)


            river_mask_occ = (jrc_arr.occurrence >= 90).astype(int).transpose("lat", "lon").values
            non_river_mask_occ = (jrc_arr.occurrence < 10).astype(int).transpose("lat", "lon").values

            fig1, ax1 = plt.subplots(figsize = (5,5))
            plot1 = (jrc_arr.occurrence >= 90).astype(int).plot(
                x = 'lon',
                y = 'lat',
                robust = True,
                ax = ax1
            )


            plt.savefig(f'{site_save_path}/02_river_mask.png', dpi = 360, bbox_inches = 'tight')
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize = (5,5))
            plot2 = (jrc_arr.occurrence < 10).astype(int).plot(
                x = 'lon',
                y = 'lat',
                ax = ax2
            )

            plt.savefig(f'{site_save_path}/03_river_nan_mask.png', dpi = 360, bbox_inches = 'tight')
            plt.close(fig2)

            print(f"Dataset dimensions for {w_index}: {ds.dims}")
            time_coords = pd.to_datetime(ds.time.values)
            ds_index = ds[w_index]
            jrc_occ = ee.ImageCollection("JRC/GSW1_4/MonthlyRecurrence").select(['monthly_recurrence'],['occ'])
            ds_occ = geemap.ee_to_xarray(
               jrc_occ, crs='EPSG:4326', scale=0.0003, geometry=reach_buffer
            )
            ds_occ = normalize_spatial_dims(ds_occ)
            ds_occ = ds_occ.interp(lon=ds.lon, lat=ds.lat, method="nearest")
            ds_occ_normal = (ds_occ / 100.0)
            ds_occ_mean = ds_occ_normal.median('time')
            ds_occ_fill = ds_occ_normal.fillna(0)
            ds_occ_fill = xr.where(ds_occ_fill == 0, ds_occ_mean, ds_occ_fill).fillna(0)
            ds_rolling = ds_index.rolling(time=8, center=True, min_periods=1).mean("time")
            ds_filled = ds_index.fillna(ds_rolling)
            def normalize(time_slice):
                return (time_slice - time_slice.min()) / (time_slice.max() - time_slice.min())
            normalized_ds = ds_filled.groupby("time").apply(normalize)
            ds_norm_filled = normalized_ds.fillna(0)
            ds_norm_filled2 = ds_norm_filled.assign_coords(month = ds_norm_filled.time.dt.month -1)

            ####### replace ds_norm_filled2 with 1 where JRC >= 0.98

            # ds_norm_filled2 = ds_norm_filled2.where(jrc_arr.occurrence <= 98, 1)


            ds_norm_filled2 = xr.where(jrc_arr.occurrence >= 90, 1, ds_norm_filled2)


            ds_plot3 = ds_norm_filled2.sel(time=slice(CONFIG["start_date"], CONFIG["end_date"]))
            ds_plot3 = ds_plot3.drop_duplicates(dim='time')

            fig3 = ds_plot3.plot(
            x='lon',
            y='lat',
            col='time',         # one subplot per time slice
            col_wrap=10         # number of columns per ro
            )

            plt.savefig(f'{site_save_path}/04_index_norm.png', dpi=360, bbox_inches='tight')
            plt.close(fig3.fig)



            reach_load = gpd.read_file('reach.shp')


            reach_geom = reach_load.geometry

            ds_sub_dim = ds_sub.rio.set_spatial_dims(x_dim = 'lon', y_dim ='lat', inplace = True)

            reach_raster = features.rasterize(
            reach_geom,
            out_shape = ds_sub_dim.rio.shape,
            transform = ds_sub_dim.rio.transform())


            reach_raster_arr = xr.DataArray(
            reach_raster,
            coords = {'lat': ds_sub_dim.lat, 'lon': ds_sub_dim.lon},
            dims = ('lat','lon') )

            reach_raster_arr = reach_raster_arr.transpose('lon','lat')


            fig4 = reach_raster_arr.plot(
            x='lon',
            y='lat'
            )

            plt.savefig(f'{site_save_path}/05_reach_raster.png', dpi=360, bbox_inches='tight')
            plt.close()


            ds_norm_filled2 = xr.where(reach_raster_arr == 1, 1, ds_norm_filled2)
            ds_norm_filled2 = ds_norm_filled2.transpose("time", "lat", "lon")

            ds_plot5 = ds_norm_filled2.sel(time=slice(CONFIG["start_date"], CONFIG["end_date"]))
            ds_plot5 = ds_plot5.drop_duplicates(dim='time')

            fig5 = ds_plot5.plot(
            x='lon',
            y='lat',
            col='time',
            col_wrap=10
            )

            plt.savefig(f'{site_save_path}/06_index_norm_filled_by_reach.png', dpi=360, bbox_inches='tight')
            plt.close(fig5.fig)

            #######

            for lambda_val in CONFIG["lambda_values"]:
                seg_result = []
                td_result = []
                for t in range(ds_norm_filled2.sizes["time"]):
                    index_slice = ds_norm_filled2.isel(time=t).transpose("lat", "lon").values
                    if index_slice.shape != (ds_norm_filled2.sizes["lat"], ds_norm_filled2.sizes["lon"]):
                        raise ValueError(f"Time slice shape mismatch: {index_slice.shape}")
                    month_index = ds_norm_filled2.isel(time = t).month.values
                    foreground_indices = np.argwhere(river_mask_occ)
                    foreground_seeds = [tuple(idx) for idx in foreground_indices]
                    background_indices = np.argwhere(non_river_mask_occ)
                    background_seeds = [tuple(idx) for idx in background_indices]
                    num_fg = min(20, len(foreground_seeds))
                    num_bg = min(20, len(background_seeds))
                    foreground_seeds_sample = random.sample(foreground_seeds, num_fg) if num_fg > 0 else []
                    background_seeds_sample = random.sample(background_seeds, num_bg) if num_bg > 0 else []
                    occ_slice = ds_occ_fill.isel(time=month_index).occ.transpose("lat", "lon").values
                    td_map = add_temporal_dependency(index_slice, occ_slice, method = CONFIG['energy_method'])

                    td_result.append(td_map)

                    bn_seg, energy_terms = binary_segmentation_with_maxflow(
                        td_map, foreground_seeds_sample, background_seeds_sample, lambda_val,
                        site=site, w_index=w_index, time_index=t, time_coord=str(time_coords[t])
                    )

                    seg_result.append(bn_seg)
                    energy_data_list.append(energy_terms)
                results_np = np.stack(seg_result, axis=0)
                td_stack = np.stack(td_result, axis = 0)



                if results_np.shape[1:] != (ds_norm_filled.sizes["lat"], ds_norm_filled.sizes["lon"]):
                    raise ValueError(f"Results shape mismatch: {results_np.shape}")



                results_arr = xr.DataArray(
                    data=results_np,
                    dims=["time", "lat", "lon"],
                    coords={"time": time_coords, "lat": ds.lat, "lon": ds.lon},
                    name=f"segmentation_{w_index}_lambda_{lambda_val}"
                )

                td_arr = xr.DataArray(
                    data=td_stack,
                    dims=["time", "lat", "lon"],
                    coords={"time": time_coords, "lat": ds.lat, "lon": ds.lon},
                    name=f"segmentation_{w_index}_lambda_{lambda_val}"
                )

                ds_plot6 = td_arr.sel(time=slice(CONFIG["start_date"], CONFIG["end_date"]))
                ds_plot6 = ds_plot6.drop_duplicates(dim='time')

                fig6 = ds_plot6.plot(
                    x='lon',
                    y='lat',
                    col='time',
                    col_wrap=10
                    )

                plt.savefig(f'{site_save_path}/07_temporal_dependency.png', dpi=360, bbox_inches='tight')
                plt.close(fig6.fig)

                ds_plot7 = results_arr.sel(time=slice(CONFIG["start_date"], CONFIG["end_date"]))
                ds_plot7 = ds_plot7.drop_duplicates(dim='time')

                fig7 = ds_plot7.plot(
                    x='lon',
                    y='lat',
                    col='time',
                    col_wrap=10
                    )

                plt.savefig(f'{site_save_path}/08_maxflow_result.png', dpi=360, bbox_inches='tight')
                plt.close(fig7.fig)


                seg_arr = results_arr

                seg_arr = xr.where(seg_arr == 1, np.nan, seg_arr)
                seg_arr = seg_arr.rio.write_crs('EPSG:4326', inplace = True)
                seg_arr = seg_arr.rio.set_spatial_dims(x_dim = 'lon', y_dim = 'lat', inplace = True)
                seg_arr.rio.write_coordinate_system(inplace = True)
                seg_arr_clip = seg_arr.rio.clip(reach_buffer_cor.geometry, reach_buffer_cor.crs)

                ds_jrc = geemap.ee_to_xarray(
                    jrc_col,
                    crs='EPSG:4326',
                    geometry=reach_buffer,
                    scale=0.0003,
                )
                ds_jrc = normalize_spatial_dims(ds_jrc)

                jrc_time_coords = pd.to_datetime(ds_jrc.time.values)

                ds_jrc_months = ds_jrc.groupby('time.month').median('time') #mean or median
                ds_jrc_fill = ds_jrc.fillna(ds_jrc_months)
                ds_jrc_fill = ds_jrc_fill.mean('month')


                jrc_norm_list = []

                for t in range(ds_jrc_fill.sizes['time']):
                  jrc_min = ds_jrc_fill.isel(time = t).min().water.values
                  jrc_max = ds_jrc_fill.isel(time = t).max().water.values
                  jrc_norm = (ds_jrc_fill.isel(time = t).water.values - jrc_min) / (jrc_max - jrc_min)
                  jrc_norm_list.append(jrc_norm)


                jrc_stack = np.stack(jrc_norm_list, axis = 0)
                jrc_norm_array = xr.DataArray(
                    jrc_stack,
                    dims = ds_jrc_fill.dims,
                    coords = ds_jrc_fill.coords,
                    name = 'water'
                )

                jrc_array = xr.where(jrc_norm_array == 1, 0.98,
                xr.where(jrc_norm_array == 0, 0.02, 0.02))

                jrc_array = jrc_array.assign_coords(month  = jrc_array.time.dt.month -1)


            for lambda_val in CONFIG["lambda_values"]:
                jrc_results = []
                for timestamp in range(jrc_array.sizes['time']):
                    jrc_target_slice = jrc_array.isel(time = timestamp).values
                    jrc_month_index = jrc_array.isel(time = timestamp).month.values
                    jrc_river_mask = jrc_array.isel(time = timestamp) == 0.98
                    jrc_2d = jrc_river_mask.values

                    foreground_indices = np.argwhere(jrc_2d == True)
                    foreground_seeds = [tuple(idx) for idx in foreground_indices]

                    jrc_non_river_mask = jrc_array.isel(time = t) == 0.02
                    background_indices = np.argwhere(jrc_2d == False)
                    background_seeds = [tuple(idx) for idx in background_indices]

                    num_fg = min(20, len(foreground_seeds))
                    num_bg = min(20, len(background_seeds))
                    foreground_seeds_sample = random.sample(foreground_seeds, num_fg) if num_fg > 0 else []
                    background_seeds_sample = random.sample(background_seeds, num_bg) if num_bg > 0 else []

                    temp_slice = ds_occ_fill.isel(time = jrc_month_index).occ.values

                    jrc_td_map = add_temporal_dependency(jrc_target_slice, temp_slice, method = CONFIG['energy_method'])
                    jrc_time_coords = pd.to_datetime(ds_jrc.time.values)


                    jrc_seg, energy_terms = binary_segmentation_with_maxflow(
                        jrc_td_map, foreground_seeds_sample, background_seeds_sample, lambda_val,
                        site=site, w_index=None, time_index=t, time_coord=str(jrc_time_coords[t])
                    )

                    jrc_results.append(jrc_seg)
                    jrc_energy_data_list.append(energy_terms)


                jrc_results_np = np.stack(jrc_results, axis = 0)
                jrc_results_arr = xr.DataArray(
                      data = jrc_results_np,
                      dims = jrc_array.dims,
                      coords = jrc_array.coords,
                      name=f"segmentation_{w_index}_lambda_{lambda_val}"
                  )


                jrc_seg_arr = jrc_results_arr
                jrc_seg_arr = xr.where(jrc_seg_arr == 1, np.nan, jrc_seg_arr)
                jrc_seg_arr = jrc_seg_arr.rio.write_crs('EPSG:4326', inplace = True)
                jrc_seg_arr = jrc_seg_arr.rio.set_spatial_dims(x_dim = 'lon', y_dim = 'lat', inplace = True)
                jrc_seg_arr.rio.write_coordinate_system(inplace = True)
                jrc_seg_arr_clip = jrc_seg_arr.rio.clip(reach_buffer_cor.geometry, reach_buffer_cor.crs)
                jrc_seg_arr_clip = jrc_seg_arr_clip.drop_vars('month')

                seg_concat = xr.concat([jrc_seg_arr_clip,seg_arr_clip], dim = 'time')
                results_by_index_lambda[w_index][lambda_val] = seg_concat

            if energy_data_list:
                energy_df = pd.DataFrame(energy_data_list)
                pivoted_data = {}
                pivoted_data["time"] = time_coords
                for lambda_val in CONFIG["lambda_values"]:
                    lambda_data = energy_df[energy_df["lambda"] == lambda_val]
                    pivoted_data[f"e_data_lambda_{lambda_val}"] = lambda_data["e_data"].values
                    pivoted_data[f"e_smooth_lambda_{lambda_val}"] = lambda_data["e_smooth"].values
                    pivoted_data[f"flow_lambda_{lambda_val}"] = lambda_data["flow"].values
                energy_df_pivoted = pd.DataFrame(pivoted_data)
                energy_csv_path = os.path.join(site_save_path, f"{site}_energy_{w_index}.csv")


                jrc_energy_df = pd.DataFrame(jrc_energy_data_list)
                jrc_pivoted_data = {}
                jrc_pivoted_data["time"] = jrc_time_coords
                for lambda_val in CONFIG["lambda_values"]:
                    lambda_data = jrc_energy_df[jrc_energy_df["lambda"] == lambda_val]
                    jrc_pivoted_data[f"e_data_lambda_{lambda_val}"] = lambda_data["e_data"].values
                    jrc_pivoted_data[f"e_smooth_lambda_{lambda_val}"] = lambda_data["e_smooth"].values
                    jrc_pivoted_data[f"flow_lambda_{lambda_val}"] = lambda_data["flow"].values
                jrc_energy_df_pivoted = pd.DataFrame(jrc_pivoted_data)
                jrc_energy_csv_path = os.path.join(site_save_path, f"{site}_jrc_energy_{w_index}.csv")

                merge_energy_csv_path = os.path.join(site_save_path, f"{site}_energy_{w_index}.csv")
                energy_out = pd.concat([jrc_energy_df_pivoted, energy_df_pivoted])
                energy_out.to_csv(merge_energy_csv_path, index = False)

        return results_by_index_lambda, start_time
    except Exception as e:
        print(f"Error in process_local_data: {e}")
        raise


def movAve_new(values, window_size):
    """Computes moving average with a sliding window centered at each point."""
    half_window = window_size // 2
    padded = np.pad(values, (half_window, half_window), mode='edge')
    moving_avg = np.array([np.mean(padded[i:i+window_size]) for i in range(len(values))])
    return moving_avg

def blunder_detection(df, window_size):
    """
    Detects and removes blunders from a time series DataFrame with
    one or MORE numeric columns (e.g. width_lambda_0.01, width_lambda_0.1, ...).

    Algorithm:
      * For each numeric column:
          - compute moving average
          - residuals = value - moving_avg
          - standardize residuals (subtract mean, divide by std)
      * Find the largest standardized residual over ALL numeric columns.
      * If this max > 3, drop that ROW (same timestamp) from the DataFrame.
      * Repeat until no residual exceeds 3σ or the series is too short.

    Args
    ----
    df : pandas.DataFrame
        Must contain a time column plus one or more numeric columns.
    window_size : int
        Window size for moving average (should be odd).

    Returns
    -------
    df_clean : pandas.DataFrame
        Same columns as df, but with some rows removed (blunders).
    """
    df_clean = df.copy()

    # All numeric columns (your width_lambda_* columns)
    value_columns = df_clean.select_dtypes(include=[np.number]).columns
    if len(value_columns) == 0:
        raise ValueError("No numeric column found for blunder detection.")

    # Iteratively remove outliers
    while True:
        # If series is already very short, stop
        if len(df_clean) <= window_size:
            break

        max_z_global = 0.0
        max_idx_global = None

        # Check each numeric column
        for col in value_columns:
            values = df_clean[col].values

            # Moving average
            moving_avg = movAve_new(values, window_size)

            # Residuals
            residuals = values - moving_avg
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)

            # If std is zero, skip this column (no variability)
            if std_res == 0:
                continue

            # Standardized residuals
            z = np.abs((residuals - mean_res) / std_res)

            # Find worst residual in this column
            idx_local = np.argmax(z)
            z_local = z[idx_local]

            # Keep track of the worst over ALL columns
            if z_local > max_z_global:
                max_z_global = z_local
                max_idx_global = idx_local

        # If no column had std>0 OR no residual > 3σ, stop
        if max_idx_global is None or max_z_global <= 3.0:
            break

        # Otherwise drop the row with the worst residual (all columns)
        df_clean = df_clean.drop(df_clean.index[max_idx_global]).reset_index(drop=True)

    return df_clean






def calculate_river_width(results_by_index_lambda, river_length, site, site_save_path, lat, lon, reach, buffer_area):
    """Calculate river widths from segmentation results and save to CSV."""
    start_time = time.time()
    results_by_index = {}
    try:
        geod = Geod(ellps="WGS84")
        sample_index = next(
            (w for w in CONFIG["water_indices"] if results_by_index_lambda[w].get(CONFIG["lambda_values"][0]) is not None),
            None
        )
        if sample_index is None:
            print(f"No valid data for site {site}, skipping river width calculation")
            return {}, start_time
        lon_coords = results_by_index_lambda[sample_index][CONFIG["lambda_values"][0]].coords["lon"].values
        lat_coords = results_by_index_lambda[sample_index][CONFIG["lambda_values"][0]].coords["lat"].values
        scale = 0.0003
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        lon_left, lon_right = lon_grid - scale / 2, lon_grid + scale / 2
        lat_bottom, lat_top = lat_grid - scale / 2, lat_grid + scale / 2
        cell_areas = np.zeros_like(lon_grid)
        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                lons = [lon_left[i, j], lon_right[i, j], lon_right[i, j], lon_left[i, j]]
                lats = [lat_bottom[i, j], lat_bottom[i, j], lat_top[i, j], lat_top[i, j]]
                cell_areas[i, j] = abs(geod.polygon_area_perimeter(lons, lats)[0])
        cell_areas_da = xr.DataArray(
            cell_areas, dims=["lat", "lon"], coords={"lat": lat_coords, "lon": lon_coords}
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
                results_arr = results_by_index_lambda[w_index].get(lambda_val)
                if results_arr is None:
                    print(f"No data for {w_index} with lambda {lambda_val}, skipping")
                    width_dict[f"width_lambda_{lambda_val}"] = (
                        [np.nan] * width_dict["time"].size if width_dict["time"] is not None else []
                    )
                    continue
                times = results_arr.time.values
                cluster_area = ((results_arr == 0) * cell_areas_da).sum(dim=["lon", "lat"])  # River pixels = 0
                river_width = cluster_area / (river_length * 1000)
                width_df_lambda = river_width.to_dataframe(name=f"width_lambda_{lambda_val}")
                width_dict[f"width_lambda_{lambda_val}"] = width_df_lambda[f"width_lambda_{lambda_val}"].values
                if width_dict["time"] is None:
                    width_dict["time"] = times
            if width_dict["time"] is not None:
                width_df = pd.DataFrame(width_dict)
                width_df_smooth = blunder_detection(width_df, 3)
                width_da = xr.DataArray(
                    data=width_df_smooth[[f"width_lambda_{lv}" for lv in CONFIG["lambda_values"]]].values,
                    dims=["time", "lambda"],
                    coords={"time": width_df_smooth["time"], "lambda": CONFIG["lambda_values"]},
                    name=f"width_{w_index}"
                )
                results_by_index[w_index] = width_da
                width_csv_path = os.path.join(site_save_path, f"{site}_widths_{w_index}.csv")
                with open(width_csv_path, "w") as f:
                    f.write("\n".join(metadata))
                width_df_smooth.to_csv(width_csv_path, mode="a", index=False)
                print(f"Saved widths to {width_csv_path}")
            else:
                print(f"No valid data for {w_index}, skipping CSV save")
                results_by_index[w_index] = None
        return results_by_index, start_time
    except Exception as e:
        print(f"Error in calculate_river_width: {e}")
        raise

def main(site, latitude, longitude):
    """Process one river location supplied directly by the user."""
    start_time = time.time()
    validate_date_range()
    os.makedirs(CONFIG["output_path"], exist_ok=True)
    initialize_earth_engine()
    reaches = ee.FeatureCollection("projects/sat-io/open-datasets/SWORD/reaches_merged")

    site = str(site)
    lat, lon = float(latitude), float(longitude)
    print(f"\nProcessing site: {site} (lat: {lat}, lon: {lon})")
    site_name = site.replace(" ", "_").replace("/", "_")
    period_name = f"{CONFIG['start_date']}_to_{CONFIG['end_date']}"
    site_save_path = os.path.join(CONFIG["output_path"], site_name, period_name)
    os.makedirs(site_save_path, exist_ok=True)

    reach, start_time = find_reach(reaches, lon=lon, lat=lat)
    reach_buffer = create_buffer_zone(reach, CONFIG["buffer_coef"])
    reach_buffer = ee.Geometry(reach_buffer).simplify(100)
    buffer_area = reach_buffer.area().divide(1e6).getInfo()
    log_time(start_time, "Reach Identification")
    print(f"Buffer area (km²): {buffer_area}")

    reach_col = ee.FeatureCollection([reach])
    reach_path = os.path.join(site_save_path, "reach.shp")
    geemap.ee_to_shp(reach_col, reach_path)
    reach_gpd = gpd.read_file(reach_path)
    reach_buf_cor = reach_gpd.buffer(0.008, cap_style='flat')

    river_length, start_time = calculate_reach_length(reach, "single_reach", reaches)
    log_time(start_time, "River Length Calculation")
    print(f"River length (km): {river_length}")

    save_map_visualization(site, lon, lat, reach_buffer, reach.geometry(), site_save_path)
    topo_condition, start_time = get_dem_and_jrc(reach_buffer, reach)
    log_time(start_time, "DEM and JRC Data Loading")
    jrc_col, start_time = process_jrc_collection(reach_buffer, buffer_area)
    log_time(start_time, "JRC monthly collection prepared")
    landsat_segmentation, start_time = process_landsat_data(reach_buffer, buffer_area, topo_condition)
    log_time(start_time, "Landsat Data Processing")
    if all(v is None for v in landsat_segmentation.values()):
        raise RuntimeError(f"No valid Landsat data found for {site}")

    results_by_index_lambda, start_time = process_local_data(
        landsat_segmentation, jrc_col, reach_buffer, reach_buf_cor, site, site_save_path
    )
    log_time(start_time, "Local Data Processing")
    calculate_river_width(
        results_by_index_lambda, river_length, site, site_save_path, lat, lon, reach, buffer_area
    )
    log_time(start_time, "River Width Calculation")

    log_time(start_time, "Total Processing")
    print(f"\nProcessing complete. Results saved in {CONFIG['output_path']}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run RiWEx Landsat for one location")
    parser.add_argument("--site", default=CONFIG["site"], help="A label for the site")
    parser.add_argument("--latitude", type=float, default=CONFIG["latitude"])
    parser.add_argument("--longitude", type=float, default=CONFIG["longitude"])
    parser.add_argument("--start-date", default=CONFIG["start_date"], help="Inclusive date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=CONFIG["end_date"], help="Exclusive date (YYYY-MM-DD)")
    parser.add_argument(
        "--project-id",
        default=CONFIG["project_id"],
        help="Your Earth Engine-enabled Google Cloud project ID",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        default=CONFIG["water_indices"]
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=CONFIG["lambda_values"]
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["project_id"] = args.project_id
    CONFIG["start_date"] = args.start_date
    CONFIG["end_date"] = args.end_date
    CONFIG["water_indices"] = args.indices
    CONFIG["lambda_values"] = args.lambdas
    main(args.site, args.latitude, args.longitude)
