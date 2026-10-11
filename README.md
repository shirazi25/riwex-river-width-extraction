# RiWEx — River Width Extraction

RiWEx is a Python workflow for estimating river width from satellite and remote-sensing products. It includes separate workflows for:

- Landsat 8/9 imagery
- Sentinel-2 imagery
- MODIS-derived Global WaterPack data

The workflows locate the nearest river reach in the SWORD dataset, prepare a reach buffer, perform water segmentation, and save maps and river-width products for one user-supplied location.

## Important notes

- RiWEx is intended for research and educational use. Validate its outputs before using them for scientific or operational decisions.
- You need your own Google Earth Engine account and an Earth Engine-enabled Google Cloud project.
- No Earth Engine project ID, password, access token, or service-account key is included in this repository.
- A CSV input file is not required. Define one site directly in the script or pass its coordinates on the command line.

## Requirements

- Conda or Miniconda
- A Google Earth Engine account
- An Earth Engine-enabled Google Cloud project

The three workflows have separate Conda environment files because their geospatial dependencies differ.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/YOUR_USERNAME/riwex.git
cd riwex
```

Create the environment for the sensor you want to use:

```bash
# Landsat
conda env create -f riwex_landsat.yml
conda activate riwex_landsat

# Sentinel-2
conda env create -f riwex_sentinel.yml
conda activate riwex_sentinel

# MODIS
conda env create -f riwex_modis.yml
conda activate riwex_modis
```

Authenticate Earth Engine once:

```bash
earthengine authenticate
```

## Configure a location

Each script contains the following public example values near the top:

```python
"site": "example_site",
"latitude": -3.312,
"longitude": -60.598,
```

Replace them with the name and WGS84 coordinates of the location you want to process. Latitude must be between `-90` and `90`; longitude must be between `-180` and `180`.

You can also leave the files unchanged and pass the location as command-line arguments.

## Supply your Earth Engine project ID

Choose one of these methods. Do not put passwords, tokens, or downloaded service-account keys in the repository.

Pass the ID for a single run:

```bash
python RiWEx_Sentinel2.py --project-id YOUR_PROJECT_ID
```

Or set the standard environment variable first:

```bash
# PowerShell
$env:GOOGLE_CLOUD_PROJECT = "YOUR_PROJECT_ID"

# macOS/Linux
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
```

The project ID identifies the Google Cloud project used for Earth Engine quota and billing configuration. Authentication is handled separately by `earthengine authenticate`.

## Run a workflow

Use the values stored in the script:

```bash
python RiWEx_Landsat.py --project-id YOUR_PROJECT_ID
python RiWEx_Sentinel2.py --project-id YOUR_PROJECT_ID
python RiWEx_MODIS.py --project-id YOUR_PROJECT_ID
```

Override the stored location without editing the file:

```bash
python RiWEx_Sentinel2.py \
  --site "Amazon Manacapuru" \
  --latitude -3.312 \
  --longitude -60.598 \
  --project-id YOUR_PROJECT_ID
```

Landsat and Sentinel-2 also accept processing parameters such as `--indices` and `--lambdas`. Run a script with `--help` to see its current options.

## Outputs

Generated files are written to sensor-specific directories beside the scripts:

- `results_landsat/`
- `results_sentinel2/`
- `results_modis/`

Depending on the workflow and available imagery, outputs can include reach shapefiles, GeoTIFF rasters, NetCDF data, map images, diagnostic tables, and river-width time series. These generated directories are excluded from Git.

## Repository structure

```text
RiWEx_Landsat.py      Landsat 8/9 workflow
RiWEx_Sentinel2.py    Sentinel-2 workflow
RiWEx_MODIS.py        MODIS/Global WaterPack workflow
riwex_landsat.yml     Landsat Conda environment
riwex_sentinel.yml    Sentinel-2 Conda environment
riwex_modis.yml       MODIS Conda environment
```

## Data sources

RiWEx uses public remote-sensing and geospatial datasets made available through services including Google Earth Engine, SWORD, and the EOC Geoservice STAC API. Users are responsible for following the terms and citation requirements of every upstream dataset they use.

## License

The source code is available under the [MIT License](LICENSE).
