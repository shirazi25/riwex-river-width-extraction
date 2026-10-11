# RiWEx — River Width Extraction

RiWEx is a Python workflow I developed to estimate river width from satellite and remote-sensing data. The repository contains separate workflows for:

- Landsat 8/9 imagery
- Sentinel-2 imagery
- MODIS-derived Global WaterPack data

Starting from a latitude and longitude, RiWEx finds the nearest river reach in the SWORD dataset, builds a working buffer around it, separates water from land, and derives a river-width time series. The three scripts follow the same general idea, but each one is adapted to the spatial resolution and data structure of its sensor.

## Before running the code

RiWEx is a research workflow, so the results should always be checked against the original imagery and any available reference observations. To run the workflows, you will need a Google Earth Engine account and an Earth Engine-enabled Google Cloud project.

## Requirements

- Conda or Miniconda
- A Google Earth Engine account
- An Earth Engine-enabled Google Cloud project

The three workflows have separate Conda environment files because their geospatial dependencies differ.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/shirazi25/riwex-river-width-extraction.git
cd riwex-river-width-extraction
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

The scripts read the Earth Engine project ID from a command-line option or from the `GOOGLE_CLOUD_PROJECT` environment variable.

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

Landsat and Sentinel-2 also accept `--indices` and `--lambdas`. Run a script with `--help` to see the available command-line options.

## Water indices

The Landsat and Sentinel-2 workflows can calculate five water-related spectral indices. In the equations below, `B`, `G`, `R`, `NIR`, `SWIR1`, and `SWIR2` refer to the blue, green, red, near-infrared, and two shortwave-infrared bands.

| Option | Equation used in RiWEx | Why it is useful |
| --- | --- | --- |
| `ndwi` | `(G - NIR) / (G + NIR)` | A common first choice for open-water detection. Water generally has a stronger green response and a weaker NIR response than vegetation and soil. |
| `ndmi` | `(NIR - SWIR1) / (NIR + SWIR1)` | Responds to surface and vegetation moisture. It can provide useful additional information where wet land and water boundaries are difficult to separate. |
| `awei_nsh` | `4 × (G - SWIR1) - (0.25 × NIR + 2.75 × SWIR2)` | The Automated Water Extraction Index for areas where terrain shadow is not the main problem. |
| `awei_sh` | `B + 2.5 × G - 1.5 × (NIR + SWIR1) - 0.25 × SWIR2` | The shadow-oriented AWEI form. It is useful where dark surfaces or terrain shadows may otherwise be confused with water. |
| `andwi` | `(B + G + R - NIR - SWIR1 - SWIR2) / (B + G + R + NIR + SWIR1 + SWIR2)` | Uses all six optical bands to increase the contrast between visible reflectance and infrared reflectance. |

The corresponding band mappings are handled inside each script. For Landsat 8/9, RiWEx uses bands `SR_B2` to `SR_B7`; for Sentinel-2 it uses `B2`, `B3`, `B4`, `B8`, `B11`, and `B12`.

More than one index can be tested in the same Landsat or Sentinel-2 run. For example:

```bash
python RiWEx_Sentinel2.py \
  --site "Amazon Manacapuru" \
  --latitude -3.312 \
  --longitude -60.598 \
  --indices ndwi awei_sh andwi \
  --project-id YOUR_PROJECT_ID
```

There is no single index that is best for every river. The result depends on water colour, sediment, vegetation, shadows, cloud contamination, and sensor resolution. Comparing several indices is often more informative than selecting one in advance.

## Lambda and the energy terms

After the spectral index is normalized, RiWEx separates river pixels from land pixels with a graph-cut/max-flow segmentation. The segmentation balances two parts:

- **Data energy (`e_data`)** describes how well each pixel fits the water or land class according to its normalized value.
- **Smoothness energy (`e_smooth`)** penalizes abrupt label changes between neighbouring pixels, especially when their values are similar.

The reported `flow` is the total minimum-cut energy returned by the max-flow solution. For the Landsat and Sentinel-2 workflows, these values are saved for each date, index, and lambda value in the generated energy CSV files. They are diagnostic values: they help compare parameter settings, but the lowest number should not automatically be treated as the best river map without visually checking the segmentation.

Lambda controls the strength of the smoothness term. A small lambda allows the segmentation to follow local pixel differences more closely, which can preserve narrow features but may also produce a noisy or fragmented river mask. A larger lambda favours a smoother and more connected result, but it may remove narrow channels or merge nearby features. The neighbour weight used by the Landsat and Sentinel-2 scripts is:

```text
weight = lambda × exp(-|pixel_1 - pixel_2| / sigma)
```

The MODIS workflow uses the same principle with a Gaussian form based on the squared pixel difference. Its current downstream segmentation uses `lambda = 1`. Landsat tests `0.01`, `0.1`, `1`, `5`, and `10` by default, while Sentinel-2 starts with `0.01`. You can test your own values, for example:

```bash
python RiWEx_Landsat.py \
  --site "Amazon Manacapuru" \
  --latitude -3.312 \
  --longitude -60.598 \
  --indices ndwi awei_sh \
  --lambdas 0.01 0.1 1 5 10 \
  --project-id YOUR_PROJECT_ID
```

RiWEx also introduces temporal information before segmentation. With `energy_method = "basic"`, the normalized index image is multiplied directly by the temporal water-occurrence map. The alternative method first transforms the temporal map with a logistic function controlled by `energy_alpha` and `energy_beta`. Here, alpha controls the steepness of the transition and beta sets its midpoint. This temporal step gives more support to pixels that have behaved like water in the reference time series.

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
