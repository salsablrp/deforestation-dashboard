# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import ee.ee_exception
import os
import json

app = Flask(__name__)
CORS(app)

# --- ROBUST GEE AUTHENTICATION ---
try:
    print("Attempting to initialize Earth Engine...")
    creds_json_str = os.environ.get('GEE_CREDENTIALS')
    if not creds_json_str:
        raise ValueError("GEE_CREDENTIALS environment variable not found.")
    
    print(f"Found GEE_CREDENTIALS environment variable. Length: {len(creds_json_str)}")
    creds_json = json.loads(creds_json_str)
    credentials = ee.ServiceAccountCredentials(creds_json['client_email'], key_data=creds_json['private_key'])
    ee.Initialize(credentials=credentials, project='ee-salsabilarp')
    print("SUCCESS: Earth Engine initialized using GEE_CREDENTIALS.")

except Exception as e:
    print(f"FATAL: Could not initialize Earth Engine. Error details: {e}")

# --- UTILS with Error Handling ---
def get_composite(year, aoi):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 9, 30)
    
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start, end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    # Create the median composite from the collection
    composite = collection.median().select(['B2', 'B3', 'B4', 'B8'])
        
    # **NEW, MORE ROBUST CHECK**: Verify that the final composite image has bands.
    # This is more reliable than checking the initial collection size.
    band_names = composite.bandNames().getInfo()
    if not band_names:
        raise ee.ee_exception.EEException(f"Could not create a cloud-free composite for {year}. All available images may be masked or invalid for this region. Please try a different year.")
        
    return composite.clip(aoi)

def get_ndvi(img):
    return img.normalizedDifference(['B8', 'B4']).rename('nd')

def get_evi(img):
    return img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': img.select('B8'), 'RED': img.select('B4'), 'BLUE': img.select('B2')}
    ).rename('nd')

# --- API ENDPOINT ---
@app.route('/analyze', methods=['POST'])
def analyze_deforestation():
    try:
        ee.Number(1).getInfo() # Quick check to see if GEE is usable
    except Exception:
        return jsonify({"error": "Earth Engine client library not initialized on the server. Check server logs."}), 500

    try:
        data = request.json
        country = data['country']
        province = data['province']
        start_year = int(data['startYear'])
        end_year = int(data['endYear'])
        method = data.get('method', 'NDVI')

        gaul = ee.FeatureCollection('FAO/GAUL/2015/level1')
        aoi = gaul.filter(ee.Filter.And(
            ee.Filter.eq('ADM0_NAME', country),
            ee.Filter.eq('ADM1_NAME', province)
        )).geometry()

        # --- Images & indices ---
        image1 = get_composite(start_year, aoi)
        image2 = get_composite(end_year, aoi)
        index1 = get_ndvi(image1) if method == 'NDVI' else get_evi(image1)
        index2 = get_ndvi(image2) if method == 'NDVI' else get_evi(image2)
        change = index2.subtract(index1)

        forest_threshold = 0.5
        forest1 = index1.gt(forest_threshold)
        forest2 = index2.gt(forest_threshold)
        deforested = forest1.And(forest2.Not()).selfMask()

        # --- MAP TILES (always created first) ---
        deforested_map = deforested.getMapId({'palette': 'red', 'min': 0, 'max': 1})
        change_map = change.getMapId({'min': -0.5, 'max': 0.5, 'palette': ['red','white','green']})
        training = image2.sample(region=aoi, scale=5000, numPixels=2000, seed=1)
        clusterer = ee.Clusterer.wekaKMeans(3).train(training)
        classified = image2.cluster(clusterer)
        classified_map = classified.getMapId({'min': 0, 'max': 2, 'palette': ['green','yellow','brown']})

        # --- AOI boundary simplified ---
        aoi_geojson = aoi.simplify(5000).getInfo()

        # --- STATISTICS (wrapped in try/except for robustness) ---
        deforested_area, min_index = 0, 0
        try:
            pixel_area = ee.Image.pixelArea().divide(1e6)
            stats_image = deforested.multiply(pixel_area).addBands(change)
            stats_info = stats_image.reduceRegion(
                reducer=ee.Reducer.sum().combine(ee.Reducer.min(), '', True),
                geometry=aoi,
                scale=5000,
                maxPixels=1e8,
                bestEffort=True
            ).getInfo()
            deforested_area = stats_info.get('nd_sum', 0)
            min_index = stats_info.get('nd_min', 0)
        except Exception as e:
            # If stats fail, we can still return the maps
            print(f"⚠️ Stats computation failed (but maps will be returned): {e}")

        return jsonify({
            'tiles': {
                'deforested': deforested_map['tile_fetcher'].url_format,
                'change': change_map['tile_fetcher'].url_format,
                'clusters': classified_map['tile_fetcher'].url_format
            },
            'stats': {
                'deforestedAreaKm2': round(deforested_area or 0, 2),
                'maxIndexLoss': round(min_index or 0, 3)
            },
            'aoi': aoi_geojson
        })

    except ee.ee_exception.EEException as e:
        print(f"GEE Error: {e}")
        return jsonify({"error": f"GEE Error: {e}"}), 500
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

