# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import ee.ee_exception

app = Flask(__name__)
CORS(app)

# --- GEE AUTHENTICATION ---
try:
    ee.Initialize(project='ee-salsabilarp')
    print("GEE Initialized using service account.")
except ee.EEException as e:
    print(f"Error initializing GEE: {e}")

# --- UTILS ---
def get_composite(year, aoi):
    start = ee.Date.fromYMD(year, 6, 1)
    end = ee.Date.fromYMD(year, 9, 30)
    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start, end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median() \
        .select(['B2', 'B3', 'B4', 'B8']) \
        .clip(aoi)

def get_ndvi(img):
    return img.normalizedDifference(['B8', 'B4']).rename('nd')

def get_evi(img):
    return img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': img.select('B8'),
            'RED': img.select('B4'),
            'BLUE': img.select('B2')
        }
    ).rename('nd')

# --- API ENDPOINT ---
@app.route('/analyze', methods=['POST'])
def analyze_deforestation():
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
        aoi_geojson = aoi.getInfo()

        # Composites
        image1 = get_composite(start_year, aoi)
        image2 = get_composite(end_year, aoi)

        # Index
        index1 = get_ndvi(image1) if method == 'NDVI' else get_evi(image1)
        index2 = get_ndvi(image2) if method == 'NDVI' else get_evi(image2)

        # Change map
        change = index2.subtract(index1)
        change_map = change.getMapId({'min': -0.5, 'max': 0.5, 'palette': ['red','white','green']})
        change_tile = change_map['tile_fetcher'].url_format

        # Forest mask + deforested
        forest_threshold = 0.5
        forest1 = index1.gt(forest_threshold)
        forest2 = index2.gt(forest_threshold)
        deforested = forest1.And(forest2.Not()).selfMask()

        pixel_area = ee.Image.pixelArea().divide(1e6)  # km²
        deforested_area_stat = deforested.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=500,
            maxPixels=1e9,
            bestEffort=True
        )
        deforested_area = deforested_area_stat.get('nd').getInfo() or 0

        deforested_map = deforested.getMapId({'palette': 'red', 'min': 0, 'max': 1})
        deforested_tile = deforested_map['tile_fetcher'].url_format

        # Max NDVI/EVI loss
        min_index = change.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=aoi,
            scale=500,
            bestEffort=True
        ).get('nd').getInfo() or 0

        # Unsupervised classification (on end-year composite)
        training = image2.sample(region=aoi, scale=500, numPixels=5000, seed=1)
        clusterer = ee.Clusterer.wekaKMeans(3).train(training)
        classified = image2.cluster(clusterer)
        classified_map = classified.getMapId({'min': 0, 'max': 2, 'palette': ['green','yellow','brown']})
        classified_tile = classified_map['tile_fetcher'].url_format

        return jsonify({
            'tiles': {
                'deforested': deforested_tile,
                'change': change_tile,
                'clusters': classified_tile
            },
            'stats': {
                'deforestedAreaKm2': round(deforested_area, 2),
                'maxIndexLoss': round(min_index, 3)
            },
            'aoi': aoi_geojson
        })

    except ee.ee_exception.EEException as e:
        print(f"GEE Error: {e}")
        return jsonify({"error": f"Google Earth Engine Error: {e}"}), 500
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
