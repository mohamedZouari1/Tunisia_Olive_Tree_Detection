// ============================================================
// Export Sentinel-2 for Olive Detection — Tunisia
// ============================================================
// 1. Draw a rectangle on the map or change the coordinates below
// 2. Click "Run"
// 3. Go to Tasks tab → click "Run" on the export task
// 4. Download from Google Drive → upload to http://localhost:5000
// ============================================================

// ---- CHANGE THIS: your area of interest ----
// Example: Sfax region (olive capital of Tunisia)
var roi = ee.Geometry.Rectangle([10.05, 34.70, 10.25, 34.85]);

// ---- CHANGE THIS: date range (dry season = best for olives) ----
var startDate = '2024-06-01';
var endDate   = '2024-09-30';

// ---- Get cloud-free Sentinel-2 composite ----
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .median()
    .clip(roi);

// ---- Select the 10 bands (model auto-computes NDVI/NDWI/EVI/SAVI) ----
var bands = s2.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']);

// ---- Scale to 0-1 reflectance ----
var scaled = bands.divide(10000);

// ---- Preview on map ----
Map.centerObject(roi, 13);
Map.addLayer(scaled, {bands: ['B4','B3','B2'], min: 0, max: 0.3}, 'RGB');
Map.addLayer(roi, {color: 'yellow'}, 'Export Area');

// ---- Export to Google Drive ----
Export.image.toDrive({
    image: scaled,
    description: 'S2_olive_test',
    folder: 'olive_detection',
    region: roi,
    scale: 10,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
});

print('✅ Click RUN, then go to Tasks tab to start the export');
print('📥 Download from Google Drive → upload to localhost:5000');
