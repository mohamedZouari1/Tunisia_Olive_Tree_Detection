// ============================================================
// AUTO-GENERATE OLIVE DETECTION TRAINING DATA — Tunisia
// ============================================================
// Uses ESA WorldCover + NDVI filtering + Dynamic World to auto-label:
//   Class 0: Background (bare soil, urban, water, roads)
//   Class 1: Olive Extensif (sparse tree cover, moderate NDVI)
//   Class 2: Olive Intensif (dense tree cover, high NDVI)
//
// Exports BOTH the 10-band Sentinel-2 image AND the label mask
// for multiple training regions across Tunisia.
// ============================================================

// ---- TRAINING REGIONS (diverse olive zones) ----
var regions = {
  // Sfax: Tunisia's olive capital (extensif, widely spaced)
  sfax_north: ee.Geometry.Rectangle([10.55, 34.75, 10.75, 34.90]),
  sfax_south: ee.Geometry.Rectangle([10.60, 34.55, 10.80, 34.70]),
  
  // Sahel: Sousse/Monastir (dense olive orchards)
  sahel:      ee.Geometry.Rectangle([10.45, 35.72, 10.65, 35.85]),
  
  // Kairouan: mixed agriculture + olives
  kairouan:   ee.Geometry.Rectangle([9.95, 35.65, 10.15, 35.78]),
  
  // Mahdia: coastal olives
  mahdia:     ee.Geometry.Rectangle([10.90, 35.40, 11.10, 35.55]),
  
  // ---- NEGATIVE SAMPLES (no olives) ----
  // Cities
  tunis_city:   ee.Geometry.Rectangle([10.10, 36.78, 10.25, 36.88]),
  sousse_city:  ee.Geometry.Rectangle([10.58, 35.80, 10.65, 35.85]),
  sfax_city:    ee.Geometry.Rectangle([10.72, 34.72, 10.80, 34.78]),
  
  // Desert / bare soil
  desert_south: ee.Geometry.Rectangle([8.50, 33.80, 8.70, 33.95]),
  sahara_dunes: ee.Geometry.Rectangle([7.80, 33.40, 8.00, 33.55]),
  
  // Salt lakes (Chott)
  chott_jerid:  ee.Geometry.Rectangle([8.20, 33.90, 8.40, 34.05]),
  
  // Forest (not olives — cork oak, pine)
  forest_ain_draham: ee.Geometry.Rectangle([8.65, 36.75, 8.80, 36.85]),
  
  // Irrigated crops (wheat, tomatoes — NOT olives)
  medjez_crops: ee.Geometry.Rectangle([9.55, 36.60, 9.70, 36.72]),
  
  // Sea / coast
  coast_djerba: ee.Geometry.Rectangle([10.85, 33.75, 11.00, 33.85]),
};

// ---- CONFIG ----
var YEAR = 2023;
var startDate = YEAR + '-06-01';
var endDate   = YEAR + '-09-30';
var EXPORT_SCALE = 10; // meters

// ---- SENTINEL-2 COMPOSITE ----
function getS2Composite(roi) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
    .map(function(img) {
      // Cloud mask using SCL band
      var scl = img.select('SCL');
      var clearMask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
      return img.updateMask(clearMask);
    })
    .median()
    .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
    .divide(10000)  // Scale to 0-1 reflectance
    .clip(roi);
}

// ---- AUTO-GENERATE LABELS ----
function generateLabels(roi) {
  // 1. ESA WorldCover 2021 (10m resolution!)
  var worldcover = ee.ImageCollection('ESA/WorldCover/v200')
    .first()
    .clip(roi);
  
  // WorldCover classes:
  // 10 = Tree cover, 20 = Shrubland, 30 = Grassland,
  // 40 = Cropland, 50 = Built-up, 60 = Bare/sparse,
  // 80 = Water, 90 = Wetland
  
  // 2. NDVI from Sentinel-2 (summer = dry season, olives stay green)
  var s2 = getS2Composite(roi);
  var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  // 3. Olive detection rules:
  // Extensif olives: sparse tree/cropland with moderate NDVI (0.1-0.35)
  //   - In Sfax, olives are widely spaced on sandy soil
  //   - WorldCover sees them as cropland (40) or sparse vegetation
  var isCropOrTree = worldcover.eq(40).or(worldcover.eq(10));
  var isModerateNDVI = ndvi.gte(0.10).and(ndvi.lt(0.35));
  var extensif = isCropOrTree.and(isModerateNDVI);
  
  // Intensif olives: dense tree cover with higher NDVI (0.35-0.60)
  //   - Modern irrigated orchards, dense canopy
  var isDenseVeg = worldcover.eq(10).or(worldcover.eq(20));
  var isHighNDVI = ndvi.gte(0.35).and(ndvi.lt(0.60));
  var intensif = isDenseVeg.and(isHighNDVI);
  
  // 4. Negative classes (force to background):
  var isUrban = worldcover.eq(50);        // Built-up
  var isWater = worldcover.eq(80);        // Water
  var isBare  = worldcover.eq(60);        // Bare/sparse vegetation
  var isBackground = isUrban.or(isWater).or(isBare);
  
  // Remove very high NDVI (likely irrigated crops, not olives)
  var isCrop = ndvi.gte(0.60);  // Green crops, not olives
  
  // 5. Build label mask: 0=bg, 1=extensif, 2=intensif
  var labels = ee.Image(0)
    .where(extensif.and(isBackground.not()).and(isCrop.not()), 1)
    .where(intensif.and(isBackground.not()).and(isCrop.not()), 2)
    .where(isBackground, 0)
    .rename('label')
    .toByte()
    .clip(roi);
  
  return labels;
}

// ---- EXPORT ALL REGIONS ----
var regionNames = Object.keys(regions);
for (var i = 0; i < regionNames.length; i++) {
  var name = regionNames[i];
  var roi = regions[name];
  
  var s2 = getS2Composite(roi);
  var labels = generateLabels(roi);
  
  // Export Sentinel-2 image
  Export.image.toDrive({
    image: s2.toFloat(),
    description: 'S2_' + name,
    folder: 'olive_training_v2',
    region: roi,
    scale: EXPORT_SCALE,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });
  
  // Export label mask
  Export.image.toDrive({
    image: labels,
    description: 'labels_' + name,
    folder: 'olive_training_v2',
    region: roi,
    scale: EXPORT_SCALE,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });
}

// ---- VISUALIZE ONE REGION (Sfax) ----
var vizRoi = regions.sfax_north;
var s2viz = getS2Composite(vizRoi);
var labelsViz = generateLabels(vizRoi);

Map.centerObject(vizRoi, 13);
Map.addLayer(s2viz, {bands: ['B4','B3','B2'], min: 0, max: 0.3}, 'RGB');
Map.addLayer(labelsViz, {min: 0, max: 2, palette: ['black','green','orange']}, 'Auto Labels');

// NDVI layer
var ndviViz = s2viz.normalizedDifference(['B8','B4']);
Map.addLayer(ndviViz, {min: 0, max: 0.5, palette: ['brown','yellow','green']}, 'NDVI', false);

print('✅ Run ALL tasks in the Tasks tab (14 exports: 7 images + 7 labels)');
print('📁 Files will appear in Google Drive → olive_training_v2/');
print('⬇ Download all files, put in learning_ai folder, then run:');
print('   python chip_training_v2.py');
