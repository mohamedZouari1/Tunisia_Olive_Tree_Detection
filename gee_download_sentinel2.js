/*
 * ============================================================
 * ZitounEye — Sentinel-2 Download for ALL Training Regions
 * ============================================================
 * 
 * HOW TO USE:
 *   1. Go to https://code.earthengine.google.com
 *   2. Paste this ENTIRE script
 *   3. Click "Run"
 *   4. Go to the "Tasks" tab (top-right)
 *   5. Click "RUN" on EVERY task (there will be 28 tasks: 7 provinces × 4 seasons)
 *   6. Wait 6-24 hours for all exports to finish
 *   7. Download ALL GeoTIFFs from Google Drive folder "ZitounEye_S2"
 *   8. Organize them into folders:
 *        data/imagery/es_j_2023/     (Jaén files)
 *        data/imagery/es_co_2023/    (Córdoba files)
 *        data/imagery/es_se_2023/    (Sevilla files)
 *        data/imagery/es_gr_2023/    (Granada files)
 *        data/imagery/es_ma_2023/    (Málaga files)
 *        data/imagery/es_ca_2023/    (Cádiz files)
 *        data/imagery/es_h_2023/     (Huelva files)
 *
 * WHAT YOU GET:
 *   • 4 seasonal GeoTIFFs per province (spring, summer, fall, winter)
 *   • 14 bands per file: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,NDVI,EVI,NDWI,SAVI
 *   • 10m resolution, cloud-free median composites
 *   • UTM Zone 30N projection (EPSG:32630)
 *
 * ============================================================
 */

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SETTINGS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

var YEAR = 2023;
var MAX_CLOUD = 30;        // Max cloud % per scene
var SCALE = 10;            // Meters per pixel
var CRS = 'EPSG:32630';   // UTM zone 30N for Spain
var DRIVE_FOLDER = 'ZitounEye_S2';


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  ALL 7 PROVINCE BOUNDING BOXES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  These cover the entire olive heartland of Andalusia, Spain.
//
//  TRAINING SET (5 provinces):
//    Jaén      — World capital of olive oil, >66 million olive trees
//    Córdoba   — Mix of olive groves + arable land
//    Sevilla   — Irrigated olives + vineyards + arable
//    Granada   — Mountainous terrain, terraced olives
//    Málaga    — Coastal + inland olives, subtropical mix
//
//  VALIDATION SET (1 province):
//    Cádiz     — Mixed agriculture, good diversity
//
//  TEST SET (1 province):
//    Huelva    — Western Andalusia, different landscape

var PROVINCES = {
  // ─── TRAINING ─────────────────────────────────────────────
  'es_j_2023': {
    name: 'Jaén',
    role: 'TRAIN',
    bbox: ee.Geometry.Rectangle([-4.12, 37.39, -2.50, 38.59])
  },
  'es_co_2023': {
    name: 'Córdoba',
    role: 'TRAIN',
    bbox: ee.Geometry.Rectangle([-5.65, 37.24, -4.07, 38.73])
  },
  'es_se_2023': {
    name: 'Sevilla',
    role: 'TRAIN',
    bbox: ee.Geometry.Rectangle([-6.54, 36.84, -4.86, 38.02])
  },
  'es_gr_2023': {
    name: 'Granada',
    role: 'TRAIN',
    bbox: ee.Geometry.Rectangle([-4.28, 36.69, -2.63, 37.95])
  },
  'es_ma_2023': {
    name: 'Málaga',
    role: 'TRAIN',
    bbox: ee.Geometry.Rectangle([-5.61, 36.31, -3.77, 37.18])
  },
  // ─── VALIDATION ───────────────────────────────────────────
  'es_ca_2023': {
    name: 'Cádiz',
    role: 'VAL',
    bbox: ee.Geometry.Rectangle([-6.44, 35.78, -5.07, 37.04])
  },
  // ─── TEST ─────────────────────────────────────────────────
  'es_h_2023': {
    name: 'Huelva',
    role: 'TEST',
    bbox: ee.Geometry.Rectangle([-7.52, 37.07, -6.18, 38.06])
  }
};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  CLOUD MASKING — using Scene Classification Layer (SCL)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function maskClouds(image) {
  var scl = image.select('SCL');
  var clear = scl.eq(2)    // Dark area
    .or(scl.eq(4))         // Vegetation
    .or(scl.eq(5))         // Bare soil
    .or(scl.eq(6))         // Water
    .or(scl.eq(7));        // Unclassified
  return image.updateMask(clear);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SPECTRAL INDICES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
//  NDVI — Olive stays 0.3-0.5 year-round (evergreen!)
//         Vineyard/almond drop to <0.2 in winter.
//         This is THE key feature for olive detection.
//
//  EVI  — Better than NDVI for dense canopy (intensive olive groves)
//  NDWI — Separates irrigated vs rainfed olives
//  SAVI — Critical for sparse traditional groves (visible soil between trees)

function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4'),
      'BLUE': image.select('B2')
    }).rename('EVI');
  
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  
  var savi = image.expression(
    '1.5 * ((NIR - RED) / (NIR + RED + 0.5))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4')
    }).rename('SAVI');
  
  return image.addBands([ndvi, evi, ndwi, savi]);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  COMPOSITE BUILDER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function getComposite(startDate, endDate, aoi) {
  var spectralBands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
  var indexBands = ['NDVI', 'EVI', 'NDWI', 'SAVI'];
  var allBands = spectralBands.concat(indexBands);
  
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filterBounds(aoi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD))
    .map(maskClouds)
    .map(addIndices);
  
  var composite = collection.select(allBands).median();
  
  // Scale reflectance from 0-10000 → 0-1
  var scaled = composite.select(spectralBands).divide(10000);
  var indices = composite.select(indexBands);
  
  return scaled.addBands(indices);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SEASON DEFINITIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
//  SPRING (Mar–May):  Olive flowering, everything green
//  SUMMER (Jun–Aug):  Peak biomass, irrigation effects visible
//  FALL   (Sep–Nov):  Olive fruit ripening, harvest
//  WINTER (Dec–Feb):  KEY SEASON! Olives stay green while
//                     vineyard/almond go bare. Maximum separation.

var SEASONS = {
  'spring': [YEAR + '-03-01', YEAR + '-06-01'],
  'summer': [YEAR + '-06-01', YEAR + '-09-01'],
  'fall':   [YEAR + '-09-01', YEAR + '-12-01'],
  'winter': [(YEAR-1) + '-12-01', YEAR + '-03-01']
};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  EXPORT ALL PROVINCES × ALL SEASONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

var provinceKeys = Object.keys(PROVINCES);
var seasonKeys = Object.keys(SEASONS);

var taskCount = 0;

for (var p = 0; p < provinceKeys.length; p++) {
  var provKey = provinceKeys[p];
  var prov = PROVINCES[provKey];
  
  for (var s = 0; s < seasonKeys.length; s++) {
    var seasonName = seasonKeys[s];
    var dates = SEASONS[seasonName];
    
    var composite = getComposite(dates[0], dates[1], prov.bbox);
    
    var taskName = provKey + '_' + seasonName + '_' + YEAR;
    
    Export.image.toDrive({
      image: composite.toFloat(),
      description: taskName,
      folder: DRIVE_FOLDER,
      fileNamePrefix: taskName,
      region: prov.bbox,
      scale: SCALE,
      crs: CRS,
      maxPixels: 1e13,
      fileFormat: 'GeoTIFF'
    });
    
    taskCount++;
  }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  VISUALIZATION — check data quality before downloading
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Center on Jaén
var jaenAOI = PROVINCES['es_j_2023'].bbox;
Map.centerObject(jaenAOI, 8);

// Show all province boundaries
for (var p2 = 0; p2 < provinceKeys.length; p2++) {
  var pk = provinceKeys[p2];
  var pr = PROVINCES[pk];
  var color = pr.role === 'TRAIN' ? 'green' : (pr.role === 'VAL' ? 'orange' : 'red');
  Map.addLayer(pr.bbox, {color: color}, pk + ' (' + pr.role + ')', true, 0.3);
}

// Show Jaén summer composite
var jaenSummer = getComposite(YEAR + '-06-01', YEAR + '-09-01', jaenAOI);
var jaenWinter = getComposite((YEAR-1) + '-12-01', YEAR + '-03-01', jaenAOI);

var rgbVis = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3};
var ndviVis = {min: -0.1, max: 0.8, palette: ['brown', 'yellow', 'green', 'darkgreen']};

Map.addLayer(jaenSummer, rgbVis, 'Jaén Summer RGB', true);
Map.addLayer(jaenWinter, rgbVis, 'Jaén Winter RGB', false);
Map.addLayer(jaenSummer.select('NDVI'), ndviVis, 'Jaén Summer NDVI', false);
Map.addLayer(jaenWinter.select('NDVI'), ndviVis, 'Jaén Winter NDVI', false);

// NDVI seasonality map — low diff = evergreen = likely olive
var ndviDiff = jaenSummer.select('NDVI').subtract(jaenWinter.select('NDVI'));
Map.addLayer(ndviDiff, 
  {min: -0.1, max: 0.5, palette: ['darkgreen', 'green', 'yellow', 'orange', 'red']},
  'NDVI Seasonality (green=evergreen=olive)', false);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  NDVI TIME SERIES CHART — verify olive phenology
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Click any point on the map to change the location.
//  Expected patterns:
//    Olive:    ~~~~ flat NDVI ~0.35-0.50 all year (evergreen)
//    Vineyard: /\   peaks in summer, drops to ~0.10 in winter
//    Wheat:    /\_  peak in spring, crashes after harvest in June

var samplePoint = ee.Geometry.Point([-3.79, 37.77]); // Jaén olive grove

var ndviTS = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(YEAR + '-01-01', YEAR + '-12-31')
  .filterBounds(samplePoint)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskClouds)
  .map(function(img) {
    return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
      .set('system:time_start', img.get('system:time_start'));
  });

print(ui.Chart.image.series(ndviTS, samplePoint, ee.Reducer.mean(), 10)
  .setOptions({
    title: 'NDVI Time Series (sample point in Jaén olive country)',
    vAxis: {title: 'NDVI', viewWindow: {min: -0.1, max: 0.9}},
    hAxis: {title: 'Date'},
    lineWidth: 2, pointSize: 4,
    series: {0: {color: '#228B22'}}
  }));


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SUMMARY
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
print('🫒 ZitounEye — Sentinel-2 Export for ALL Regions');
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
print('');
print('TRAINING provinces:');
print('  • es_j_2023  — Jaén (olive capital of the world)');
print('  • es_co_2023 — Córdoba');
print('  • es_se_2023 — Sevilla');
print('  • es_gr_2023 — Granada');
print('  • es_ma_2023 — Málaga');
print('');
print('VALIDATION province:');
print('  • es_ca_2023 — Cádiz');
print('');
print('TEST province:');
print('  • es_h_2023  — Huelva');
print('');
print('Total export tasks: ' + taskCount);
print('Seasons: spring, summer, fall, winter');
print('Bands per file: 14 (10 spectral + NDVI + EVI + NDWI + SAVI)');
print('Resolution: ' + SCALE + 'm');
print('');
print('👉 Go to TASKS tab (top-right) and click RUN on all ' + taskCount + ' tasks');
print('👉 Files will appear in Google Drive: ' + DRIVE_FOLDER);
print('');
print('After download, organize into:');
print('  data/imagery/es_j_2023/es_j_2023_spring_2023.tif');
print('  data/imagery/es_j_2023/es_j_2023_summer_2023.tif');
print('  data/imagery/es_j_2023/es_j_2023_fall_2023.tif');
print('  data/imagery/es_j_2023/es_j_2023_winter_2023.tif');
print('  ... (same for all 7 provinces)');
