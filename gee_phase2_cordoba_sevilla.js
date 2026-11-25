/*
 * ============================================================
 * ZitounEye - Phase 2: Download Cordoba + Sevilla
 * ============================================================
 *
 * HOW TO USE:
 *   1. Go to https://code.earthengine.google.com
 *   2. Paste this ENTIRE script
 *   3. Click "Run"
 *   4. Go to the "Tasks" tab (top-right panel)
 *   5. Click "RUN ALL" (there will be 8 tasks: 2 provinces x 4 seasons)
 *   6. Wait 4-12 hours for all exports to finish
 *   7. Download ALL GeoTIFFs from Google Drive folder "ZitounEye_Phase2"
 *   8. Put them in: c:\Users\AORUS\Documents\zitoun_eye\spain_phase2_images\
 *
 * WHY THESE 2 PROVINCES?
 *   - Cordoba: huge olive zone + diverse arable crops
 *   - Sevilla: mix of olives + palms + vineyards + rice fields
 *     (This will teach the model to NOT confuse palms with olives!)
 *
 * WHAT YOU GET:
 *   - 8 GeoTIFF files (4 seasons x 2 provinces)
 *   - 14 bands per file: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,NDVI,EVI,NDWI,SAVI
 *   - 10m resolution, cloud-free composites
 *   - Same format as Jaen imagery (plug & play into training)
 * ============================================================
 */

// ============================================================
// SETTINGS
// ============================================================
var YEAR        = 2023;
var MAX_CLOUD   = 30;
var SCALE       = 10;
var CRS         = 'EPSG:32630';   // UTM Zone 30N
var DRIVE_FOLDER = 'ZitounEye_Phase2';

// ============================================================
// PROVINCES
// ============================================================
var PROVINCES = {
  'es_co_2023': {
    name: 'Cordoba',
    bbox: ee.Geometry.Rectangle([-5.65, 37.24, -4.07, 38.73])
  },
  'es_se_2023': {
    name: 'Sevilla',
    bbox: ee.Geometry.Rectangle([-6.54, 36.84, -4.86, 38.02])
  }
};

// ============================================================
// SEASONS
// ============================================================
var SEASONS = {
  'spring': { start: YEAR + '-03-01', end: YEAR + '-05-31' },
  'summer': { start: YEAR + '-06-01', end: YEAR + '-08-31' },
  'fall':   { start: YEAR + '-09-01', end: YEAR + '-11-30' },
  'winter': { start: YEAR + '-12-01', end: (YEAR+1) + '-02-28' }
};

// ============================================================
// FUNCTIONS
// ============================================================

// Cloud masking using SCL band (Sentinel-2 L2A)
function maskClouds(image) {
  var scl = image.select('SCL');
  var clearMask = scl.eq(4)    // Vegetation
    .or(scl.eq(5))             // Not vegetated
    .or(scl.eq(6))             // Water
    .or(scl.eq(11));           // Snow
  return image.updateMask(clearMask);
}

// Add vegetation indices
function addIndices(image) {
  var B4  = image.select('B4');   // Red
  var B8  = image.select('B8');   // NIR
  var B3  = image.select('B3');   // Green
  var B2  = image.select('B2');   // Blue
  var B11 = image.select('B11');  // SWIR1

  var NDVI = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI');
  var EVI  = B8.subtract(B4)
               .divide(B8.add(B4.multiply(6)).subtract(B2.multiply(7.5)).add(1))
               .multiply(2.5).rename('EVI');
  var NDWI = B3.subtract(B8).divide(B3.add(B8)).rename('NDWI');
  var SAVI = B8.subtract(B4).divide(B8.add(B4).add(0.5)).multiply(1.5).rename('SAVI');

  return image.addBands([NDVI, EVI, NDWI, SAVI]);
}

// Get cloud-free seasonal composite for a bbox
function getSeasonalComposite(bbox, startDate, endDate) {
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(bbox)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD))
    .map(maskClouds)
    .map(addIndices);

  var composite = collection.median();

  var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
               'NDVI','EVI','NDWI','SAVI'];
  return composite.select(bands);
}

// ============================================================
// EXPORT ALL TASKS
// ============================================================
var taskCount = 0;

Object.keys(PROVINCES).forEach(function(provinceKey) {
  var prov = PROVINCES[provinceKey];

  Object.keys(SEASONS).forEach(function(seasonName) {
    var season = SEASONS[seasonName];

    var composite = getSeasonalComposite(prov.bbox, season.start, season.end);
    var taskName  = provinceKey + '_' + seasonName + '_' + YEAR;

    Export.image.toDrive({
      image:          composite,
      description:    taskName,
      folder:         DRIVE_FOLDER,
      fileNamePrefix: taskName,
      region:         prov.bbox,
      scale:          SCALE,
      crs:            CRS,
      maxPixels:      1e12,
      fileFormat:     'GeoTIFF'
    });

    taskCount++;
    print('Queued: ' + taskName + ' [' + prov.name + ' - ' + seasonName + ']');
  });
});

// ============================================================
// SUMMARY
// ============================================================
print('');
print('============================================');
print('ZitounEye Phase 2 - Export Summary');
print('============================================');
print('Provinces: Cordoba + Sevilla');
print('Seasons:   spring, summer, fall, winter');
print('Bands:     14 per file (10 spectral + 4 indices)');
print('Scale:     ' + SCALE + 'm resolution');
print('Folder:    Google Drive -> ' + DRIVE_FOLDER);
print('');
print('Total tasks queued: ' + taskCount);
print('');
print('>>> Go to TASKS tab and click RUN on all ' + taskCount + ' tasks <<<');
print('');
print('WHY SEVILLA? It has palm groves!');
print('The model will learn palm != olive in Phase 2.');
print('============================================');
