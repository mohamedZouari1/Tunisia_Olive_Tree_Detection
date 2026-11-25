# ZitounEye - What Are We Doing and Why?

## The Big Picture

We are building an **AI model that can look at a satellite image and tell you: "this area is an olive grove."**

Think of it like teaching a child to recognize olives on a map:
- First, you show them **pictures** of olive groves (satellite images)
- Then, you **point at the olives** and say "this is olive, this is vineyard, this is wheat" (labels)
- The child **learns patterns** (the AI training)
- Now they can look at **new pictures** and find olives on their own (inference)

That's exactly what we're doing, but with a neural network instead of a child.

---

## The 7 Steps — What Each One Does and WHY

### Step 1: Crop Code Mapping
**What:** We read the Spain EuroCrop CSV files that contain crop codes (numbers like `3303050000` = olive).

**Why:** The government database uses hundreds of numeric codes. We simplify them into 7 categories:
```
Code 3303050000 → "olive_plantations" → Class 1 (OLIVE)
Code 3303040000 → "grape_vine"        → Class 2 (VINEYARD)  
Code 3201010000 → "common_wheat"      → Class 4 (ARABLE)
```

**Why 7 classes and not just "olive vs not olive"?**
Because the model learns BETTER when it knows what things are NOT olive. If it knows "this is a vineyard" and "this is wheat", it can focus on the real differences. Vineyards look very similar to olives in summer — both are rows of plants. But in winter, olives stay green and vineyards go bare. The model needs to learn this.

---

### Step 2: Match Seasonal Tiles
**What:** We find the 16 GeoTIFF files you downloaded from Google Earth Engine and group them.

**Why:** GEE split each season into 4 spatial tiles (the province is too large for one file). We match them:
```
Tile 1 (northwest corner):  spring.tif + summer.tif + fall.tif + winter.tif
Tile 2 (northeast corner):  spring.tif + summer.tif + fall.tif + winter.tif
Tile 3 (southwest corner):  spring.tif + summer.tif + fall.tif + winter.tif
Tile 4 (southeast corner):  spring.tif + summer.tif + fall.tif + winter.tif
```

---

### Step 3: Load Shapefile (EuroCrop Labels)
**What:** We read the Jaén province shapefile — a map of every agricultural parcel with its crop type.

**Why:** This is our **ground truth**. The Spanish government (SIGPAC) records what every farmer grows on every parcel. This is legally declared data — it's the most accurate crop map possible. Each polygon = one farm parcel, each parcel has a crop code.

```
Parcel #12345 → shape: [polygon coordinates] → crop: "olive_plantations"
Parcel #12346 → shape: [polygon coordinates] → crop: "common_wheat"
```

---

### Step 4: Rasterize + Cut Patches (THIS IS WHERE WE ARE NOW)
**What:** Two things happen here:

#### 4a. Rasterize Labels
We convert the vector polygons (shapes) into a pixel grid that matches the satellite image:

```
BEFORE (vector):              AFTER (raster):
  ┌──────┐                    0 0 0 0 0 0 0
  │olive │                    0 1 1 1 1 0 0    ← 1 = olive
  │      │                    0 1 1 1 1 0 0
  └──────┘                    0 0 0 0 0 0 0
     ┌───┐                    0 0 0 0 2 2 0    ← 2 = vineyard
     │vine│                   0 0 0 0 2 2 0
     └───┘                    0 0 0 0 0 0 0
```

**Why?** Neural networks work with grids of pixels, not vector shapes. We need pixel-level labels.

#### 4b. Cut Into 256×256 Patches
The full image is ~9000×9000 pixels (way too big for a GPU). We cut it into small 256×256 tiles:

```
Full image (9000×9000):          Patches (256×256 each):
┌─────────────────────┐          ┌────┐ ┌────┐ ┌────┐
│                     │          │ p1 │ │ p2 │ │ p3 │
│                     │    →     └────┘ └────┘ └────┘
│                     │          ┌────┐ ┌────┐ ┌────┐
│                     │          │ p4 │ │ p5 │ │ p6 │
│                     │          └────┘ └────┘ └────┘
└─────────────────────┘          ... thousands of patches
```

**Why?** 
- GPUs have limited memory (your GPU has ~8-16 GB)
- The full stacked image is 56 bands × 9000 × 9000 = **17 GB** — won't fit!
- Small patches let us batch 16 at a time and train efficiently
- We also FILTER out boring patches (all background, all empty)

**Why is this step slow?**
- We're reading 42 GB of satellite imagery from disk
- We're reading a 400 MB shapefile with 780,000 farm parcels
- We're reprojecting parcels from one coordinate system to another
- We're rasterizing polygons into pixels
- We're saving thousands of .npy files to disk

---

### Step 5: Compute Normalization Statistics
**What:** We calculate the mean and standard deviation of each of the 56 bands across all training patches.

**Why:** Neural networks learn better when input values are centered around 0. Band values come in different ranges:
```
BEFORE normalization:             AFTER normalization:
  B2 (Blue):    0.05 - 0.40       B2:  -1.5 to +1.5
  B8 (NIR):     0.15 - 0.60       B8:  -1.5 to +1.5
  NDVI:        -0.10 - 0.80       NDVI: -1.5 to +1.5
```

This makes training faster and more stable.

---

### Step 6: Build and Load the Model
**What:** We create a U-Net neural network with an EfficientNet-B4 encoder.

**Why U-Net?** It's the gold standard for **semantic segmentation** (classifying every pixel). It has:
- An **encoder** (EfficientNet-B4): looks at the image and extracts features (edges, textures, shapes)
- A **decoder**: takes those features and produces a classification for every pixel
- **Skip connections**: preserve fine details (important for parcel boundaries!)

```
Input: 56-band satellite patch (256×256)
  ↓
[Encoder] → extracts: "I see green rows", "dense canopy", "evergreen pattern"
  ↓
[Decoder] → outputs: "pixel (120,84) is OLIVE, pixel (120,85) is OLIVE, pixel (200,100) is VINEYARD"
  ↓
Output: 7-channel prediction map (256×256) — one probability per class per pixel
```

**Why EfficientNet-B4?** It was pre-trained on millions of photos (ImageNet). Even though satellite images are different from photos, the low-level features (edges, textures, patterns) transfer well. This is called **transfer learning** — we get a head start instead of learning from scratch.

---

### Step 7: Training
**What:** We show the model thousands of patches, it makes predictions, we tell it how wrong it was, and it adjusts.

**Why this specific training setup?**

**Loss function (DiceFocalLoss):** Measures how wrong the model is.
- **Dice Loss**: "How much do your predicted olive areas overlap with the real olive areas?" Good for imbalanced data.
- **Focal Loss**: "Pay extra attention to the hard examples!" (like olive vs vineyard boundaries)

**Class weights [0.1, 3.0, 1.5, ...]:** Olive is weighted 3× higher because:
- Background covers 70% of pixels — without weights, the model would just predict "everything is background" and get 70% accuracy
- We want it to focus on getting olives RIGHT

**Balanced sampler:** Patches containing olives are shown 5× more often.

**One training epoch:**
```
For each batch of 16 patches:
  1. Feed 16 images through the model → get 16 prediction maps
  2. Compare predictions to true labels → calculate loss (how wrong)
  3. Backpropagate → calculate gradients (which weights caused the error)
  4. Update weights → nudge the model to be less wrong next time
```

---

## Why 4 Seasons? (The Key Insight)

This is the MOST IMPORTANT design decision. Here's why:

**In summer, olive and vineyard look almost identical:**
```
Summer satellite image:
  Olive grove:  🟩🟩🟩🟩  (green rows, NDVI = 0.45)
  Vineyard:     🟩🟩🟩🟩  (green rows, NDVI = 0.42)
  → IMPOSSIBLE to tell apart!
```

**In winter, they look completely different:**
```
Winter satellite image:
  Olive grove:  🟩🟩🟩🟩  (STILL green! NDVI = 0.40) ← EVERGREEN!
  Vineyard:     🟫🟫🟫🟫  (bare sticks, NDVI = 0.10) ← DECIDUOUS!
  → EASY to tell apart!
```

By stacking 4 seasons, the model sees the **temporal pattern**:
```
56-band input = [spring_14bands | summer_14bands | fall_14bands | winter_14bands]

Olive:    [green | green | green | green]  → "always green = olive!"
Vineyard: [bare  | green | green | bare ]  → "seasonal = vineyard!"
Wheat:    [green | dead  | bare  | bare ]  → "spring only = wheat!"
```

---

## What Happens After Training?

Once the model is trained, you can point it at ANY Sentinel-2 image (anywhere in the world) and it will color-code every pixel:

```
Input: New satellite image of unknown area
  ↓
[Trained Model]
  ↓
Output: Classification map
  ■ Green  = Olive grove
  ■ Purple = Vineyard
  ■ Orange = Other orchard
  ■ Yellow = Arable crop
  ■ Red    = Building
  ■ Gray   = Road
  ■ Black  = Background
```

You can open this output in QGIS, Google Earth, or any GIS software as a georeferenced map.

---

## Current Status

Right now we're on **Step 4** — the script is:
1. ✅ Reading your 16 GeoTIFF files (42 GB total)
2. ✅ Reading the Jaén shapefile (780,000 farm parcels)
3. 🔄 Cutting satellite images into 256×256 training patches
4. ⏳ Then it will train for 2 test epochs (debug mode)

After this debug run confirms everything works, we'll do a **full training run** (50-100 epochs) which will take a few hours but produce a real, usable model.
