# Pixel-Perfect Aligner (AI Fix) for GIMP 3

A GIMP 3 plugin that converts messy, AI-generated or upscaled sprites into clean, grid-aligned pixel art by **re-sampling** the selected area into a target pixel grid (e.g. 64×64), with optional **noise cleanup**, **palette reduction**, **binary alpha**, **outline generation**, and **alpha-bleed fix** to prevent dark halos in-game.

Designed for fast iteration on UI icons, spritesheets, and small game assets where “almost pixel art” needs to become **actually pixel-perfect**.

---

## Features

### Core resampling (pixel alignment)
- Converts a selected region into an exact **Grid Width × Grid Height** output.
- Multiple sampling methods (e.g. Most-used/clustered, neighbor-based, averaging) to decide the final color per output pixel.
- Fine alignment controls (**Offset** + **Viewport Pan**) to “snap” details onto the grid.
- Optional **Auto Align** to search for an offset that yields the cleanest alignment.

### AI artifact cleanup (pre-processing)
- Optional **Pre denoise** step to reduce typical AI speckle/noise **before** sampling:
  - `Trimmed` mode: removes outliers while preserving edges (recommended).
  - `Median` mode: stronger cleanup, can flatten detail.

### Transparency and halo prevention
- **Alpha cutoff** to remove fringe pixels below a threshold.
- **Binary alpha** option to ensure the final sprite has **no semi-transparency**:
  - Everything becomes either fully opaque (255) or fully transparent (0).
- **Fix transparent RGB (alpha bleed)** with **Bleed radius** to fill RGB values under transparent pixels using nearby opaque colors:
  - Prevents “dark halos” when sprites are filtered in engines or UI scaling.

### Palette reduction and clustering
- **Max colors** limits the number of unique colors in the output.
- **Palette mode** dropdown:
  - `Top-N (most frequent)` (fast)
  - `K-Means clusters (perceptual)` (best for “car with blue window + black tires + green body” style palettes)
  - `Median cut (fallback)` (present depending on build)
- **Preserve rare** slider to keep small-but-important color groups from being lost.
- Optional **Perceptual mapping (Oklab)** for more visually correct palette matching.

### Outline generation
- Optional **Outline silhouette** pass:
  - Outline width
  - Outline alpha
  - Outline color

### Output options
- Output as a new layer placed at the selection position.
- Fit output to selection size (nearest neighbor).
- Replace pixels in active layer (bounds-based, destructive).
- Scale output layer (nearest neighbor).
- Output to **new image** (pixel size), creating a clean 1:1 pixel-art image at the target grid size.

### Presets
- Save and load named presets.
- Selecting a preset can apply it immediately (depending on build).
- Presets store all relevant settings (live preview is intentionally not stored).

### GIMP “Repeat Last” support
- Works with **Filters → Repeat** by storing and re-applying the last used settings.

---

## Installation

### Requirements
- **GIMP 3.x** with Python plugin support enabled.

### Install steps
1. Close GIMP.
2. Copy the plugin folder into your GIMP 3 plug-ins directory so the final path looks like:
   - `.../plug-ins/pixel_perfect_aligner/pixel_perfect_aligner.py`

**Default plug-ins folder locations**
- **Windows:** `%APPDATA%\GIMP\3.0\plug-ins\`
- **Linux:** `~/.config/GIMP/3.0/plug-ins/`
- **macOS:** `~/Library/Application Support/GIMP/3.0/plug-ins/`

3. Start GIMP.

### Updating
When updating, **delete the old plugin folder first** to avoid stale files:
- Remove `pixel_perfect_aligner/`
- Copy the new version
- Restart GIMP

---

## Usage

### Quick start workflow
1. Open your image in GIMP.
2. Make a **selection** around the sprite/icon you want to fix.
3. Run the plugin:
   - `Filters → Pixel-Perfect Aligner (AI Fix)`
4. Set:
   - **Grid width / height** (e.g. 64 × 64)
   - Sampling **Method**
   - Optional: enable **Pre denoise** and set Strength to ~20–35
   - Optional: set **Alpha cutoff** (e.g. 1–32)
   - Optional: enable **Fix transparent RGB (alpha bleed)** with radius 3–5
5. Click **Render preview** (or enable Live preview if you want constant updates).
6. Click **Apply**.

### Understanding Offset vs Viewport Pan
These two exist because “alignment” has two different kinds of shifts:

#### Offset X/Y (fine alignment)
Shifts the sampling grid *within* the selected region.  
Use this to “snap” details that are just slightly off-grid.

Example:
- A wheel rim sits between two pixels.
- Set **Offset X = +0.25** to shift the sampling phase so it locks onto the grid.

#### Viewport Pan X/Y (coarse shift)
Shifts the sampling window relative to the selection bounds.  
Use this when the content in the selection is generally placed wrong.

Example:
- The sprite is 2 px too far left inside your selection.
- Set **Viewport pan X = +2**.

---

## Recommended Settings (Practical Recipes)

### A) “AI sprite cleanup” (reduce speckle, keep edges)
- Pre denoise: **ON**
- Denoise mode: `Trimmed (recommended)`
- Strength: **20–35**
- Method: `Most Used (clustered)` or neighbor-based
- Alpha cutoff: **1–16**
- Fix transparent RGB: **ON**
- Bleed radius: **4**

### B) “No semi-transparent pixels” (engine-friendly)
- Alpha cutoff: **1–32** (depending on your asset)
- Binary alpha: **ON**
- Fix transparent RGB: **ON**
- Bleed radius: **4**

### C) “Smart palette” for multi-material sprites (car example)
- Max colors: **24–48**
- Palette mode: **K-Means clusters (perceptual)**
- Preserve rare: **50–70**
- Perceptual mapping (Oklab): **ON**

### D) “Hard pixel-art output as new 64×64 image”
- Output to new image (pixel size): **ON**
- Fit output to selection size: **OFF**
- Scale output layer: **OFF**

---

## Presets

### Save a preset
1. Adjust settings.
2. Click **Save**.
3. Enter a name.

### Load/apply a preset
- Pick it from the preset dropdown (auto-load on selection depending on version)  
  or click **Load** if the build still keeps it as a manual action.

### Notes
- Live preview state is intentionally not stored in presets.
- Presets are stored in your user config.

---

## Repeat Last (GIMP)

GIMP’s menu entry:
- `Filters → Repeat “Pixel-Perfect Aligner (AI Fix)”`

This triggers the plugin in a “use last values” run-mode. The plugin supports this by saving the last applied settings and re-running with them.

---

## Limitations / Known Behavior

- **Replace pixels in active layer** works on the **selection bounds** rectangle (destructive) rather than complex selection masks.
- Very large selections may be slow depending on method and preview settings.
- Palette clustering is best-effort; extremely noisy inputs may benefit from Pre denoise first.
- If your output looks different than preview, ensure the same options are enabled and that you are not using different output modes (fit/scale/new image).

---

## Troubleshooting

### Plugin doesn’t show up in the menu
Common causes:
- Wrong folder nesting (double folder).
  - Correct: `plug-ins/pixel_perfect_aligner/pixel_perfect_aligner.py`
  - Wrong: `plug-ins/pixel_perfect_aligner/pixel_perfect_aligner/pixel_perfect_aligner.py`
- Old plugin files still exist after update. Delete the folder and re-copy.
- Python plugin support missing in your GIMP build.

### “Repeat” does nothing
This usually means the plugin doesn’t handle the last-values run mode.  
Use a version that explicitly supports “Repeat Last” (WITH_LAST_VALS).

### Output is too big / scaled
Check:
- `Fit output to selection size`
- `Scale output layer`
- Or use `Output to new image (pixel size)` for a true 1:1 grid-sized output.

### Dark halos or fringe artifacts in-game
Enable:
- `Fix transparent RGB (alpha bleed)` and set `Bleed radius` to 3–5  
Also consider:
- `Binary alpha` and a small `Alpha cutoff` to remove fringes entirely.

---

## Development

### Repo layout (recommended)
- `pixel_perfect_aligner.py` – GIMP entry point, UI, pipeline glue
- `ppa_palette.py` – palette building + mapping
- `ppa_oklab.py` – perceptual color conversion helpers
- (optional) `ppa_denoise.py`, `ppa_outline.py`, `ppa_sampling.py` for clarity

### Contributing
PRs welcome. Keep changes:
- deterministic (same settings → same output)
- GIMP 3.x compatible
- UI consistent (tooltips on labels and controls, grouped sections)

---

## License
MIT License

Copyright (c) 2026 <YOUR NAME>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---

## Credits
Created for practical pixel-art production workflows: aligning sprites, removing AI artifacts, and producing engine-friendly outputs.
