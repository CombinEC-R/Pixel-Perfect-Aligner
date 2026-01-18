#!/usr/bin/env python3
# Pixel-Perfect Aligner (AI Fix) for GIMP 3.0.4
# v17:
# - Fix: GIMP "Repeat" now works (stores last settings; supports WITH_LAST_VALS)
# - UI: Preset auto-load on selection; Save defaults to current preset name; Live preview not stored in presets
# - UI: Clear section separators (thin lines + etched frames)
# v16:
# - Add: Binary alpha option (remove all semi-transparency; output alpha becomes 0/255)
# v15:
# - Add: palette mode dropdown (Top-N / K-Means clusters) + preserve rare colors + perceptual mapping
# - Refactor: palette + perceptual helpers moved into separate files (ppa_palette.py, ppa_oklab.py)
# v14:
# - Add: optional pre-denoise (AI speckle/noise reducer) before sampling
# - UI: denoise controls with full tooltips
# v13:
# - Output bleed after outline; framed UI + tooltips
# v11:
# - Fix: preview brightness mismatch (proper alpha compositing over checker)
# - Add: Fix transparent RGB (alpha-bleed) option + radius
# - Add: tooltips for all controls (description + example)
# - Add: optional output to new image (pixel size)
# v10:
# - Apply: place output at selection position (optional)
# - Apply: optional fit output to selection bounds (nearest)
# - Apply: optional destructive replace into active layer (bounds)
# - Fix: restore interpolation context after scaling layer
# v9:
# - Fix: grid overlay loop could freeze preview (indent bug)
# - Default live preview off; render on button press
# v8:
# - Preview grid aligns to pixels (preview uses integer cell scale that fits)
# - Optional palette limit (max colors) for output
# - Optional non-live preview (Render Preview button)
import sys
import os
import json
import math
import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
gi.require_version("Gtk", "3.0")

from gi.repository import Gimp, GimpUi, Gegl, Gtk, GLib

# Allow importing helper modules shipped alongside this plug-in
PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
if PLUGIN_DIR not in sys.path:
	sys.path.insert(0, PLUGIN_DIR)

import ppa_palette

PROC_NAME = "python-fu-pixel-perfect-aligner-ai-fix"
MENU_PATH = "<Image>/Filters/Enhance"

# Persist last-used settings so GIMP's built-in "Repeat" (WITH_LAST_VALS)
# can re-run the plug-in without showing the dialog.
LAST_SETTINGS_KEY = "pixel_perfect_aligner_ai_fix_last_settings"


def _clamp(v, lo, hi):
	if v < lo:
		return lo
	if v > hi:
		return hi
	return v


def _clamp_int(v, lo, hi):
	if v < lo:
		return lo
	if v > hi:
		return hi
	return v


def _get_selection_bounds(image):
	b = Gimp.Selection.bounds(image)
	non_empty = bool(getattr(b, "non_empty", False))
	if not non_empty:
		return (0, 0, image.get_width(), image.get_height())
	x1 = int(getattr(b, "x1", 0))
	y1 = int(getattr(b, "y1", 0))
	x2 = int(getattr(b, "x2", image.get_width()))
	y2 = int(getattr(b, "y2", image.get_height()))
	return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _brightness(r, g, b):
	return (0.299 * float(r)) + (0.587 * float(g)) + (0.114 * float(b))


def _color_dist_sq(r1, g1, b1, r2, g2, b2):
	dr = r1 - r2
	dg = g1 - g2
	db = b1 - b2
	return (dr * dr) + (dg * dg) + (db * db)


def _representative_most_used_clustered(pixels_rgba, ignore_transparent, similarity_threshold, weighted_mode):
	thr_sq = int(similarity_threshold) * int(similarity_threshold)
	clusters = []
	total = len(pixels_rgba) // 4
	idx = 0
	for _ in range(total):
		r = int(pixels_rgba[idx])
		g = int(pixels_rgba[idx + 1])
		b = int(pixels_rgba[idx + 2])
		a = int(pixels_rgba[idx + 3])
		idx += 4

		if ignore_transparent and a == 0:
			continue

		weight = 1.0
		if weighted_mode != "none":
			br = _brightness(r, g, b) / 255.0
			if weighted_mode == "light":
				weight = br
			elif weighted_mode == "dark":
				weight = 1.0 - br
			if weight < 0.001:
				weight = 0.001

		found = -1
		for i, c in enumerate(clusters):
			if _color_dist_sq(r, g, b, c["r"], c["g"], c["b"]) <= thr_sq:
				found = i
				break

		if found < 0:
			clusters.append({
				"r": r, "g": g, "b": b,
				"w": weight,
				"sum_r": float(r) * weight,
				"sum_g": float(g) * weight,
				"sum_b": float(b) * weight,
				"sum_a": float(a) * weight,
			})
		else:
			c = clusters[found]
			c["w"] += weight
			c["sum_r"] += float(r) * weight
			c["sum_g"] += float(g) * weight
			c["sum_b"] += float(b) * weight
			c["sum_a"] += float(a) * weight
			wgt = c["w"]
			c["r"] = int(round(c["sum_r"] / wgt))
			c["g"] = int(round(c["sum_g"] / wgt))
			c["b"] = int(round(c["sum_b"] / wgt))

	if len(clusters) == 0:
		return (0, 0, 0, 0)

	best = clusters[0]
	for c in clusters[1:]:
		if c["w"] > best["w"]:
			best = c

	wgt = best["w"]
	return (
		_clamp_int(int(round(best["sum_r"] / wgt)), 0, 255),
		_clamp_int(int(round(best["sum_g"] / wgt)), 0, 255),
		_clamp_int(int(round(best["sum_b"] / wgt)), 0, 255),
		_clamp_int(int(round(best["sum_a"] / wgt)), 0, 255),
	)


def _representative_average(pixels_rgba, ignore_transparent):
	sum_r = 0
	sum_g = 0
	sum_b = 0
	sum_a = 0
	count = 0
	total = len(pixels_rgba) // 4
	idx = 0
	for _ in range(total):
		r = int(pixels_rgba[idx])
		g = int(pixels_rgba[idx + 1])
		b = int(pixels_rgba[idx + 2])
		a = int(pixels_rgba[idx + 3])
		idx += 4
		if ignore_transparent and a == 0:
			continue
		sum_r += r
		sum_g += g
		sum_b += b
		sum_a += a
		count += 1
	if count <= 0:
		return (0, 0, 0, 0)
	return (
		int(round(sum_r / count)),
		int(round(sum_g / count)),
		int(round(sum_b / count)),
		int(round(sum_a / count)),
	)




def _denoise_cell_samples(cell_bytes, ignore_transparent, denoise_enabled, denoise_strength, denoise_mode):
	"""Lightweight noise reduction for typical AI speckle.
	Applied per output-cell BEFORE the representative color is chosen.
	
	Modes:
	- trimmed: remove outliers by luminance deviation, then keep a subset for the representative step
	- median: compute a median color (robust against speckles)
	
	Strength 0..100. At 0: no effect. At 100: aggressive (keeps ~25% closest samples).
	"""
	if not denoise_enabled:
		return cell_bytes
	strength = int(denoise_strength)
	if strength <= 0:
		return cell_bytes
	mode = str(denoise_mode)
	if mode == '' or mode == 'off':
		return cell_bytes

	mv = memoryview(cell_bytes)
	n = len(cell_bytes) // 4
	if n <= 3:
		return cell_bytes

	samples = []
	for p in range(n):
		i = p * 4
		r = int(mv[i])
		g = int(mv[i + 1])
		b = int(mv[i + 2])
		a = int(mv[i + 3])
		if ignore_transparent and a == 0:
			continue
		lum = _brightness(r, g, b)
		samples.append((lum, r, g, b, a))

	if len(samples) <= 3:
		return cell_bytes

	# Median luminance (robust center)
	lums = sorted([s[0] for s in samples])
	med_l = lums[len(lums) // 2]

	if mode == 'median':
		rs = sorted([s[1] for s in samples])
		gs = sorted([s[2] for s in samples])
		bs = sorted([s[3] for s in samples])
		as_ = sorted([s[4] for s in samples])
		m = len(rs) // 2
		return bytes([int(rs[m]), int(gs[m]), int(bs[m]), int(as_[m])])

	# trimmed (default): keep only samples closest to median luminance
	dev = [(abs(s[0] - med_l), s) for s in samples]
	dev.sort(key=lambda t: t[0])
	keep_ratio = 1.0 - (float(strength) / 100.0) * 0.75
	if keep_ratio < 0.25:
		keep_ratio = 0.25
	keep = max(3, int(round(len(dev) * keep_ratio)))
	kept = [t[1] for t in dev[:keep]]
	out = bytearray(len(kept) * 4)
	for i, s2 in enumerate(kept):
		o = i * 4
		out[o] = int(s2[1])
		out[o + 1] = int(s2[2])
		out[o + 2] = int(s2[3])
		out[o + 3] = int(s2[4])
	return bytes(out)
def _render_grid_from_source(drawable, src_x, src_y, src_w, src_h, grid_w, grid_h, offset_x, offset_y, scale, method, similarity_threshold, ignore_transparent, neighbor_margin, denoise_enabled, denoise_strength, denoise_mode):
	buf = drawable.get_buffer()

	pad = int(math.ceil(max(src_w, src_h) * 0.02)) + 4
	x0 = _clamp_int(src_x - pad, 0, drawable.get_width())
	y0 = _clamp_int(src_y - pad, 0, drawable.get_height())
	x1 = _clamp_int(src_x + src_w + pad, 0, drawable.get_width())
	y1 = _clamp_int(src_y + src_h + pad, 0, drawable.get_height())
	rw = max(1, x1 - x0)
	rh = max(1, y1 - y0)

	rect = Gegl.Rectangle.new(x0, y0, rw, rh)
	src_bytes = buf.get(rect, 1.0, "RGBA u8", Gegl.AbyssPolicy.CLAMP)
	mv = memoryview(src_bytes)
	rowstride = rw * 4

	out = bytearray(grid_w * grid_h * 4)

	samp_w = float(src_w) * float(scale)
	samp_h = float(src_h) * float(scale)
	cell_w = samp_w / float(grid_w)
	cell_h = samp_h / float(grid_h)

	def sample_cell_pixels(i, j):
		cx0 = float(i) * cell_w
		cy0 = float(j) * cell_h
		cx1 = float(i + 1) * cell_w
		cy1 = float(j + 1) * cell_h

		if method == "neighbor":
			mx = cell_w * neighbor_margin
			my = cell_h * neighbor_margin
			cx0 -= mx
			cy0 -= my
			cx1 += mx
			cy1 += my

		sx0_f = float(src_x) + offset_x + cx0
		sy0_f = float(src_y) + offset_y + cy0
		sx1_f = float(src_x) + offset_x + cx1
		sy1_f = float(src_y) + offset_y + cy1

		sx0 = int(math.floor(sx0_f))
		sy0 = int(math.floor(sy0_f))
		sx1 = int(math.ceil(sx1_f))
		sy1 = int(math.ceil(sy1_f))

		sx0 = _clamp_int(sx0, x0, x0 + rw)
		sy0 = _clamp_int(sy0, y0, y0 + rh)
		sx1 = _clamp_int(sx1, x0, x0 + rw)
		sy1 = _clamp_int(sy1, y0, y0 + rh)

		if sx1 <= sx0 or sy1 <= sy0:
			return bytes()

		tmp = bytearray()
		for sy in range(sy0, sy1):
			py = float(sy) + 0.5
			if py < sy0_f or py >= sy1_f:
				continue
			row_off = (sy - y0) * rowstride
			for sx in range(sx0, sx1):
				px = float(sx) + 0.5
				if px < sx0_f or px >= sx1_f:
					continue
				off = row_off + (sx - x0) * 4
				tmp.extend(mv[off:off + 4])
		return bytes(tmp)

	for j in range(grid_h):
		for i in range(grid_w):
			cell = sample_cell_pixels(i, j)
			cell = _denoise_cell_samples(cell, ignore_transparent, denoise_enabled, denoise_strength, denoise_mode)
			if len(cell) == 0:
				r, g, b, a = (0, 0, 0, 0)
			else:
				cmv = memoryview(cell)
				if method == "average" or method == "neighbor":
					r, g, b, a = _representative_average(cmv, ignore_transparent)
				elif method == "most_used":
					r, g, b, a = _representative_most_used_clustered(cmv, ignore_transparent, similarity_threshold, "none")
				elif method == "most_used_light":
					r, g, b, a = _representative_most_used_clustered(cmv, ignore_transparent, similarity_threshold, "light")
				elif method == "most_used_dark":
					r, g, b, a = _representative_most_used_clustered(cmv, ignore_transparent, similarity_threshold, "dark")
				else:
					r, g, b, a = _representative_average(cmv, ignore_transparent)
			oi = ((j * grid_w) + i) * 4
			out[oi] = r
			out[oi + 1] = g
			out[oi + 2] = b
			out[oi + 3] = a

	return bytes(out)


def _alpha_cut(rgba, cutoff):
	if cutoff <= 0:
		return rgba
	mv = memoryview(rgba)
	out = bytearray(len(rgba))
	for i in range(0, len(rgba), 4):
		out[i] = mv[i]
		out[i + 1] = mv[i + 1]
		out[i + 2] = mv[i + 2]
		a = int(mv[i + 3])
		out[i + 3] = 0 if a < cutoff else a
	return bytes(out)


def _binary_alpha(rgba, cutoff, enabled):
	"""Force alpha to be strictly binary (0 or 255).

	If enabled:
	- Pixels with alpha < threshold become fully transparent (alpha=0)
	- Pixels with alpha >= threshold become fully opaque (alpha=255)

	The threshold is derived from alpha cutoff:
	- threshold = max(1, cutoff)
	- If cutoff==0, this effectively treats alpha==0 as background and everything else as opaque.

	This is useful when you want a sprite with no semi-transparency at all.
	"""
	if not enabled:
		return rgba
	thr = int(cutoff)
	if thr <= 0:
		thr = 1
	mv = memoryview(rgba)
	out = bytearray(len(rgba))
	for i in range(0, len(rgba), 4):
		out[i] = mv[i]
		out[i + 1] = mv[i + 1]
		out[i + 2] = mv[i + 2]
		a = int(mv[i + 3])
		out[i + 3] = 255 if a >= thr else 0
	return bytes(out)


def _mask_from_alpha(rgba, w, h, cutoff):
	mv = memoryview(rgba)
	mask = bytearray(w * h)
	k = 0
	i = 0
	for _y in range(h):
		for _x in range(w):
			a = int(mv[i + 3])
			mask[k] = 1 if a >= cutoff else 0
			i += 4
			k += 1
	return mask


def _dilate(mask, w, h, radius):
	if radius <= 0:
		return mask
	out = bytearray(w * h)
	for y in range(h):
		for x in range(w):
			val = 0
			for oy in range(-radius, radius + 1):
				yy = y + oy
				if yy < 0 or yy >= h:
					continue
				row = yy * w
				for ox in range(-radius, radius + 1):
					xx = x + ox
					if xx < 0 or xx >= w:
						continue
					if mask[row + xx] != 0:
						val = 1
						break
				if val == 1:
					break
			out[y * w + x] = val
	return out


def _apply_outline(rgba, w, h, alpha_cutoff, outline_width, outline_rgb, outline_alpha):
	if outline_width <= 0 or outline_alpha <= 0:
		return rgba
	base = _mask_from_alpha(rgba, w, h, alpha_cutoff)
	dil = _dilate(base, w, h, outline_width)
	out = bytearray(rgba)
	r_o = int(outline_rgb[0])
	g_o = int(outline_rgb[1])
	b_o = int(outline_rgb[2])
	a_o = int(outline_alpha)
	for y in range(h):
		for x in range(w):
			k = y * w + x
			if dil[k] == 1 and base[k] == 0:
				i = k * 4
				out[i] = r_o
				out[i + 1] = g_o
				out[i + 2] = b_o
				out[i + 3] = a_o
	return bytes(out)


def _palette_limit(rgba, max_colors, ignore_transparent):
	if max_colors <= 0:
		return rgba
	max_colors = _clamp_int(int(max_colors), 2, 256)
	mv = memoryview(rgba)
	hist = {}
	for i in range(0, len(rgba), 4):
		a = int(mv[i + 3])
		if ignore_transparent and a == 0:
			continue
		r = int(mv[i]) >> 3
		g = int(mv[i + 1]) >> 3
		b = int(mv[i + 2]) >> 3
		key = (r << 10) | (g << 5) | b
		hist[key] = hist.get(key, 0) + 1
	if len(hist) <= max_colors:
		return rgba
	top = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[:max_colors]
	pal = []
	for key, _cnt in top:
		r5 = (key >> 10) & 31
		g5 = (key >> 5) & 31
		b5 = key & 31
		r8 = (r5 << 3) | (r5 >> 2)
		g8 = (g5 << 3) | (g5 >> 2)
		b8 = (b5 << 3) | (b5 >> 2)
		pal.append((r8, g8, b8))

	out = bytearray(rgba)
	for i in range(0, len(out), 4):
		a = int(out[i + 3])
		if ignore_transparent and a == 0:
			continue
		r = int(out[i])
		g = int(out[i + 1])
		b = int(out[i + 2])
		best = pal[0]
		best_d = _color_dist_sq(r, g, b, best[0], best[1], best[2])
		for p in pal[1:]:
			d = _color_dist_sq(r, g, b, p[0], p[1], p[2])
			if d < best_d:
				best_d = d
				best = p
		out[i] = best[0]
		out[i + 1] = best[1]
		out[i + 2] = best[2]
	return bytes(out)


def _nearest_scale_rgba(src_rgba, src_w, src_h, scale_factor):
	mv = memoryview(src_rgba)
	dst_w = max(1, int(src_w * scale_factor))
	dst_h = max(1, int(src_h * scale_factor))
	out = bytearray(dst_w * dst_h * 4)
	for y in range(dst_h):
		sy = int(float(y) / float(scale_factor))
		if sy >= src_h:
			sy = src_h - 1
		for x in range(dst_w):
			sx = int(float(x) / float(scale_factor))
			if sx >= src_w:
				sx = src_w - 1
			si = ((sy * src_w) + sx) * 4
			di = ((y * dst_w) + x) * 4
			out[di:di + 4] = mv[si:si + 4]
	return (bytes(out), dst_w, dst_h)


def _scale_rgba_to_size(src_rgba, src_w, src_h, dst_w, dst_h):
	mv = memoryview(src_rgba)
	dst_w = max(1, int(dst_w))
	dst_h = max(1, int(dst_h))
	out = bytearray(dst_w * dst_h * 4)
	for y in range(dst_h):
		sy = int((float(y) * float(src_h)) / float(dst_h))
		if sy >= src_h:
			sy = src_h - 1
		for x in range(dst_w):
			sx = int((float(x) * float(src_w)) / float(dst_w))
			if sx >= src_w:
				sx = src_w - 1
			si = ((sy * src_w) + sx) * 4
			di = ((y * dst_w) + x) * 4
			out[di:di + 4] = mv[si:si + 4]
	return bytes(out)



def _compose_checker(pw, ph):
	canvas = bytearray(pw * ph * 4)
	for y in range(ph):
		for x in range(pw):
			ck = 205 if (((x // 8) + (y // 8)) % 2 == 0) else 170
			di = ((y * pw) + x) * 4
			canvas[di] = ck
			canvas[di + 1] = ck
			canvas[di + 2] = ck
			canvas[di + 3] = 255
	return canvas


def _blit_center(canvas, pw, ph, img_rgba, iw, ih):
	ox = (pw - iw) // 2
	oy = (ph - ih) // 2
	if iw <= 0 or ih <= 0:
		return (ox, oy)
	src_x0 = 0
	src_y0 = 0
	dst_x0 = ox
	dst_y0 = oy
	if dst_x0 < 0:
		src_x0 = -dst_x0
		dst_x0 = 0
	if dst_y0 < 0:
		src_y0 = -dst_y0
		dst_y0 = 0
	copy_w = min(iw - src_x0, pw - dst_x0)
	copy_h = min(ih - src_y0, ph - dst_y0)
	if copy_w <= 0 or copy_h <= 0:
		return (ox, oy)
	# IMPORTANT: alpha composite over the checker, then force alpha=255.
	# This avoids preview looking darker than the real layer display.
	umv = memoryview(img_rgba)
	for y in range(copy_h):
		sr = ((y + src_y0) * iw + src_x0) * 4
		dr = ((y + dst_y0) * pw + dst_x0) * 4
		for x in range(copy_w):
			si = sr + (x * 4)
			di = dr + (x * 4)
			a = int(umv[si + 3])
			if a <= 0:
				continue
			if a >= 255:
				canvas[di] = int(umv[si])
				canvas[di + 1] = int(umv[si + 1])
				canvas[di + 2] = int(umv[si + 2])
				canvas[di + 3] = 255
				continue
			ia = 255 - a
			canvas[di] = (int(umv[si]) * a + int(canvas[di]) * ia) // 255
			canvas[di + 1] = (int(umv[si + 1]) * a + int(canvas[di + 1]) * ia) // 255
			canvas[di + 2] = (int(umv[si + 2]) * a + int(canvas[di + 2]) * ia) // 255
			canvas[di + 3] = 255
	return (ox, oy)


def _make_bleed_offsets(radius):
	# Sorted offsets by distance, excluding (0,0).
	offs = []
	r = int(max(1, radius))
	for oy in range(-r, r + 1):
		for ox in range(-r, r + 1):
			if ox == 0 and oy == 0:
				continue
			d2 = (ox * ox) + (oy * oy)
			offs.append((d2, ox, oy))
	offs.sort(key=lambda t: t[0])
	return [(ox, oy) for _d2, ox, oy in offs]


def _fix_transparent_rgb_inplace(rgba_u8, w, h, radius, alpha_threshold):
	# Fixes "alpha bleed" by copying RGB from nearby opaque pixels into fully
	# transparent pixels. Alpha stays unchanged.
	if radius <= 0:
		return
	w = int(w)
	h = int(h)
	if w <= 0 or h <= 0:
		return
	thr = int(alpha_threshold)
	offs = _make_bleed_offsets(radius)
	for y in range(h):
		row = y * w
		for x in range(w):
			i = (row + x) * 4
			a = int(rgba_u8[i + 3])
			if a > thr:
				continue
			best = None
			for ox, oy in offs:
				xx = x + ox
				yy = y + oy
				if xx < 0 or xx >= w or yy < 0 or yy >= h:
					continue
				j = ((yy * w) + xx) * 4
				aj = int(rgba_u8[j + 3])
				if aj > thr:
					best = j
					break
			if best is not None:
				rgba_u8[i] = rgba_u8[best]
				rgba_u8[i + 1] = rgba_u8[best + 1]
				rgba_u8[i + 2] = rgba_u8[best + 2]


def _apply_output_bleed(rgba, w, h, enabled, radius, alpha_cutoff):
	"""Copy RGB from nearby non-transparent pixels into pixels that are treated as transparent.

	This prevents dark/bright halos when the output is later filtered (bilinear, mipmaps, UI scaling).
	The threshold is derived from alpha_cutoff: pixels with alpha < alpha_cutoff are treated as transparent.

	If alpha_cutoff is 0, only fully transparent pixels (alpha==0) are affected.
	"""
	if not bool(enabled):
		return rgba
	r = int(radius)
	if r <= 0:
		return rgba
	thr = 0
	try:
		c = int(alpha_cutoff)
		thr = 0 if c <= 0 else max(0, c - 1)
	except Exception:
		thr = 0
	tmp = bytearray(rgba)
	_fix_transparent_rgb_inplace(tmp, int(w), int(h), r, thr)
	return bytes(tmp)


def _overlay_grid(canvas, pw, ph, origin_x, origin_y, step, alpha):
	if step < 2:
		return
	alpha = _clamp_int(int(alpha), 0, 255)

	def blend_pixel(px_i):
		bg_r = int(canvas[px_i])
		bg_g = int(canvas[px_i + 1])
		bg_b = int(canvas[px_i + 2])
		a = alpha
		ia = 255 - a
		canvas[px_i] = (0 * a + bg_r * ia) // 255
		canvas[px_i + 1] = (0 * a + bg_g * ia) // 255
		canvas[px_i + 2] = (0 * a + bg_b * ia) // 255
		canvas[px_i + 3] = 255

	x = origin_x
	while x < pw:
		if x >= 0:
			for y in range(0, ph):
				blend_pixel(((y * pw) + x) * 4)
		x += step

	y = origin_y
	while y < ph:
		if y >= 0:
			row = y * pw
			for x2 in range(0, pw):
				blend_pixel((row + x2) * 4)
		y += step


def _ensure_dir(path):
	if not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)


def _presets_path():
	cfg = GLib.get_user_config_dir()
	dirp = os.path.join(cfg, "pixel_perfect_aligner")
	_ensure_dir(dirp)
	return os.path.join(dirp, "presets.json")


def _last_settings_path():
	# Store last-used settings next to presets so "Repeat" can work even if
	# Gimp.get_data/set_data is unavailable in a given build.
	cfg = GLib.get_user_config_dir()
	dirp = os.path.join(cfg, "pixel_perfect_aligner")
	_ensure_dir(dirp)
	return os.path.join(dirp, "last_settings.json")


def _save_last_settings_dict(d):
	# Best-effort: try GIMP's internal data store first, then fallback to file.
	try:
		payload = json.dumps(d).encode("utf-8")
	except Exception:
		return
	try:
		if hasattr(Gimp, "set_data"):
			try:
				Gimp.set_data(LAST_SETTINGS_KEY, payload)
			except Exception:
				pass
	except Exception:
		pass
	try:
		path = _last_settings_path()
		with open(path, "w", encoding="utf-8") as f:
			json.dump(d, f, indent=2)
	except Exception:
		pass


def _load_last_settings_dict():
	# Try GIMP data store, fallback to file.
	try:
		if hasattr(Gimp, "get_data"):
			b = Gimp.get_data(LAST_SETTINGS_KEY)
			if b is not None:
				try:
					# Some GI builds return GLib.Bytes.
					if hasattr(b, "get_data"):
						b = b.get_data()
				except Exception:
					pass
				try:
					if isinstance(b, (bytes, bytearray)):
						obj = json.loads(bytes(b).decode("utf-8"))
						if isinstance(obj, dict):
							return obj
				except Exception:
					pass
	except Exception:
		pass
	path = _last_settings_path()
	if not os.path.isfile(path):
		return None
	try:
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
		if isinstance(obj, dict):
			return obj
	except Exception:
		return None
	return None


def _load_presets():
	path = _presets_path()
	if not os.path.isfile(path):
		return {}
	try:
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
		if isinstance(obj, dict):
			return obj
	except Exception:
		return {}
	return {}


def _save_presets(presets):
	path = _presets_path()
	try:
		with open(path, "w", encoding="utf-8") as f:
			json.dump(presets, f, indent=2)
	except Exception:
		pass


def _read_drawable_region_rgba(drawable, x, y, w, h):
	w = max(1, int(w))
	h = max(1, int(h))
	x = _clamp_int(int(x), 0, max(0, drawable.get_width() - 1))
	y = _clamp_int(int(y), 0, max(0, drawable.get_height() - 1))
	w = min(w, drawable.get_width() - x)
	h = min(h, drawable.get_height() - y)
	rect = Gegl.Rectangle.new(x, y, w, h)
	buf = drawable.get_buffer()
	data = buf.get(rect, 1.0, "RGBA u8", Gegl.AbyssPolicy.CLAMP)
	return (bytes(data), w, h)


class PixelPerfectAlignerDialog(Gtk.Dialog):
	def __init__(self, image, drawable, sel_bounds):
		super().__init__(title="Pixel-Perfect Aligner", transient_for=None, flags=0)
		self.set_default_size(1140, 660)
		self._image = image
		self._drawable = drawable
		self._sel_x, self._sel_y, self._sel_w, self._sel_h = sel_bounds

		self.grid_w = 64
		self.grid_h = 64
		self.method = "most_used"
		self.similarity_threshold = 16
		self.ignore_transparent = True
		# Pre-denoise (AI speckle/noise reducer) applied per cell before sampling.
		self.denoise_enabled = False
		self.denoise_strength = 35
		self.denoise_mode = "trimmed"
		# Fix transparent RGB (alpha-bleed) to avoid dark fringes after filtering.
		self.fix_transparent_rgb = True
		self.bleed_radius = 4
		self.neighbor_margin = 0.40
		self.offset_x = 0.0
		self.offset_y = 0.0
		self.scale = 1.0
		self.pan_x = 0.0
		self.pan_y = 0.0

		self.preview_cell = 8
		self.fit_preview = True
		self.show_grid = True
		self.grid_alpha = 90
		# Default to manual preview to avoid expensive recompute loops while tweaking values.
		self.live_preview = False

		self.alpha_cutoff = 1
		# Optional: remove all semi-transparency at the end (only 0/255 alpha).
		self.binary_alpha = False
		self.outline_enabled = False
		self.outline_width = 1
		self.outline_alpha = 255
		self.outline_color = (0, 0, 0)

		self.palette_max = 0

		# Palette reduction
		self.palette_mode = 'topn'
		self.palette_preserve_rare = 40
		self.palette_use_oklab = False

		self.scale_output_enabled = False
		self.output_scale = 1

		self.place_output_on_selection = True
		self.fit_output_to_selection = False
		self.replace_in_active_layer = False
		self.output_to_new_image = False

		self.presets = _load_presets()

		self.preview_needs_update = True
		self._preview_update_source_id = 0

		self._build_ui()
		self._refresh_preset_combo()
		if self.live_preview:
			self._queue_preview_update()

	def _tip(self, widget, text):
		try:
			widget.set_tooltip_text(text)
		except Exception:
			pass

	def _build_ui(self):
		box = self.get_content_area()
		root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
		root.set_border_width(10)
		box.add(root)

		left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
		root.pack_start(left, True, True, 0)
		lbl_l = Gtk.Label(label="Source (selection used for sampling)")
		lbl_l.set_xalign(0.0)
		left.pack_start(lbl_l, False, False, 0)
		self.src_area = GimpUi.PreviewArea.new()
		self.src_area.set_size_request(500, 500)
		left.pack_start(self.src_area, True, True, 0)

		right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
		root.pack_start(right, True, True, 0)
		lbl_r = Gtk.Label(label="Output Preview")
		lbl_r.set_xalign(0.0)
		right.pack_start(lbl_r, False, False, 0)
		self.out_area = GimpUi.PreviewArea.new()
		self.out_area.set_size_request(500, 500)
		right.pack_start(self.out_area, True, True, 0)

		self.src_area.connect("size-allocate", self._on_preview_resized)
		self.out_area.connect("size-allocate", self._on_preview_resized)

		ctrl = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
		root.pack_start(ctrl, False, False, 0)

		pbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
		ctrl.pack_start(pbox, False, False, 0)
		self.combo_preset = Gtk.ComboBoxText()
		self.combo_preset.set_hexpand(True)
		# Auto-load presets on selection (no extra "Load" click required).
		self._preset_ui_lock = False
		self.combo_preset.connect("changed", self._on_preset_selected)
		lbl_preset = Gtk.Label(label="Preset")
		lbl_preset.set_xalign(0.0)
		pbox.pack_start(lbl_preset, False, False, 0)
		pbox.pack_start(self.combo_preset, True, True, 0)
		btn_save = Gtk.Button(label="Save")
		btn_load = Gtk.Button(label="Load")
		btn_del = Gtk.Button(label="Delete")
		btn_save.connect("clicked", self._on_preset_save)
		btn_load.connect("clicked", self._on_preset_load)
		btn_del.connect("clicked", self._on_preset_delete)
		pbox.pack_start(btn_save, False, False, 0)
		pbox.pack_start(btn_load, False, False, 0)
		pbox.pack_start(btn_del, False, False, 0)

		# ---------- Helpers for framed sections + tooltips on BOTH labels and controls ----------
		def mk_frame(title_text):
			# Visually separate sections with a thin line, because on dark themes
			# Frame borders can be hard to see.
			if not hasattr(self, "_frame_count"):
				self._frame_count = 0
			if int(getattr(self, "_frame_count", 0)) > 0:
				sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
				try:
					sep.set_margin_top(4)
					sep.set_margin_bottom(4)
				except Exception:
					pass
				ctrl.pack_start(sep, False, False, 0)
			self._frame_count = int(getattr(self, "_frame_count", 0)) + 1
			fr = Gtk.Frame(label=title_text)
			try:
				fr.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
			except Exception:
				pass
			try:
				fr.set_margin_top(2)
				fr.set_margin_bottom(2)
			except Exception:
				pass
			inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
			inner.set_border_width(8)
			fr.add(inner)
			ctrl.pack_start(fr, False, False, 0)
			return inner

		def mk_grid(parent_box):
			g = Gtk.Grid(column_spacing=10, row_spacing=6)
			parent_box.pack_start(g, False, False, 0)
			return g

		def add_label(g, text, col, row_i, tip):
			lbl = Gtk.Label(label=text)
			lbl.set_xalign(0.0)
			g.attach(lbl, col, row_i, 1, 1)
			self._tip(lbl, tip)
			return lbl

		def tip2(widget, tip):
			self._tip(widget, tip)
			return widget

		# ---------- Tooltips (always include direction + example) ----------
		tip_preset = "Preset: save/load a full settings bundle.\nExample: store your 64x64 + 32 colors + cutoff setup as 'Icons64'."
		tip_save = "Save preset: stores the current settings under a name."
		tip_load = "Load preset: applies the selected preset to all controls."
		tip_del = "Delete preset: removes the selected preset from disk."

		tip_grid_w = "Grid width (output pixels). One grid cell becomes one output pixel.\nExample: 64 produces a 64px wide sprite."
		tip_grid_h = "Grid height (output pixels). One grid cell becomes one output pixel.\nExample: 64 produces a 64px tall sprite."
		tip_method = "Sampling method per output cell.\nMost Used: crisp pixel art. Average: smoother. Neighbor: stabilizes noisy edges.\nExample: start with Most Used, threshold 12-20."
		tip_sim = "Similarity threshold used by Most Used clustering.\nLower = stricter, higher = merges close shades.\nExample: 16 reduces AI speckle without flattening everything."
		tip_margin = "Neighbor margin (Neighbor method only).\nHigher samples a larger area around each cell to stabilize color.\nExample: 0.40 is a good default."
		tip_denoise = "Pre denoise (AI noise reducer): removes tiny speckles / pixel trash before the main sampling.\nThis happens PER OUTPUT CELL, so it keeps edges but ignores outliers.\nExample: enable, Strength 35, Trimmed. If details disappear, lower to 15."
		tip_denoise_strength = "Denoise strength 0..100.\nHigher = stronger outlier removal (more smoothing), lower = keeps more detail.\nExample: 25 for mild cleanup, 50 for aggressive AI artifacts."
		tip_denoise_mode = "Denoise mode.\nTrimmed: keeps a subset of samples near the median brightness (recommended).\nMedian: very strong, can wipe fine texture.\nExample: use Median only when you have heavy speckle noise."
		tip_ignore = "Ignore fully transparent source pixels while sampling (alpha==0).\nExample: prevents background bleed into your sprite."
		tip_bleed = "Fix transparent RGB (alpha bleed).\nAfter alpha cutoff/palette/outline, RGB is copied into pixels treated as transparent (alpha < cutoff).\nPrevents halos when later filtered (bilinear, mipmaps).\nExample: enable + radius 4. Set cutoff=0 to only affect alpha==0."
		tip_bleed_r = "Bleed radius in output pixels.\nRecommended: 3 to 5 (default 4).\nExample: 4 removes dark fringes around silhouettes."

		tip_offx = "Offset X (fine): shifts the sampling grid inside the selection.\nDirection: +X samples more to the right (content appears to move left).\nExample: try +0.25 or -0.25 to lock edges onto pixel centers."
		tip_offy = "Offset Y (fine): shifts the sampling grid inside the selection.\nDirection: +Y samples more downward (content appears to move up).\nExample: try +/-0.25 for subpixel alignment."
		tip_scale = "Scale (fine): tiny sampling scale correction.\n>1 samples a slightly larger area, <1 slightly smaller.\nExample: 1.002 or 0.998 when the source feels stretched."
		tip_panx = "Viewport pan X (coarse): moves the whole sampling window relative to the selection bounds.\nDirection: +X samples further right (content appears to move left).\nExample: +2 if the subject is 2px too far left in the selection."
		tip_pany = "Viewport pan Y (coarse): moves the whole sampling window relative to the selection bounds.\nDirection: +Y samples further down (content appears to move up).\nExample: -1 if the subject is 1px too low."
		tip_auto = "Auto Align: searches offset/scale to reduce blockiness and improve alignment.\nTip: start with Scale=1.0, then run Auto Align once, then fine-tune Offset X/Y."

		tip_live = "Live preview: re-renders automatically on every change.\nWarning: can be heavy on big selections.\nExample: keep off, use 'Render preview' after edits."
		tip_render = "Render preview: updates the preview once (manual mode).\nExample: tweak Offset X/Y, then click to see the result."
		tip_prev_zoom = "Preview zoom (visual only).\nHigher value = bigger pixels in the preview. Does not change output.\nExample: 10 draws each output pixel as 10x10."
		tip_fit = "Fit preview: clamps preview zoom so the whole output fits in the panel.\nExample: keeps 64x64 visible without scrolling."
		tip_show_grid = "Show pixel grid overlay in preview (visual only).\nExample: helps judge alignment."
		tip_grid_alpha = "Grid overlay opacity in preview (visual only).\nHigher = more visible.\nExample: 90 is readable but not too intrusive."

		tip_cut = "Alpha cutoff: sets alpha below this value to 0 (hard edges).\nHigher = more aggressive cut.\nExample: 225 makes anti-aliased edges snap to crisp pixels."
		tip_bin = "Binary alpha (remove semi-transparency): forces the FINAL result to only use alpha 0 or 255.\nPixels treated as background become alpha=0, everything else becomes alpha=255 (including outline pixels).\nThreshold uses Alpha cutoff (background = alpha < max(1, cutoff)).\nDirection: enabling increases solidity, removes soft edges and any semi-transparent outline.\nExample: cutoff 32 + Binary ON makes all remaining pixels fully opaque. Set cutoff=0 to treat only alpha==0 as background."
		tip_pal = "Max colors (palette limit). 0 = off.\nLower = stronger palette reduction.\nExample: 32 for UI icons, 16 for very stylized sprites."
		tip_pal_mode = "Palette mode: how the available colors are chosen.\nTop-N: picks the most frequent colors (fast, classic).\nK-Means clusters: groups colors into K clusters (better at keeping distinct materials like glass/tires/body).\nExample: K-Means + Preserve 60 for AI sprites."
		tip_pal_preserve = "Preserve rare colors 0..100 (diversity bias).\nHigher = tries harder to keep small but distinct color groups (e.g. blue windshield, black tires) instead of being swallowed by the dominant body color.\nExample: 50 keeps glass/tires while still reducing noise."
		tip_pal_oklab = "Perceptual mapping (Oklab): uses a perceptual color distance when snapping pixels to the palette.\nON = better matches for humans (less weird hue jumps), slightly slower.\nExample: enable when you see colors snapping to the wrong shade."
		tip_outline = "Outline silhouette: draws a border around the alpha-cut silhouette.\nExample: improves readability on busy backgrounds."
		tip_ow = "Outline width in output pixels.\nHigher = thicker border.\nExample: 1 for classic pixel outline."
		tip_oa = "Outline alpha (opacity).\nHigher = more solid.\nExample: 255 for solid."
		tip_oc = "Outline color.\nExample: black for classic sprite outlines."

		tip_new_img = "Output to new image (pixel size): creates a NEW image at grid size.\nDisables placement/fit/scale options for a true 1:1 export.\nExample: grid 64x64 creates a 64x64 image."
		tip_place = "Place output at selection position: new output layer is offset to selection top-left.\nExample: result lands exactly over the sampled region."
		tip_fit_out = "Fit output to selection size: scales the grid result to selection bounds (nearest).\nExample: grid 64x64 into a 128x128 selection."
		tip_replace = "Replace pixels in active layer (destructive): writes into the active layer inside selection bounds.\nWarning: this overwrites pixels (undoable via GIMP undo).\nExample: use to permanently clean a selected sprite region."
		tip_scale_out = "Scale output layer (nearest): enlarges the output layer after creation.\nExample: scale 4 turns 64x64 into 256x256."
		tip_scale_factor = "Output scale factor used with 'Scale output layer'.\nHigher = larger layer.\nExample: 8 for a big pixel preview."

		# Apply tooltips to preset controls (including the label)
		self._tip(lbl_preset, tip_preset)
		self._tip(self.combo_preset, tip_preset)
		self._tip(btn_save, tip_save)
		self._tip(btn_load, tip_load)
		self._tip(btn_del, tip_del)

		# ---------- Sampling frame ----------
		frm_sampling = mk_frame("Sampling")
		g_s = mk_grid(frm_sampling)
		r = 0
		self.spin_gw = tip2(Gtk.SpinButton.new_with_range(1, 512, 1), tip_grid_w)
		self.spin_gw.set_value(self.grid_w)
		self.spin_gh = tip2(Gtk.SpinButton.new_with_range(1, 512, 1), tip_grid_h)
		self.spin_gh.set_value(self.grid_h)
		add_label(g_s, "Grid width", 0, r, tip_grid_w)
		g_s.attach(self.spin_gw, 1, r, 1, 1)
		add_label(g_s, "Grid height", 2, r, tip_grid_h)
		g_s.attach(self.spin_gh, 3, r, 1, 1)
		r += 1
		self.combo_method = Gtk.ComboBoxText()
		self.combo_method.append("most_used", "Most Used (clustered)")
		self.combo_method.append("most_used_light", "Most Used (weighted light)")
		self.combo_method.append("most_used_dark", "Most Used (weighted dark)")
		self.combo_method.append("average", "Average")
		self.combo_method.append("neighbor", "Neighbor (margin avg)")
		self.combo_method.set_active_id(self.method)
		tip2(self.combo_method, tip_method)
		add_label(g_s, "Method", 0, r, tip_method)
		g_s.attach(self.combo_method, 1, r, 3, 1)
		r += 1
		self.spin_thr = tip2(Gtk.SpinButton.new_with_range(0, 255, 1), tip_sim)
		self.spin_thr.set_value(self.similarity_threshold)
		add_label(g_s, "Similarity", 0, r, tip_sim)
		g_s.attach(self.spin_thr, 1, r, 1, 1)
		self.spin_margin = Gtk.SpinButton.new_with_range(0.0, 2.0, 0.05)
		self.spin_margin.set_digits(2)
		self.spin_margin.set_value(self.neighbor_margin)
		tip2(self.spin_margin, tip_margin)
		add_label(g_s, "Neighbor margin", 2, r, tip_margin)
		g_s.attach(self.spin_margin, 3, r, 1, 1)
		r += 1
		self.check_denoise = Gtk.CheckButton(label="Pre denoise (reduce AI noise)")
		self.check_denoise.set_active(self.denoise_enabled)
		tip2(self.check_denoise, tip_denoise)
		g_s.attach(self.check_denoise, 0, r, 3, 1)
		self.spin_denoise = tip2(Gtk.SpinButton.new_with_range(0, 100, 1), tip_denoise_strength)
		self.spin_denoise.set_value(self.denoise_strength)
		add_label(g_s, "Strength", 3, r, tip_denoise_strength)
		g_s.attach(self.spin_denoise, 4, r, 1, 1)
		r += 1
		self.combo_denoise = Gtk.ComboBoxText()
		self.combo_denoise.append("trimmed", "Trimmed (recommended)")
		self.combo_denoise.append("median", "Median (strong)")
		self.combo_denoise.set_active_id(self.denoise_mode)
		tip2(self.combo_denoise, tip_denoise_mode)
		add_label(g_s, "Denoise mode", 0, r, tip_denoise_mode)
		g_s.attach(self.combo_denoise, 1, r, 3, 1)
		r += 1
		self.check_alpha = Gtk.CheckButton(label="Ignore transparent (alpha=0)")
		self.check_alpha.set_active(self.ignore_transparent)
		tip2(self.check_alpha, tip_ignore)
		g_s.attach(self.check_alpha, 0, r, 4, 1)
		r += 1
		self.check_bleed = Gtk.CheckButton(label="Fix transparent RGB (alpha bleed)")
		self.check_bleed.set_active(self.fix_transparent_rgb)
		tip2(self.check_bleed, tip_bleed)
		g_s.attach(self.check_bleed, 0, r, 3, 1)
		self.spin_bleed = tip2(Gtk.SpinButton.new_with_range(0, 16, 1), tip_bleed_r)
		self.spin_bleed.set_value(self.bleed_radius)
		add_label(g_s, "Bleed radius", 3, r, tip_bleed_r)
		g_s.attach(self.spin_bleed, 4, r, 1, 1)

		# ---------- Alignment frame ----------
		frm_align = mk_frame("Alignment")
		g_a = mk_grid(frm_align)
		r = 0
		self.spin_offx = Gtk.SpinButton.new_with_range(-512.0, 512.0, 0.05)
		self.spin_offx.set_digits(2)
		self.spin_offx.set_value(self.offset_x)
		tip2(self.spin_offx, tip_offx)
		self.spin_offy = Gtk.SpinButton.new_with_range(-512.0, 512.0, 0.05)
		self.spin_offy.set_digits(2)
		self.spin_offy.set_value(self.offset_y)
		tip2(self.spin_offy, tip_offy)
		add_label(g_a, "Offset X", 0, r, tip_offx)
		g_a.attach(self.spin_offx, 1, r, 1, 1)
		add_label(g_a, "Offset Y", 2, r, tip_offy)
		g_a.attach(self.spin_offy, 3, r, 1, 1)
		r += 1
		self.spin_scale = Gtk.SpinButton.new_with_range(0.90, 1.10, 0.001)
		self.spin_scale.set_digits(3)
		self.spin_scale.set_value(self.scale)
		tip2(self.spin_scale, tip_scale)
		add_label(g_a, "Scale", 0, r, tip_scale)
		g_a.attach(self.spin_scale, 1, r, 1, 1)
		r += 1
		self.spin_panx = Gtk.SpinButton.new_with_range(-2048.0, 2048.0, 0.5)
		self.spin_panx.set_digits(1)
		self.spin_panx.set_value(self.pan_x)
		tip2(self.spin_panx, tip_panx)
		self.spin_pany = Gtk.SpinButton.new_with_range(-2048.0, 2048.0, 0.5)
		self.spin_pany.set_digits(1)
		self.spin_pany.set_value(self.pan_y)
		tip2(self.spin_pany, tip_pany)
		add_label(g_a, "Viewport pan X", 0, r, tip_panx)
		g_a.attach(self.spin_panx, 1, r, 1, 1)
		add_label(g_a, "Viewport pan Y", 2, r, tip_pany)
		g_a.attach(self.spin_pany, 3, r, 1, 1)
		r += 1
		self.btn_auto = Gtk.Button(label="Auto Align")
		self.btn_auto.connect("clicked", self._on_auto_align_clicked)
		tip2(self.btn_auto, tip_auto)
		g_a.attach(self.btn_auto, 0, r, 4, 1)

		# ---------- Preview frame ----------
		frm_prev = mk_frame("Preview")
		g_p = mk_grid(frm_prev)
		r = 0
		self.check_live = Gtk.CheckButton(label="Live preview")
		self.check_live.set_active(self.live_preview)
		tip2(self.check_live, tip_live)
		g_p.attach(self.check_live, 0, r, 2, 1)
		self.btn_preview = Gtk.Button(label="Render preview")
		self.btn_preview.connect("clicked", self._on_render_preview_clicked)
		tip2(self.btn_preview, tip_render)
		g_p.attach(self.btn_preview, 2, r, 2, 1)
		r += 1
		self.spin_prevcell = tip2(Gtk.SpinButton.new_with_range(1, 32, 1), tip_prev_zoom)
		self.spin_prevcell.set_value(self.preview_cell)
		add_label(g_p, "Preview zoom", 0, r, tip_prev_zoom)
		g_p.attach(self.spin_prevcell, 1, r, 1, 1)
		r += 1
		self.check_fit = Gtk.CheckButton(label="Fit preview")
		self.check_fit.set_active(self.fit_preview)
		tip2(self.check_fit, tip_fit)
		g_p.attach(self.check_fit, 0, r, 2, 1)
		r += 1
		self.check_grid = Gtk.CheckButton(label="Show pixel grid")
		self.check_grid.set_active(self.show_grid)
		tip2(self.check_grid, tip_show_grid)
		g_p.attach(self.check_grid, 0, r, 2, 1)
		self.spin_grid_alpha = tip2(Gtk.SpinButton.new_with_range(0, 255, 5), tip_grid_alpha)
		self.spin_grid_alpha.set_value(self.grid_alpha)
		add_label(g_p, "Grid alpha", 2, r, tip_grid_alpha)
		g_p.attach(self.spin_grid_alpha, 3, r, 1, 1)

		# ---------- Post frame ----------
		frm_post = mk_frame("Post")
		g_post = mk_grid(frm_post)
		r = 0
		self.spin_cut = tip2(Gtk.SpinButton.new_with_range(0, 255, 1), tip_cut)
		self.spin_cut.set_value(self.alpha_cutoff)
		add_label(g_post, "Alpha cutoff", 0, r, tip_cut)
		g_post.attach(self.spin_cut, 1, r, 1, 1)
		r += 1
		self.check_bin_alpha = Gtk.CheckButton(label="Binary alpha (no semi-transparent pixels)")
		self.check_bin_alpha.set_active(self.binary_alpha)
		tip2(self.check_bin_alpha, tip_bin)
		g_post.attach(self.check_bin_alpha, 0, r, 4, 1)
		r += 1
		self.spin_pal = tip2(Gtk.SpinButton.new_with_range(0, 256, 1), tip_pal)
		self.spin_pal.set_value(self.palette_max)
		add_label(g_post, "Max colors", 0, r, tip_pal)
		g_post.attach(self.spin_pal, 1, r, 1, 1)
		r += 1

		# Palette mode + helpers
		self.combo_pal_mode = Gtk.ComboBoxText()
		self.combo_pal_mode.append('topn', 'Top-N (most frequent)')
		self.combo_pal_mode.append('kmeans', 'K-Means clusters (perceptual)')
		self.combo_pal_mode.append('median', 'Median cut (fallback)')
		self.combo_pal_mode.set_active_id(self.palette_mode)
		tip2(self.combo_pal_mode, tip_pal_mode)
		add_label(g_post, 'Palette mode', 0, r, tip_pal_mode)
		g_post.attach(self.combo_pal_mode, 1, r, 3, 1)
		r += 1
		self.spin_pal_preserve = tip2(Gtk.SpinButton.new_with_range(0, 100, 1), tip_pal_preserve)
		self.spin_pal_preserve.set_value(self.palette_preserve_rare)
		add_label(g_post, 'Preserve rare', 0, r, tip_pal_preserve)
		g_post.attach(self.spin_pal_preserve, 1, r, 1, 1)
		r += 1
		self.check_pal_oklab = Gtk.CheckButton(label='Perceptual mapping (Oklab)')
		self.check_pal_oklab.set_active(self.palette_use_oklab)
		tip2(self.check_pal_oklab, tip_pal_oklab)
		g_post.attach(self.check_pal_oklab, 0, r, 4, 1)
		r += 1

		# Separator: palette vs outline
		sep_post = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
		sep_post.set_margin_top(4)
		sep_post.set_margin_bottom(4)
		g_post.attach(sep_post, 0, r, 4, 1)
		r += 1

		self.check_outline = Gtk.CheckButton(label="Outline silhouette")
		self.check_outline.set_active(self.outline_enabled)
		tip2(self.check_outline, tip_outline)
		g_post.attach(self.check_outline, 0, r, 2, 1)
		r += 1
		self.spin_ow = tip2(Gtk.SpinButton.new_with_range(1, 8, 1), tip_ow)
		self.spin_ow.set_value(self.outline_width)
		add_label(g_post, "Outline width", 0, r, tip_ow)
		g_post.attach(self.spin_ow, 1, r, 1, 1)
		r += 1
		self.spin_oa = tip2(Gtk.SpinButton.new_with_range(0, 255, 5), tip_oa)
		self.spin_oa.set_value(self.outline_alpha)
		add_label(g_post, "Outline alpha", 0, r, tip_oa)
		g_post.attach(self.spin_oa, 1, r, 1, 1)
		r += 1
		self.color_outline = Gtk.ColorButton()
		self.color_outline.set_use_alpha(False)
		self.color_outline.set_rgba(self._rgb_to_rgba(self.outline_color))
		tip2(self.color_outline, tip_oc)
		add_label(g_post, "Outline color", 0, r, tip_oc)
		g_post.attach(self.color_outline, 1, r, 1, 1)

		# ---------- Output frame ----------
		frm_out = mk_frame("Output")
		g_o = mk_grid(frm_out)
		r = 0
		self.check_new_image = Gtk.CheckButton(label="Output to new image (pixel size)")
		self.check_new_image.set_active(self.output_to_new_image)
		tip2(self.check_new_image, tip_new_img)
		g_o.attach(self.check_new_image, 0, r, 4, 1)
		r += 1
		self.check_place_output = Gtk.CheckButton(label="Place output at selection position")
		self.check_place_output.set_active(self.place_output_on_selection)
		tip2(self.check_place_output, tip_place)
		g_o.attach(self.check_place_output, 0, r, 4, 1)
		r += 1
		self.check_fit_output = Gtk.CheckButton(label="Fit output to selection size")
		self.check_fit_output.set_active(self.fit_output_to_selection)
		tip2(self.check_fit_output, tip_fit_out)
		g_o.attach(self.check_fit_output, 0, r, 4, 1)
		r += 1
		self.check_replace = Gtk.CheckButton(label="Replace pixels in active layer (bounds, destructive)")
		self.check_replace.set_active(self.replace_in_active_layer)
		tip2(self.check_replace, tip_replace)
		g_o.attach(self.check_replace, 0, r, 4, 1)
		r += 1
		self.check_scale_output = Gtk.CheckButton(label="Scale output layer")
		self.check_scale_output.set_active(self.scale_output_enabled)
		tip2(self.check_scale_output, tip_scale_out)
		g_o.attach(self.check_scale_output, 0, r, 2, 1)
		self.spin_outscale = tip2(Gtk.SpinButton.new_with_range(1, 64, 1), tip_scale_factor)
		self.spin_outscale.set_value(self.output_scale)
		add_label(g_o, "Scale factor", 2, r, tip_scale_factor)
		g_o.attach(self.spin_outscale, 3, r, 1, 1)
		self._update_scale_output_sensitivity()
		self._update_replace_sensitivity()

		# Dialog buttons
		self.add_button("Cancel", Gtk.ResponseType.CANCEL)
		self.add_button("Apply", Gtk.ResponseType.OK)

		# Connect change signals
		widgets = [
			self.spin_gw, self.spin_gh, self.combo_method, self.spin_thr, self.spin_margin,
			self.check_denoise, self.spin_denoise, self.combo_denoise,
			self.check_alpha, self.check_bleed, self.spin_bleed,
			self.spin_offx, self.spin_offy, self.spin_scale,
			self.spin_panx, self.spin_pany,
			self.spin_prevcell, self.check_fit, self.check_grid, self.spin_grid_alpha,
			self.spin_cut, self.spin_pal, self.combo_pal_mode, self.spin_pal_preserve, self.check_pal_oklab,
			self.check_outline, self.spin_ow, self.spin_oa, self.color_outline,
			self.check_new_image,
			self.check_scale_output, self.spin_outscale,
			self.check_place_output, self.check_fit_output, self.check_replace,
			self.check_live,
		]
		for w in widgets:
			if isinstance(w, Gtk.SpinButton):
				w.connect("value-changed", self._on_params_changed)
			elif isinstance(w, Gtk.ComboBoxText):
				w.connect("changed", self._on_params_changed)
			elif isinstance(w, Gtk.CheckButton):
				w.connect("toggled", self._on_params_changed)
			elif isinstance(w, Gtk.ColorButton):
				w.connect("color-set", self._on_params_changed)

		self.check_new_image.connect("toggled", self._on_new_image_toggled)
		self.check_scale_output.connect("toggled", self._on_scale_output_toggled)
		self.check_replace.connect("toggled", self._on_replace_toggled)

		self.show_all()
		self._update_new_image_sensitivity()
		try:
			self._update_denoise_sensitivity()
		except Exception:
			pass
		try:
			self.btn_preview.set_sensitive(not bool(self.check_live.get_active()))
		except Exception:
			pass

	def _rgb_to_rgba(self, rgb):
		try:
			from gi.repository import Gdk
			c = Gdk.RGBA()
			c.red = float(rgb[0]) / 255.0
			c.green = float(rgb[1]) / 255.0
			c.blue = float(rgb[2]) / 255.0
			c.alpha = 1.0
			return c
		except Exception:
			class _C(object):
				pass
			c = _C()
			c.red = float(rgb[0]) / 255.0
			c.green = float(rgb[1]) / 255.0
			c.blue = float(rgb[2]) / 255.0
			c.alpha = 1.0
			return c

	def _on_render_preview_clicked(self, _btn):
		self._queue_preview_update(force=True)

	def _on_new_image_toggled(self, _w):
		enabled = bool(self.check_new_image.get_active())
		if enabled:
			# New-image output is meant as "pixel size" export. Disable the
			# other placement/scaling modes to avoid surprising results.
			try:
				self.check_fit_output.set_active(False)
			except Exception:
				pass
			try:
				self.check_place_output.set_active(False)
			except Exception:
				pass
			try:
				self.check_replace.set_active(False)
			except Exception:
				pass
			try:
				self.check_scale_output.set_active(False)
			except Exception:
				pass
		self._update_new_image_sensitivity()
		try:
			self._update_denoise_sensitivity()
		except Exception:
			pass
		self._on_params_changed(_w)

	
	def _update_denoise_sensitivity(self):
		enabled = bool(getattr(self, 'check_denoise', None) and self.check_denoise.get_active())
		for w in [getattr(self, 'spin_denoise', None), getattr(self, 'combo_denoise', None)]:
			if w is None:
				continue
			try:
				w.set_sensitive(enabled)
			except Exception:
				pass

	def _update_new_image_sensitivity(self):
		enabled = bool(getattr(self, 'check_new_image', None) and self.check_new_image.get_active())
		# Controls that do not make sense for new-image output.
		for w in [
			getattr(self, 'check_place_output', None),
			getattr(self, 'check_fit_output', None),
			getattr(self, 'check_replace', None),
			getattr(self, 'check_scale_output', None),
			getattr(self, 'spin_outscale', None),
		]:
			if w is None:
				continue
			try:
				w.set_sensitive(not enabled)
			except Exception:
				pass

	def _on_scale_output_toggled(self, _w):
		self._update_scale_output_sensitivity()
		self._on_params_changed(_w)

	def _on_replace_toggled(self, _w):
		enabled = bool(self.check_replace.get_active())
		if enabled:
			try:
				self.check_fit_output.set_active(True)
			except Exception:
				pass
			try:
				self.check_scale_output.set_active(False)
			except Exception:
				pass
		self._update_replace_sensitivity()
		self._update_scale_output_sensitivity()
		self._update_new_image_sensitivity()
		try:
			self._update_denoise_sensitivity()
		except Exception:
			pass
		self._on_params_changed(_w)

	def _update_replace_sensitivity(self):
		replace_enabled = bool(getattr(self, 'check_replace', None) and self.check_replace.get_active())
		try:
			self.check_fit_output.set_sensitive(not replace_enabled)
		except Exception:
			pass
		try:
			self.check_scale_output.set_sensitive(not replace_enabled)
		except Exception:
			pass
		try:
			self.spin_outscale.set_sensitive(bool(self.check_scale_output.get_active()) and not replace_enabled)
		except Exception:
			pass

	def _update_scale_output_sensitivity(self):
		replace_enabled = bool(getattr(self, 'check_replace', None) and self.check_replace.get_active())
		enabled = bool(self.check_scale_output.get_active()) and not replace_enabled
		self.spin_outscale.set_sensitive(enabled)
		try:
			self.check_scale_output.set_sensitive(not replace_enabled)
		except Exception:
			pass

	def _on_preview_resized(self, *_args):
		# Resizes can spam size-allocate; only re-render automatically in live mode.
		self.live_preview = bool(self.check_live.get_active())
		if self.live_preview:
			self._queue_preview_update(force=True)
		else:
			self.preview_needs_update = True

	def _refresh_preset_combo(self):
		self.combo_preset.remove_all()
		self.combo_preset.append("__none__", "(none)")
		for k in sorted(list(self.presets.keys())):
			self.combo_preset.append(k, k)
		self.combo_preset.set_active_id("__none__")

	def _prompt_text(self, title, message, default_text, select_all=False):
		d = Gtk.Dialog(title=title, transient_for=self, flags=0)
		d.add_button("Cancel", Gtk.ResponseType.CANCEL)
		d.add_button("OK", Gtk.ResponseType.OK)
		box = d.get_content_area()
		box.set_spacing(8)
		lbl = Gtk.Label(label=message)
		lbl.set_xalign(0.0)
		box.add(lbl)
		entry = Gtk.Entry()
		entry.set_text(default_text)
		if select_all:
			try:
				entry.select_region(0, -1)
			except Exception:
				pass
		box.add(entry)
		d.show_all()
		resp = d.run()
		txt = entry.get_text() if resp == Gtk.ResponseType.OK else None
		d.destroy()
		return txt

	def _current_settings_dict(self):
		c = self.color_outline.get_rgba()
		return {
			"grid_w": self.grid_w,
			"grid_h": self.grid_h,
			"method": self.method,
			"similarity_threshold": self.similarity_threshold,
			"ignore_transparent": self.ignore_transparent,
			"denoise_enabled": self.denoise_enabled,
			"denoise_strength": self.denoise_strength,
			"denoise_mode": self.denoise_mode,
			"fix_transparent_rgb": self.fix_transparent_rgb,
			"bleed_radius": self.bleed_radius,
			"neighbor_margin": self.neighbor_margin,
			"offset_x": self.offset_x,
			"offset_y": self.offset_y,
			"scale": self.scale,
			"pan_x": self.pan_x,
			"pan_y": self.pan_y,
			"preview_cell": self.preview_cell,
			"fit_preview": self.fit_preview,
			"show_grid": self.show_grid,
			"grid_alpha": self.grid_alpha,
			# Intentionally NOT stored in presets: live preview is a session preference.
			"alpha_cutoff": self.alpha_cutoff,
			"binary_alpha": self.binary_alpha,
			"palette_max": self.palette_max,
			"palette_mode": self.palette_mode,
			"palette_preserve_rare": self.palette_preserve_rare,
			"palette_use_oklab": self.palette_use_oklab,
			"outline_enabled": self.outline_enabled,
			"outline_width": self.outline_width,
			"outline_alpha": self.outline_alpha,
			"outline_color": [int(c.red * 255), int(c.green * 255), int(c.blue * 255)],
			"output_to_new_image": self.output_to_new_image,
			"scale_output_enabled": self.scale_output_enabled,
			"output_scale": self.output_scale,
			"place_output_on_selection": self.place_output_on_selection,
			"fit_output_to_selection": self.fit_output_to_selection,
			"replace_in_active_layer": self.replace_in_active_layer,
		}

	def _apply_settings_dict(self, d):
		def g(key, default):
			return d[key] if key in d else default
		self.spin_gw.set_value(float(g("grid_w", self.grid_w)))
		self.spin_gh.set_value(float(g("grid_h", self.grid_h)))
		self.combo_method.set_active_id(str(g("method", self.method)))
		self.spin_thr.set_value(float(g("similarity_threshold", self.similarity_threshold)))
		self.spin_margin.set_value(float(g("neighbor_margin", self.neighbor_margin)))
		self.check_denoise.set_active(bool(g("denoise_enabled", self.denoise_enabled)))
		self.spin_denoise.set_value(float(g("denoise_strength", self.denoise_strength)))
		self.combo_denoise.set_active_id(str(g("denoise_mode", self.denoise_mode)))
		try:
			self._update_denoise_sensitivity()
		except Exception:
			pass
		self.check_alpha.set_active(bool(g("ignore_transparent", self.ignore_transparent)))
		self.check_bleed.set_active(bool(g("fix_transparent_rgb", self.fix_transparent_rgb)))
		self.spin_bleed.set_value(float(g("bleed_radius", self.bleed_radius)))
		self.spin_offx.set_value(float(g("offset_x", self.offset_x)))
		self.spin_offy.set_value(float(g("offset_y", self.offset_y)))
		self.spin_scale.set_value(float(g("scale", self.scale)))
		self.spin_panx.set_value(float(g("pan_x", self.pan_x)))
		self.spin_pany.set_value(float(g("pan_y", self.pan_y)))
		self.spin_prevcell.set_value(float(g("preview_cell", self.preview_cell)))
		self.check_fit.set_active(bool(g("fit_preview", self.fit_preview)))
		self.check_grid.set_active(bool(g("show_grid", self.show_grid)))
		self.spin_grid_alpha.set_value(float(g("grid_alpha", self.grid_alpha)))
		# Do not touch live preview when loading presets.
		self.spin_cut.set_value(float(g("alpha_cutoff", self.alpha_cutoff)))
		try:
			self.check_bin_alpha.set_active(bool(g("binary_alpha", self.binary_alpha)))
		except Exception:
			pass
		self.spin_pal.set_value(float(g("palette_max", self.palette_max)))
		try:
			self.combo_pal_mode.set_active_id(str(g("palette_mode", self.palette_mode)))
		except Exception:
			pass
		self.spin_pal_preserve.set_value(float(g("palette_preserve_rare", self.palette_preserve_rare)))
		self.check_pal_oklab.set_active(bool(g("palette_use_oklab", self.palette_use_oklab)))

		self.check_outline.set_active(bool(g("outline_enabled", self.outline_enabled)))
		self.spin_ow.set_value(float(g("outline_width", self.outline_width)))
		self.spin_oa.set_value(float(g("outline_alpha", self.outline_alpha)))
		oc = g("outline_color", [0, 0, 0])
		self.color_outline.set_rgba(self._rgb_to_rgba((int(oc[0]), int(oc[1]), int(oc[2]))))

		self.check_new_image.set_active(bool(g("output_to_new_image", self.output_to_new_image)))
		self.check_scale_output.set_active(bool(g("scale_output_enabled", self.scale_output_enabled)))
		self.spin_outscale.set_value(float(g("output_scale", self.output_scale)))
		self.check_place_output.set_active(bool(g("place_output_on_selection", self.place_output_on_selection)))
		self.check_fit_output.set_active(bool(g("fit_output_to_selection", self.fit_output_to_selection)))
		self.check_replace.set_active(bool(g("replace_in_active_layer", self.replace_in_active_layer)))
		self._update_replace_sensitivity()
		self._update_scale_output_sensitivity()

		try:
			self.check_place_output.set_active(bool(g("place_output_on_selection", self.place_output_on_selection)))
		except Exception:
			pass
		try:
			self.check_fit_output.set_active(bool(g("fit_output_to_selection", self.fit_output_to_selection)))
		except Exception:
			pass
		try:
			self.check_replace.set_active(bool(g("replace_in_active_layer", self.replace_in_active_layer)))
		except Exception:
			pass
		self._update_replace_sensitivity()

	def _on_preset_save(self, _btn):
		self._read_params()
		cur = self.combo_preset.get_active_id()
		default_name = cur if (cur is not None and cur != "__none__") else "my_preset"
		name = self._prompt_text("Save preset", "Preset name:", default_name, select_all=True)
		if name is None:
			return
		name = name.strip()
		if name == "" or name == "(none)":
			return
		self.presets[name] = self._current_settings_dict()
		_save_presets(self.presets)
		self._refresh_preset_combo()
		self.combo_preset.set_active_id(name)

	def _on_preset_load(self, _btn):
		pid = self.combo_preset.get_active_id()
		if pid is None or pid == "__none__":
			return
		if pid in self.presets:
			self._apply_settings_dict(self.presets[pid])
			self._queue_preview_update(force=True)

	def _on_preset_selected(self, _combo):
		# Auto-load when selecting a preset in the dropdown.
		if bool(getattr(self, "_preset_ui_lock", False)):
			return
		pid = self.combo_preset.get_active_id()
		if pid is None or pid == "__none__":
			return
		if pid in self.presets:
			self._preset_ui_lock = True
			try:
				self._apply_settings_dict(self.presets[pid])
			finally:
				self._preset_ui_lock = False
			self._queue_preview_update(force=True)

	def _on_preset_delete(self, _btn):
		pid = self.combo_preset.get_active_id()
		if pid is None or pid == "__none__":
			return
		if pid in self.presets:
			del self.presets[pid]
			_save_presets(self.presets)
			self._refresh_preset_combo()
			self._queue_preview_update(force=True)

	def _on_preset_selected(self, _combo):
		# Auto-load preset when selected in the dropdown.
		# Uses a small lock to avoid recursion when we programmatically change
		# combo selection during save/refresh operations.
		if bool(getattr(self, "_preset_ui_lock", False)):
			return
		pid = self.combo_preset.get_active_id()
		if pid is None or pid == "__none__":
			return
		if pid in self.presets:
			try:
				self._preset_ui_lock = True
				self._apply_settings_dict(self.presets[pid])
			finally:
				self._preset_ui_lock = False
			self._queue_preview_update(force=True)

	def _read_params(self):
		self.grid_w = int(self.spin_gw.get_value())
		self.grid_h = int(self.spin_gh.get_value())
		self.method = str(self.combo_method.get_active_id())
		self.similarity_threshold = int(self.spin_thr.get_value())
		self.neighbor_margin = float(self.spin_margin.get_value())
		self.denoise_enabled = bool(self.check_denoise.get_active())
		self.denoise_strength = int(self.spin_denoise.get_value())
		self.denoise_mode = str(self.combo_denoise.get_active_id())
		self.ignore_transparent = bool(self.check_alpha.get_active())
		self.fix_transparent_rgb = bool(self.check_bleed.get_active())
		self.bleed_radius = int(self.spin_bleed.get_value())
		self.offset_x = float(self.spin_offx.get_value())
		self.offset_y = float(self.spin_offy.get_value())
		self.scale = float(self.spin_scale.get_value())
		self.pan_x = float(self.spin_panx.get_value())
		self.pan_y = float(self.spin_pany.get_value())
		self.preview_cell = int(self.spin_prevcell.get_value())
		self.fit_preview = bool(self.check_fit.get_active())
		self.show_grid = bool(self.check_grid.get_active())
		self.grid_alpha = int(self.spin_grid_alpha.get_value())
		self.live_preview = bool(self.check_live.get_active())
		self.alpha_cutoff = int(self.spin_cut.get_value())
		self.binary_alpha = bool(self.check_bin_alpha.get_active())
		self.palette_max = int(self.spin_pal.get_value())
		self.palette_mode = str(self.combo_pal_mode.get_active_id())
		self.palette_preserve_rare = int(self.spin_pal_preserve.get_value())
		self.palette_use_oklab = bool(self.check_pal_oklab.get_active())
		self.outline_enabled = bool(self.check_outline.get_active())
		self.outline_width = int(self.spin_ow.get_value())
		self.outline_alpha = int(self.spin_oa.get_value())
		c = self.color_outline.get_rgba()
		self.outline_color = (int(c.red * 255), int(c.green * 255), int(c.blue * 255))
		self.output_to_new_image = bool(self.check_new_image.get_active())
		self.scale_output_enabled = bool(self.check_scale_output.get_active())
		self.output_scale = int(self.spin_outscale.get_value())
		self.output_to_new_image = bool(self.check_new_image.get_active())
		self.place_output_on_selection = bool(self.check_place_output.get_active())
		self.fit_output_to_selection = bool(self.check_fit_output.get_active())
		self.replace_in_active_layer = bool(self.check_replace.get_active())

	def _on_params_changed(self, _widget):
		# Don't rely on a cached flag here, toggles come in before _read_params runs.
		try:
			self._update_replace_sensitivity()
		except Exception:
			pass
		try:
			self._update_new_image_sensitivity()
		except Exception:
			pass
		try:
			self._update_denoise_sensitivity()
		except Exception:
			pass
		self.live_preview = bool(self.check_live.get_active())
		try:
			self.btn_preview.set_sensitive(not self.live_preview)
		except Exception:
			pass
		if self.live_preview:
			self._queue_preview_update(force=False)
		else:
			self.preview_needs_update = True

	def _queue_preview_update(self, force=False):
		self.preview_needs_update = True
		if self._preview_update_source_id != 0:
			return
		self._preview_update_source_id = GLib.timeout_add(40, self._update_preview)

	def _pick_integer_zoom_to_fit(self, target_w, target_h, panel_w, panel_h, requested):
		if target_w <= 0 or target_h <= 0:
			return max(1, int(requested))
		if not self.fit_preview:
			return max(1, int(requested))
		max_step = min(panel_w // target_w, panel_h // target_h)
		max_step = max(1, int(max_step))
		return max(1, min(int(requested), max_step))

	def _update_preview(self):
		self._preview_update_source_id = 0
		if not self.preview_needs_update:
			return False
		self.preview_needs_update = False
		self._read_params()

		sx0 = _clamp_int(int(round(self._sel_x + self.pan_x)), 0, max(0, self._drawable.get_width() - 1))
		sy0 = _clamp_int(int(round(self._sel_y + self.pan_y)), 0, max(0, self._drawable.get_height() - 1))
		sw = max(1, self._sel_w)
		sh = max(1, self._sel_h)

		src_pw = max(16, self.src_area.get_allocated_width())
		src_ph = max(16, self.src_area.get_allocated_height())
		out_pw = max(16, self.out_area.get_allocated_width())
		out_ph = max(16, self.out_area.get_allocated_height())

		src_rgba, src_w, src_h = _read_drawable_region_rgba(self._drawable, sx0, sy0, sw, sh)
		src_zoom = self._pick_integer_zoom_to_fit(src_w, src_h, src_pw, src_ph, 16)
		src_up, src_up_w, src_up_h = _nearest_scale_rgba(src_rgba, src_w, src_h, src_zoom)
		src_canvas = _compose_checker(src_pw, src_ph)
		_blit_center(src_canvas, src_pw, src_ph, src_up, src_up_w, src_up_h)
		self.src_area.draw(0, 0, src_pw, src_ph, Gimp.ImageType.RGBA_IMAGE, bytes(src_canvas), src_pw * 4)

		grid_rgba = _render_grid_from_source(
			self._drawable, sx0, sy0, sw, sh,
			self.grid_w, self.grid_h,
			self.offset_x, self.offset_y,
			self.scale,
			self.method,
			self.similarity_threshold,
			self.ignore_transparent,
			self.neighbor_margin,
			self.denoise_enabled,
			self.denoise_strength,
			self.denoise_mode
		)
		grid_rgba = _alpha_cut(grid_rgba, self.alpha_cutoff)
		if self.palette_max > 0:
			grid_rgba = ppa_palette.apply_palette_quantization(grid_rgba, self.palette_max, self.palette_mode, self.palette_preserve_rare, self.palette_use_oklab)
		if self.outline_enabled:
			grid_rgba = _apply_outline(grid_rgba, self.grid_w, self.grid_h, self.alpha_cutoff, self.outline_width, self.outline_color, self.outline_alpha)
		grid_rgba = _binary_alpha(grid_rgba, self.alpha_cutoff, self.binary_alpha)
		grid_rgba = _apply_output_bleed(grid_rgba, self.grid_w, self.grid_h, self.fix_transparent_rgb, self.bleed_radius, self.alpha_cutoff)

		step = self._pick_integer_zoom_to_fit(self.grid_w, self.grid_h, out_pw, out_ph, self.preview_cell)
		up_rgba, up_w, up_h = _nearest_scale_rgba(grid_rgba, self.grid_w, self.grid_h, step)

		out_canvas = _compose_checker(out_pw, out_ph)
		ox, oy = _blit_center(out_canvas, out_pw, out_ph, up_rgba, up_w, up_h)
		if self.show_grid:
			_overlay_grid(out_canvas, out_pw, out_ph, ox, oy, step, self.grid_alpha)

		self.out_area.draw(0, 0, out_pw, out_ph, Gimp.ImageType.RGBA_IMAGE, bytes(out_canvas), out_pw * 4)
		return False

	def _alignment_score(self, src_x, src_y, src_w, src_h, offset_x, offset_y, scale, stride_hint):
		buf = self._drawable.get_buffer()
		rect = Gegl.Rectangle.new(src_x, src_y, src_w, src_h)
		src_bytes = buf.get(rect, 1.0, "RGBA u8", Gegl.AbyssPolicy.CLAMP)
		mv = memoryview(src_bytes)
		rowstride = src_w * 4
		samp_w = float(src_w) * float(scale)
		samp_h = float(src_h) * float(scale)
		cell_w = samp_w / float(max(1, self.grid_w))
		cell_h = samp_h / float(max(1, self.grid_h))
		ncells = self.grid_w * self.grid_h
		sum_r = [0.0] * ncells
		sum_g = [0.0] * ncells
		sum_b = [0.0] * ncells
		cnt = [0.0] * ncells
		step = max(1, int(stride_hint))
		for y in range(0, src_h, step):
			py = float(y) + 0.5
			j = int((py - offset_y) // cell_h)
			if j < 0 or j >= self.grid_h:
				continue
			for x in range(0, src_w, step):
				px = float(x) + 0.5
				i = int((px - offset_x) // cell_w)
				if i < 0 or i >= self.grid_w:
					continue
				off = (y * rowstride) + (x * 4)
				a = int(mv[off + 3])
				if a == 0:
					continue
				r = int(mv[off])
				g = int(mv[off + 1])
				b = int(mv[off + 2])
				k = (j * self.grid_w) + i
				sum_r[k] += float(r)
				sum_g[k] += float(g)
				sum_b[k] += float(b)
				cnt[k] += 1.0
		mean_r = [0.0] * ncells
		mean_g = [0.0] * ncells
		mean_b = [0.0] * ncells
		for k in range(ncells):
			if cnt[k] > 0.0:
				mean_r[k] = sum_r[k] / cnt[k]
				mean_g[k] = sum_g[k] / cnt[k]
				mean_b[k] = sum_b[k] / cnt[k]
		err = 0.0
		for y in range(0, src_h, step):
			py = float(y) + 0.5
			j = int((py - offset_y) // cell_h)
			if j < 0 or j >= self.grid_h:
				continue
			for x in range(0, src_w, step):
				px = float(x) + 0.5
				i = int((px - offset_x) // cell_w)
				if i < 0 or i >= self.grid_w:
					continue
				off = (y * rowstride) + (x * 4)
				a = int(mv[off + 3])
				if a == 0:
					continue
				r = float(int(mv[off]))
				g = float(int(mv[off + 1]))
				b = float(int(mv[off + 2]))
				k = (j * self.grid_w) + i
				dr = r - mean_r[k]
				dg = g - mean_g[k]
				db = b - mean_b[k]
				err += (dr * dr) + (dg * dg) + (db * db)
		contrast = 0.0
		for j in range(self.grid_h):
			for i in range(self.grid_w):
				k = (j * self.grid_w) + i
				r0 = mean_r[k]
				g0 = mean_g[k]
				b0 = mean_b[k]
				if i + 1 < self.grid_w:
					k2 = (j * self.grid_w) + (i + 1)
					contrast += abs(r0 - mean_r[k2]) + abs(g0 - mean_g[k2]) + abs(b0 - mean_b[k2])
				if j + 1 < self.grid_h:
					k2 = ((j + 1) * self.grid_w) + i
					contrast += abs(r0 - mean_r[k2]) + abs(g0 - mean_g[k2]) + abs(b0 - mean_b[k2])
		return err - (contrast * 0.35)

	def _on_auto_align_clicked(self, _btn):
		self._read_params()
		sx0 = _clamp_int(int(round(self._sel_x + self.pan_x)), 0, max(0, self._drawable.get_width() - 1))
		sy0 = _clamp_int(int(round(self._sel_y + self.pan_y)), 0, max(0, self._drawable.get_height() - 1))
		sw = max(1, self._sel_w)
		sh = max(1, self._sel_h)
		cell_w = (float(sw) * float(self.scale)) / float(max(1, self.grid_w))
		cell_h = (float(sh) * float(self.scale)) / float(max(1, self.grid_h))
		rx = max(0.5, min(16.0, cell_w * 0.55))
		ry = max(0.5, min(16.0, cell_h * 0.55))
		stride_hint = int(max(1.0, min(6.0, min(cell_w, cell_h) / 2.0)))
		def eval_point(ox, oy, sc):
			return self._alignment_score(sx0, sy0, sw, sh, ox, oy, sc, stride_hint)
		best_ox = self.offset_x
		best_oy = self.offset_y
		best_sc = self.scale
		best_score = float("inf")
		steps = 7
		for si in [-1, 0, 1]:
			sc = _clamp(self.scale + (si * 0.01), 0.90, 1.10)
			for y in range(steps):
				oy = self.offset_y + (-ry + (2.0 * ry) * (float(y) / float(steps - 1)))
				for x in range(steps):
					ox = self.offset_x + (-rx + (2.0 * rx) * (float(x) / float(steps - 1)))
					s = eval_point(ox, oy, sc)
					if s < best_score:
						best_score = s
						best_ox, best_oy, best_sc = ox, oy, sc
		for _ in range(3):
			rx *= 0.45
			ry *= 0.45
			for si in [-1, 0, 1]:
				sc = _clamp(best_sc + (si * 0.003), 0.90, 1.10)
				for y in range(steps):
					oy = best_oy + (-ry + (2.0 * ry) * (float(y) / float(steps - 1)))
					for x in range(steps):
						ox = best_ox + (-rx + (2.0 * rx) * (float(x) / float(steps - 1)))
						s = eval_point(ox, oy, sc)
						if s < best_score:
							best_score = s
							best_ox, best_oy, best_sc = ox, oy, sc
		self.spin_offx.set_value(best_ox)
		self.spin_offy.set_value(best_oy)
		self.spin_scale.set_value(best_sc)
		self._queue_preview_update(force=True)


def _set_layer_offsets(layer, x, y):
	ix = int(x)
	iy = int(y)
	try:
		layer.set_offsets(ix, iy)
		return
	except Exception:
		pass
	try:
		layer.set_offset(ix, iy)
		return
	except Exception:
		pass
	try:
		layer.set_offset_x(ix)
		layer.set_offset_y(iy)
	except Exception:
		pass


def _apply_to_image(image, drawable, selection_bounds, dlg):
	sx, sy, sw, sh = selection_bounds
	sw = max(1, sw)
	sh = max(1, sh)
	sx = _clamp_int(int(round(float(sx) + float(dlg.pan_x))), 0, max(0, drawable.get_width() - 1))
	sy = _clamp_int(int(round(float(sy) + float(dlg.pan_y))), 0, max(0, drawable.get_height() - 1))

	image.undo_group_start()
	try:
		grid_rgba = _render_grid_from_source(
			drawable, sx, sy, sw, sh,
			dlg.grid_w, dlg.grid_h,
			dlg.offset_x, dlg.offset_y,
			dlg.scale,
			dlg.method,
			dlg.similarity_threshold,
			dlg.ignore_transparent,
			dlg.neighbor_margin,
			dlg.denoise_enabled,
			dlg.denoise_strength,
			dlg.denoise_mode
		)
		grid_rgba = _alpha_cut(grid_rgba, dlg.alpha_cutoff)
		if dlg.palette_max > 0:
			grid_rgba = ppa_palette.apply_palette_quantization(grid_rgba, dlg.palette_max, getattr(dlg, "palette_mode", "topn"), getattr(dlg, "palette_preserve_rare", 40), getattr(dlg, "palette_use_oklab", False))
		if dlg.outline_enabled:
			grid_rgba = _apply_outline(grid_rgba, dlg.grid_w, dlg.grid_h, dlg.alpha_cutoff, dlg.outline_width, dlg.outline_color, dlg.outline_alpha)
		grid_rgba = _binary_alpha(grid_rgba, dlg.alpha_cutoff, getattr(dlg, "binary_alpha", False))
		grid_rgba = _apply_output_bleed(grid_rgba, dlg.grid_w, dlg.grid_h, dlg.fix_transparent_rgb, dlg.bleed_radius, dlg.alpha_cutoff)

		out_rgba = grid_rgba
		out_w = int(dlg.grid_w)
		out_h = int(dlg.grid_h)

		# Option: output as a NEW image at pixel size.
		if bool(getattr(dlg, 'output_to_new_image', False)):
			new_img = None
			try:
				new_img = Gimp.Image.new(out_w, out_h, Gimp.ImageBaseType.RGB)
			except Exception:
				new_img = None
			if new_img is not None:
				try:
					layer = Gimp.Layer.new(
						new_img,
						"PixelPerfect_Aligned",
						out_w,
						out_h,
						Gimp.ImageType.RGBA_IMAGE,
						100.0,
						Gimp.LayerMode.NORMAL
					)
					new_img.insert_layer(layer, None, 0)
					out_buffer = layer.get_buffer()
					rect = Gegl.Rectangle.new(0, 0, out_w, out_h)
					out_buffer.set(rect, "RGBA u8", out_rgba)
					out_buffer.flush()
					layer.update(0, 0, out_w, out_h)
					try:
						Gimp.Display.new(new_img)
					except Exception:
						pass
					try:
						Gimp.displays_flush()
					except Exception:
						pass
					return
				except Exception:
					# If anything fails, fall back to normal layer output.
					pass
		if dlg.fit_output_to_selection or dlg.replace_in_active_layer:
			out_rgba = _scale_rgba_to_size(out_rgba, out_w, out_h, sw, sh)
			out_w = int(sw)
			out_h = int(sh)

		if dlg.replace_in_active_layer:
			buf = drawable.get_buffer()
			rect = Gegl.Rectangle.new(sx, sy, out_w, out_h)
			buf.set(rect, "RGBA u8", out_rgba)
			buf.flush()
			try:
				drawable.update(sx, sy, out_w, out_h)
			except Exception:
				pass
		else:
			layer = Gimp.Layer.new(
				image,
				"PixelPerfect_Aligned",
				out_w,
				out_h,
				Gimp.ImageType.RGBA_IMAGE,
				100.0,
				Gimp.LayerMode.NORMAL
			)
			image.insert_layer(layer, None, 0)
			if dlg.place_output_on_selection:
				_set_layer_offsets(layer, sx, sy)

			out_buffer = layer.get_buffer()
			rect = Gegl.Rectangle.new(0, 0, out_w, out_h)
			out_buffer.set(rect, "RGBA u8", out_rgba)
			out_buffer.flush()
			layer.update(0, 0, out_w, out_h)

			if dlg.scale_output_enabled and dlg.output_scale != 1:
				old_interp = None
				try:
					old_interp = Gimp.context_get_interpolation()
				except Exception:
					old_interp = None
				try:
					Gimp.context_set_interpolation(Gimp.InterpolationType.NONE)
					layer.scale(out_w * dlg.output_scale, out_h * dlg.output_scale, False)
				finally:
					if old_interp is not None:
						try:
							Gimp.context_set_interpolation(old_interp)
						except Exception:
							pass

		try:
			Gimp.displays_flush()
		except Exception:
			pass
	finally:
		image.undo_group_end()


def _default_settings_dict():
	# Keep in sync with PixelPerfectAlignerDialog defaults.
	return {
		"grid_w": 64,
		"grid_h": 64,
		"method": "most_used",
		"similarity_threshold": 16,
		"ignore_transparent": True,
		"denoise_enabled": False,
		"denoise_strength": 35,
		"denoise_mode": "trimmed",
		"fix_transparent_rgb": True,
		"bleed_radius": 4,
		"neighbor_margin": 0.40,
		"offset_x": 0.0,
		"offset_y": 0.0,
		"scale": 1.0,
		"pan_x": 0.0,
		"pan_y": 0.0,
		"preview_cell": 8,
		"fit_preview": True,
		"show_grid": True,
		"grid_alpha": 90,
		"alpha_cutoff": 1,
		"binary_alpha": False,
		"palette_max": 0,
		"palette_mode": "topn",
		"palette_preserve_rare": 40,
		"palette_use_oklab": False,
		"outline_enabled": False,
		"outline_width": 1,
		"outline_alpha": 255,
		"outline_color": [0, 0, 0],
		"output_to_new_image": False,
		"scale_output_enabled": False,
		"output_scale": 1,
		"place_output_on_selection": True,
		"fit_output_to_selection": False,
		"replace_in_active_layer": False,
	}


class _Params(object):
	pass


def _params_from_settings_dict(d):
	base = _default_settings_dict()
	if isinstance(d, dict):
		for k, v in d.items():
			base[k] = v
	obj = _Params()
	# Types + safety clamping
	obj.grid_w = int(base.get("grid_w", 64))
	obj.grid_h = int(base.get("grid_h", 64))
	obj.method = str(base.get("method", "most_used"))
	obj.similarity_threshold = int(base.get("similarity_threshold", 16))
	obj.ignore_transparent = bool(base.get("ignore_transparent", True))
	obj.denoise_enabled = bool(base.get("denoise_enabled", False))
	obj.denoise_strength = int(base.get("denoise_strength", 35))
	obj.denoise_mode = str(base.get("denoise_mode", "trimmed"))
	obj.fix_transparent_rgb = bool(base.get("fix_transparent_rgb", True))
	obj.bleed_radius = int(base.get("bleed_radius", 4))
	obj.neighbor_margin = float(base.get("neighbor_margin", 0.40))
	obj.offset_x = float(base.get("offset_x", 0.0))
	obj.offset_y = float(base.get("offset_y", 0.0))
	obj.scale = float(base.get("scale", 1.0))
	obj.pan_x = float(base.get("pan_x", 0.0))
	obj.pan_y = float(base.get("pan_y", 0.0))
	obj.alpha_cutoff = int(base.get("alpha_cutoff", 1))
	obj.binary_alpha = bool(base.get("binary_alpha", False))
	obj.palette_max = int(base.get("palette_max", 0))
	obj.palette_mode = str(base.get("palette_mode", "topn"))
	obj.palette_preserve_rare = int(base.get("palette_preserve_rare", 40))
	obj.palette_use_oklab = bool(base.get("palette_use_oklab", False))
	obj.outline_enabled = bool(base.get("outline_enabled", False))
	obj.outline_width = int(base.get("outline_width", 1))
	obj.outline_alpha = int(base.get("outline_alpha", 255))
	oc = base.get("outline_color", [0, 0, 0])
	try:
		obj.outline_color = (int(oc[0]), int(oc[1]), int(oc[2]))
	except Exception:
		obj.outline_color = (0, 0, 0)
	obj.output_to_new_image = bool(base.get("output_to_new_image", False))
	obj.scale_output_enabled = bool(base.get("scale_output_enabled", False))
	obj.output_scale = int(base.get("output_scale", 1))
	obj.place_output_on_selection = bool(base.get("place_output_on_selection", True))
	obj.fit_output_to_selection = bool(base.get("fit_output_to_selection", False))
	obj.replace_in_active_layer = bool(base.get("replace_in_active_layer", False))
	return obj


def plugin_run(procedure, run_mode, image, drawables, config, run_data):
	try:
		if drawables is None or len(drawables) < 1:
			return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, None)
		drawable = drawables[0]
		sel = _get_selection_bounds(image)

		if run_mode == Gimp.RunMode.INTERACTIVE:
			GimpUi.init("pixel_perfect_aligner")
			Gegl.init(None)
			dlg = PixelPerfectAlignerDialog(image, drawable, sel)
			resp = dlg.run()
			if resp != Gtk.ResponseType.OK:
				dlg.destroy()
				return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, None)
			dlg._read_params()
			# Persist last-used settings so "Repeat" can run without the dialog.
			try:
				_save_last_settings_dict(dlg._current_settings_dict())
			except Exception:
				pass
			_apply_to_image(image, drawable, sel, dlg)
			dlg.destroy()
			return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

		# "Repeat" (WITH_LAST_VALS) and scripted runs.
		last = _load_last_settings_dict()
		if isinstance(last, dict):
			params = _params_from_settings_dict(last)
			_apply_to_image(image, drawable, sel, params)
			try:
				_save_last_settings_dict(last)
			except Exception:
				pass
			return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

		# No last settings stored yet: fall back to interactive.
		GimpUi.init("pixel_perfect_aligner")
		Gegl.init(None)
		dlg = PixelPerfectAlignerDialog(image, drawable, sel)
		resp = dlg.run()
		if resp != Gtk.ResponseType.OK:
			dlg.destroy()
			return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, None)
		dlg._read_params()
		try:
			_save_last_settings_dict(dlg._current_settings_dict())
		except Exception:
			pass
		_apply_to_image(image, drawable, sel, dlg)
		dlg.destroy()
		return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

	except Exception as e:
		try:
			Gimp.message("Pixel-Perfect Aligner failed: %s" % str(e))
		except Exception:
			pass
		return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, None)


class PixelPerfectAligner(Gimp.PlugIn):
	def do_set_i18n(self, name):
		return False

	def do_query_procedures(self):
		return [PROC_NAME]

	def do_create_procedure(self, name):
		procedure = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, plugin_run, None)
		procedure.set_image_types("RGB*, GRAY*")
		procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
		procedure.set_menu_label("Pixel-Perfect Aligner (AI Fix)...")
		procedure.add_menu_path(MENU_PATH)
		procedure.set_documentation(
			"Align + reconstruct true pixel art from blurry AI 'pixel' images.",
			"Make a rectangular selection around the sprite, then use Auto Align and Apply.",
			name
		)
		procedure.set_attribution("reppiz", "MIT", "2026")
		return procedure


Gimp.main(PixelPerfectAligner.__gtype__, sys.argv)
