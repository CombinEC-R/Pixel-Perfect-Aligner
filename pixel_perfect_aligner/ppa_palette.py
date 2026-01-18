# file: ppa_palette.py
# Palette building + mapping helpers.
# Adds palette modes (Top-N / K-Means clusters) and perceptual mapping (Oklab).

import random

import ppa_oklab


def _clamp_int(v: int, lo: int, hi: int) -> int:
	if v < lo:
		return lo
	if v > hi:
		return hi
	return v


def _color_dist_sq_rgb(c1, c2) -> int:
	dr = int(c1[0]) - int(c2[0])
	dg = int(c1[1]) - int(c2[1])
	db = int(c1[2]) - int(c2[2])
	return (dr * dr) + (dg * dg) + (db * db)



def _quant_key_5bit(rgb):
	# 5 bits per channel -> 15-bit key
	r = int(rgb[0]) >> 3
	g = int(rgb[1]) >> 3
	b = int(rgb[2]) >> 3
	return (r << 10) | (g << 5) | b


def _key_to_rgb_5bit(key: int):
	r5 = (key >> 10) & 31
	g5 = (key >> 5) & 31
	b5 = key & 31
	r8 = (r5 << 3) | (r5 >> 2)
	g8 = (g5 << 3) | (g5 >> 2)
	b8 = (b5 << 3) | (b5 >> 2)
	return (int(r8), int(g8), int(b8))


def build_palette_topn(points_rgb, k: int, preserve_rare: int):
	k = _clamp_int(int(k), 2, 256)
	hist = {}
	for rgb in points_rgb:
		key = _quant_key_5bit(rgb)
		hist[key] = hist.get(key, 0) + 1
	if len(hist) <= k:
		return [_key_to_rgb_5bit(key) for key in hist.keys()][:k]

	# Candidates: take more than k and pick a diverse subset if requested.
	items = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)
	cand_n = min(len(items), max(k * 8, k))
	cands = [(_key_to_rgb_5bit(items[i][0]), items[i][1]) for i in range(cand_n)]

	# No diversity bias
	strength = int(preserve_rare)
	if strength <= 0:
		return [c[0] for c in cands[:k]]

	# Diversity selection: frequency + distance bonus
	selected = []
	while len(selected) < k and len(cands) > 0:
		best = None
		best_score = -1e30
		for (rgb, cnt) in cands:
			base = float(cnt)
			if len(selected) == 0:
				score = base
			else:
				min_d = 1e30
				for s in selected:
					d = float(_color_dist_sq_rgb(rgb, s))
					if d < min_d:
						min_d = d
				score = base + (float(strength) / 100.0) * min_d
			if score > best_score:
				best_score = score
				best = (rgb, cnt)
		selected.append(best[0])
		cands.remove(best)
	return selected




def build_palette_topn(points_rgb, k: int, preserve_rare: int):
	# Histogram in 5-bit bins, then pick diverse representatives.
	if k <= 0:
		return []
	k = _clamp_int(int(k), 2, 256)

	hist = {}
	for rgb in points_rgb:
		key = _quant_key_5bit(rgb)
		hist[key] = hist.get(key, 0) + 1

	if len(hist) <= k:
		# Expand bins to 8-bit colors
		return [_key_to_rgb_5bit(key) for key in hist.keys()]

	items = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)
	cand_count = max(k * 10, k)
	candidates = [(_key_to_rgb_5bit(key), cnt) for key, cnt in items[:cand_count]]

	# If preserve_rare <= 0: plain Top-N.
	pr = _clamp_int(int(preserve_rare), 0, 100)
	if pr <= 0:
		return [c[0] for c in candidates[:k]]

	# Diversity-aware selection:
	# score = frequency + (preserve_rare * diversity)
	# diversity = min distance to already selected colors
	selected = []
	while len(selected) < k and len(candidates) > 0:
		best_idx = 0
		best_score = -1.0
		for i, (col, cnt) in enumerate(candidates):
			base = float(cnt)
			if len(selected) == 0:
				score = base
			else:
				min_d = 1e18
				for s in selected:
					d = _color_dist_sq_rgb(col, s)
					if d < min_d:
						min_d = d
				div = float(min_d)
				score = base + (float(pr) / 100.0) * div
			if score > best_score:
				best_score = score
				best_idx = i
		selected.append(candidates[best_idx][0])
		del candidates[best_idx]

	return selected


	top = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)
	# Candidate pool: allow more than k to let diversity keep small but distinct colors.
	cand_n = min(len(top), max(k * 10, k))
	candidates = []
	for i in range(cand_n):
		key = top[i][0]
		candidates.append(_key_to_rgb_5bit(key))

	if preserve_rare <= 0:
		return candidates[:k]

	strength = _clamp_int(int(preserve_rare), 0, 100)
	w_div = float(strength) / 100.0

	selected = []
	# Always start with the most frequent bin
	selected.append(candidates[0])
	remaining = candidates[1:]

	while len(selected) < k and len(remaining) > 0:
		best = remaining[0]
		best_score = -1e30
		for c in remaining:
			# Base: favor common colors (approx by inverse distance to first, but we lack counts per candidate)
			# We instead add a small constant and rely mostly on diversity.
			base = 1.0
			# Diversity: maximin distance to existing selected colors
			min_d = 1e30
			for s in selected:
				d = float(_color_dist_sq_rgb(c, s))
				if d < min_d:
					min_d = d
			score = base + w_div * min_d
			if score > best_score:
				best_score = score
				best = c
		selected.append(best)
		try:
			remaining.remove(best)
		except Exception:
			remaining = [x for x in remaining if x != best]

	return selected


	# Greedy diverse selection: base frequency + distance bonus.
	selected = []
	while len(selected) < k and len(candidates) > 0:
		best = candidates[0]
		best_score = -1.0
		for c in candidates:
			base = 1.0
			base = float(hist.get(_quant_key_5bit(c), 1))
			# Distance to nearest selected color
			div = 0.0
			if len(selected) > 0:
				md = 1e18
				for s in selected:
					d = _color_dist_sq_rgb(c, s)
					if d < md:
						md = d
				div = md
			# Weight: more preserve_rare -> more diversity influence.
			score = base + (float(strength) / 100.0) * div
			if score > best_score:
				best_score = score
				best = c
		selected.append(best)
		candidates.remove(best)
	return selected





def _subsample(points, max_points: int):
	if max_points <= 0 or len(points) <= max_points:
		return points
	step = max(1, len(points) // max_points)
	return points[::step]


def build_palette_kmeans_oklab(points_rgb, k: int, preserve_rare: int, max_points: int = 20000, iters: int = 12):
	if k <= 0:
		return []
	k = _clamp_int(int(k), 2, 256)
	if len(points_rgb) == 0:
		return []

	pts_rgb = _subsample(points_rgb, max_points)
	pts = [ppa_oklab.rgb_to_oklab_u8(p[0], p[1], p[2]) for p in pts_rgb]

	rng = random.Random(1337)

	# K-Means++ init (perceptual)
	centers = [pts[rng.randrange(0, len(pts))]]
	if preserve_rare < 0:
		preserve_rare = 0
	if preserve_rare > 100:
		preserve_rare = 100
	boost = 1.0 + (float(preserve_rare) / 100.0) * 2.0

	for _ in range(1, k):
		weights = []
		sum_w = 0.0
		for p in pts:
			md = 1e18
			for c in centers:
				d = ppa_oklab.dist_sq_oklab(p, c)
				if d < md:
					md = d
			w = md ** boost
			weights.append(w)
			sum_w += w
		if sum_w <= 0.0:
			centers.append(pts[rng.randrange(0, len(pts))])
			continue
		r = rng.random() * sum_w
		acc = 0.0
		chosen = pts[-1]
		for i in range(len(pts)):
			acc += weights[i]
			if acc >= r:
				chosen = pts[i]
				break
		centers.append(chosen)

	# Lloyd iterations
	for _ in range(int(iters)):
		buckets = [[0.0, 0.0, 0.0, 0] for _ in range(k)]
		for p in pts:
			best_i = 0
			best_d = 1e18
			for i, c in enumerate(centers):
				d = ppa_oklab.dist_sq_oklab(p, c)
				if d < best_d:
					best_d = d
					best_i = i
			b = buckets[best_i]
			b[0] += p[0]
			b[1] += p[1]
			b[2] += p[2]
			b[3] += 1

		moved = 0.0
		for i in range(k):
			if buckets[i][3] > 0:
				n = float(buckets[i][3])
				nc = (buckets[i][0] / n, buckets[i][1] / n, buckets[i][2] / n)
				moved += ppa_oklab.dist_sq_oklab(nc, centers[i])
				centers[i] = nc
		if moved < 1e-12:
			break

	# Convert centers back to RGB
	palette = [ppa_oklab.oklab_to_rgb_u8(c[0], c[1], c[2]) for c in centers]
	return palette



def collect_points_from_rgba(rgba_bytes, alpha_min: int = 1):
	mv = memoryview(rgba_bytes)
	pts = []
	for i in range(0, len(mv), 4):
		a = int(mv[i + 3])
		if a < int(alpha_min):
			continue
		r = int(mv[i])
		g = int(mv[i + 1])
		b = int(mv[i + 2])
		pts.append((r, g, b))
	return pts


def map_rgba_to_palette(rgba_bytes, palette_rgb, use_oklab: bool):
	if palette_rgb is None or len(palette_rgb) == 0:
		return rgba_bytes
	out = bytearray(rgba_bytes)
	mv = memoryview(out)

	if use_oklab:
		pal_lab = [ppa_oklab.rgb_to_oklab_u8(c[0], c[1], c[2]) for c in palette_rgb]
		for i in range(0, len(mv), 4):
			a = int(mv[i + 3])
			if a <= 0:
				continue
			p = ppa_oklab.rgb_to_oklab_u8(int(mv[i]), int(mv[i + 1]), int(mv[i + 2]))
			best_j = 0
			best_d = ppa_oklab.dist_sq_oklab(p, pal_lab[0])
			for j in range(1, len(pal_lab)):
				d = ppa_oklab.dist_sq_oklab(p, pal_lab[j])
				if d < best_d:
					best_d = d
					best_j = j
			c = palette_rgb[best_j]
			mv[i] = int(c[0])
			mv[i + 1] = int(c[1])
			mv[i + 2] = int(c[2])
		return bytes(out)

	# RGB nearest
	for i in range(0, len(mv), 4):
		a = int(mv[i + 3])
		if a <= 0:
			continue
		r = int(mv[i])
		g = int(mv[i + 1])
		b = int(mv[i + 2])
		best = palette_rgb[0]
		best_d = _color_dist_sq_rgb((r, g, b), best)
		for c in palette_rgb[1:]:
			d = _color_dist_sq_rgb((r, g, b), c)
			if d < best_d:
				best_d = d
				best = c
		mv[i] = int(best[0])
		mv[i + 1] = int(best[1])
		mv[i + 2] = int(best[2])
	return bytes(out)


def apply_palette_quantization(rgba_bytes, max_colors: int, mode: str, preserve_rare: int, use_oklab: bool):
	# Expects RGBA bytes already alpha-cut (so a==0 is transparent).
	if int(max_colors) <= 0:
		return rgba_bytes
	k = _clamp_int(int(max_colors), 2, 256)
	m = str(mode).strip().lower()
	pts = collect_points_from_rgba(rgba_bytes, 1)
	if len(pts) == 0:
		return rgba_bytes

	palette = None
	if m in ["kmeans", "clusters", "cluster"]:
		palette = build_palette_kmeans_oklab(pts, k, int(preserve_rare))
		return map_rgba_to_palette(rgba_bytes, palette, True)
	if m in ["median", "mediancut", "median_cut"]:
		# Simple fallback: treat as Top-N for now (keeps code small and predictable)
		palette = build_palette_topn(pts, k, int(preserve_rare))
		return map_rgba_to_palette(rgba_bytes, palette, bool(use_oklab))

	# Default: topn
	palette = build_palette_topn(pts, k, int(preserve_rare))
	return map_rgba_to_palette(rgba_bytes, palette, bool(use_oklab))


