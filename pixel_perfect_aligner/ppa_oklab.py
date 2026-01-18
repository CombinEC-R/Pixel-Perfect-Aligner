# file: ppa_oklab.py
# Perceptual color helpers (Oklab) for palette clustering and mapping.
# Pure python, designed for GIMP Python plug-ins.


def _clamp01(x: float) -> float:
	if x < 0.0:
		return 0.0
	if x > 1.0:
		return 1.0
	return x


def srgb_to_linear_u8(u8: int) -> float:
	x = float(int(u8)) / 255.0
	if x <= 0.04045:
		return x / 12.92
	return ((x + 0.055) / 1.055) ** 2.4


def linear_to_srgb_u8(x: float) -> int:
	x = _clamp01(float(x))
	if x <= 0.0031308:
		y = 12.92 * x
	else:
		y = 1.055 * (x ** (1.0 / 2.4)) - 0.055
	u = int(round(y * 255.0))
	if u < 0:
		return 0
	if u > 255:
		return 255
	return u



def rgb_to_oklab_u8(r: int, g: int, b: int):
	rl = srgb_to_linear_u8(r)
	gl = srgb_to_linear_u8(g)
	bl = srgb_to_linear_u8(b)

	# linear sRGB to LMS
	l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl
	m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl
	s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl

	l_ = l ** (1.0 / 3.0)
	m_ = m ** (1.0 / 3.0)
	s_ = s ** (1.0 / 3.0)

	L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
	A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
	B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
	return (L, A, B)



def oklab_to_rgb_u8(L: float, A: float, B: float):
	l_ = L + 0.3963377774 * A + 0.2158037573 * B
	m_ = L - 0.1055613458 * A - 0.0638541728 * B
	s_ = L - 0.0894841775 * A - 1.2914855480 * B

	l = l_ * l_ * l_
	m = m_ * m_ * m_
	s = s_ * s_ * s_

	rl = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
	gl = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
	bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

	return (
		linear_to_srgb_u8(rl),
		linear_to_srgb_u8(gl),
		linear_to_srgb_u8(bl),
	)


def dist_sq_oklab(p1, p2) -> float:
	dL = float(p1[0]) - float(p2[0])
	dA = float(p1[1]) - float(p2[1])
	dB = float(p1[2]) - float(p2[2])
	return (dL * dL) + (dA * dA) + (dB * dB)
