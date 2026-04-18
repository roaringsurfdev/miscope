import colorsys


def get_frequency_color(frequency: int, num_frequencies: int) -> str:
    """Return a consistent RGB color string for frequency (1-indexed)."""
    hue = (frequency - 1) / num_frequencies
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.5)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
