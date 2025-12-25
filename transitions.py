import cv2
import numpy as np


def fade_zoom_in(img1, img2, frames=30):
    """
    Continuous zoom fade transition.
    img1 continues zooming from 1.15 to 1.30 while fading out.
    img2 zooms from 1.0 to 1.15 while fading in.
    Creates seamless continuous zoom throughout entire video.
    """
    h, w, _ = img1.shape
    output = []

    for i in range(frames):
        progress = i / frames

        # img1 continues zooming: 1.15 -> 1.30 (continues from static display)
        scale1 = 1.15 + 0.15 * progress

        # img2 starts zooming: 1.0 -> 1.15 (beginning of its zoom)
        scale2 = 1.0 + 0.15 * progress

        # Apply zoom to both images
        def zoom_crop(img, scale):
            sh, sw = int(h * scale), int(w * scale)
            zoomed = cv2.resize(img, (sw, sh))
            crop_y = (sh - h) // 2
            crop_x = (sw - w) // 2
            return zoomed[crop_y:crop_y+h, crop_x:crop_x+w]

        zoomed_img1 = zoom_crop(img1, scale1)
        zoomed_img2 = zoom_crop(img2, scale2)

        # Fade between the two zooming images
        alpha = progress
        frame = cv2.addWeighted(zoomed_img1, 1 - alpha, zoomed_img2, alpha, 0)
        output.append(frame)

    return output


TRANSITIONS = {
    "Fade Zoom In": fade_zoom_in
}