import numpy as np
from PIL import Image

def write_debug_image(image_data):
    image = Image.fromarray((image_data*255).astype(np.uint8))
    image.save('debug.jpg')

def interset(box1, box2):
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box1[3])
    return [left, top, right, bottom]

def overlap(bbx1, bbx2):
    left = max(-bbx1[2] / 2.0, -bbx2[2] / 2.0)
    top = max(-bbx1[3] / 2.0, -bbx2[3] / 2.0)
    right = min(bbx1[2] / 2.0, bbx2[2] / 2.0)
    bottom = min(bbx1[3] / 2.0, bbx2[3] / 2.0)
    if right - left < 0:
        return 0
    if bottom - top < 0:
        return 0
    intersect_area = (right - left) * (bottom - top)
    union_area = (bbx1[2] - bbx1[0]) * (bbx1[3]-bbx1[1]) + (
        bbx2[2] - bbx2[0]) * (bbx2[3] - bbx2[1]) - intersect_area

    return intersect_area / union_area

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0 : 2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)
