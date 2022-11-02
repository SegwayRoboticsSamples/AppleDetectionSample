import tensorflow as tf


def area(boxlist, scope=None):
    """Computes area of boxes.

    Args:
        boxlist: BoxList holding N boxes (x, y, w, h)
    scope: name scope.

    Returns:
        a tensor with shape [N] representing box areas.
    """
    x, y, w, h = tf.split(
        value=boxlist, num_or_size_splits=4, axis=1)
    return w*h
    
def matched_intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxlist1: BoxList holding N boxes (x, y, w, h)
        boxlist2: BoxList holding N boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, 1] representing pairwise intersections
    """
    x1, y1, w1, h1 = tf.split(
        value=boxlist1, num_or_size_splits=4, axis=1)
    x2, y2, w2, h2 = tf.split(
        value=boxlist2, num_or_size_splits=4, axis=1)
    pairs_min_ymax = tf.minimum(y1 + h1, y2 + h2)
    pairs_max_ymin = tf.maximum(y1, y2)
    intersect_heights = tf.maximum(0.0, pairs_min_ymax - pairs_max_ymin)
    pairs_min_xmax = tf.minimum(x1 + w1, x2 + w2)
    pairs_max_xmin = tf.maximum(x1, x2)
    intersect_widths = tf.maximum(0.0, pairs_min_xmax - pairs_max_xmin)
    return intersect_heights * intersect_widths


def matched_iou(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.
    Args:
        boxlist1: BoxList holding N boxes (x, y, w, h)
        boxlist2: BoxList holding N boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, 1] representing pairwise iou scores.
    """
    center_x, center_y, w, h = tf.split(boxlist1, num_or_size_splits=4, axis=1 )
    center_x2, center_y2, w2, h2 = tf.split(boxlist2, num_or_size_splits=4, axis=1 )
    boxlist1 = tf.concat([center_x - w / 2, center_y - h / 2, w, h], axis=1)
    boxlist2 = tf.concat([center_x2 - w2 / 2, center_y2 - h2 /2, w2, h2], axis=1)
    intersections = matched_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = areas1 + areas2 - intersections
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

def intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxlist1: BoxList holding N boxes(x,y,w,h)
        boxlist2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise intersections
    """
    x1, y1, w1, h1 = tf.split(
        value=boxlist1, num_or_size_splits=4, axis=1)
    x2, y2, w2, h2 = tf.split(
        value=boxlist2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y1 + h1, tf.transpose(y2 + h2))
    all_pairs_max_ymin = tf.maximum(y1, tf.transpose(y2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x1 + w1, tf.transpose(x2 + w2))
    all_pairs_max_xmin = tf.maximum(x1, tf.transpose(x2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def iou(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.
    Args:
        boxlist1: BoxList holding N boxes(x,y,w,h)
        boxlist2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
    """
    center_x, center_y, w, h = tf.split(boxlist1, num_or_size_splits=4, axis=1 )
    center_x2, center_y2, w2, h2 = tf.split(boxlist2, num_or_size_splits=4, axis=1 )
    boxlist1 = tf.concat([center_x - w / 2, center_y - h / 2, w, h], axis=1)
    boxlist2 = tf.concat([center_x2 - w2 / 2, center_y2 - h2 / 2, w2, h2], axis=1)
    intersections = intersection(boxlist1, boxlist2)
    areas1 = tf.squeeze(area(boxlist1), [1])
    areas2 = tf.squeeze(area(boxlist2), [1])
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))