import tensorflow as tf

def area(boxlist, scope=None):
    """Computes area of boxes.

    Args:
        boxlist: BoxList holding N boxes (x, y, w, h)
    scope: name scope.

    Returns:
        a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        y, x, h, w = tf.split(
            value=boxlist, num_or_size_splits=4, axis=1)
        return w*h

def intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxlist1: BoxList holding N boxes(x,y,w,h)
        boxlist2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'Intersection'):
        y1, x1, h1, w1 = tf.split(
            value=boxlist1, num_or_size_splits=4, axis=1)
        y2, x2, h2, w2 = tf.split(
            value=boxlist2, num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y1 + h1, tf.transpose(y2 + h2))
        all_pairs_max_ymin = tf.maximum(y1, tf.transpose(y2))
        intersect_heights = tf.maximum(
            0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x1 + w1, tf.transpose(x2 + w2))
        all_pairs_max_xmin = tf.maximum(x1, tf.transpose(x2))
        intersect_widths = tf.maximum(
            0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths

def ioa(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.
    Args:
        boxlist1: BoxList holding N boxes(x,y,w,h)
        boxlist2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'IOU'):
        intersections = intersection(boxlist1, boxlist2)
        areas1 = tf.squeeze(area(boxlist1), [1])
        areas2 = tf.squeeze(area(boxlist2), [1])
        unions = (
            tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections), tf.truediv(intersections, unions))

def scale(boxlist, y_scale, x_scale, scope=None):
    """scale box coordinates in x and y dimensions.
    Args:
        boxlist: BoxList holding N boxes
        y_scale: (float) scalar tensor
        x_scale: (float) scalar tensor
        scope: name scope.
    Returns:
        boxlist: BoxList holding N boxes
    """
    with tf.name_scope(scope, 'Scale'):
        y_scale = tf.cast(y_scale, tf.float32)
        x_scale = tf.cast(x_scale, tf.float32)
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist, num_or_size_splits=4, axis=1)
        y_min = y_scale * y_min
        y_max = y_scale * y_max
        x_min = x_scale * x_min
        x_max = x_scale * x_max
        scaled_boxlist = tf.concat([y_min, x_min, y_max, x_max], 1)
        return scaled_boxlist

def change_coordinate_frame(boxlist, window, scope=None):
    """Change coordinate frame of the boxlist to be relative to window's frame.
    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).
    An example use case is data augmentation: where we are given groundtruth
    boxes (boxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth box to be relative to this new window.
    Args:
        boxlist: A BoxList object holding N boxes.
        window: A rank 1 tensor [4].
        scope: name scope.
    Returns:
        Returns a BoxList object with N boxes.
    """
    with tf.name_scope(scope, 'ChangeCoordinateFrame'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxlist_new = scale((boxlist - [window[0], window[1], window[0], window[1]]),
                            1.0 / win_height, 1.0 / win_width)
        return boxlist_new

def prune_non_overlapping_boxes(
    boxlist1, boxlist2, min_overlap=0.0, scope=None):
    """Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.
    For each box in boxlist1, we want its IOA to be more than minoverlap with
    at least one of the boxes in boxlist2. If it does not, we remove it.
    Args:
        boxlist1: BoxList holding N boxes.
        boxlist2: BoxList holding M boxes.
        min_overlap: Minimum required overlap between boxes, to count them as
                    overlapping.
        scope: name scope.
    Returns:
        new_boxlist1: A pruned boxlist with size [N', 4].
        keep_inds: A tensor with shape [N'] indexing kept bounding boxes in the
        first input BoxList `boxlist1`.
    """
    with tf.name_scope(scope, 'PruneNonOverlappingBoxes'):
        ioa_ = ioa(boxlist2, boxlist1)  # [M, N] tensor
        ioa_ = tf.reduce_max(ioa_, reduction_indices=[0])  # [N] tensor
        keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
        keep_inds = tf.squeeze(tf.where(keep_bool), squeeze_dims=[1])
        new_boxlist1 = tf.gather(boxlist1, keep_inds)
        return new_boxlist1, keep_inds

def prune_completely_outside_window(boxlist, window, scope=None):
    """Prunes bounding boxes that fall completely outside of the given window.
    This function does not clip partially overflowing boxes.
    Args:
        boxlist: a BoxList holding M_in boxes.
        window: a float tensor of shape [4] representing [ymin, xmin, ymax, xmax]
        of the window
        scope: name scope.
    Returns:
        pruned_boxlist: a new BoxList with all bounding boxes partially or fully in
        the window.
        valid_indices: a tensor with shape [M_out] indexing the valid bounding boxes
        in the input tensor.
    """
    with tf.name_scope(scope, 'PruneCompleteleyOutsideWindow'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        coordinate_violations = tf.concat([
            tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
            tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
        ], 1)
        valid_indices = tf.reshape(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        return tf.gather(boxlist, valid_indices), valid_indices

def random_adjust_brightness(image, max_delta=0.2):
    delta = tf.random_uniform([], -max_delta, max_delta)
    image = tf.image.adjust_brightness(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image

def random_adjust_contrast(image,
                           min_delta=0.8,
                           max_delta=1.25):
    factor = tf.random_uniform([], min_delta, max_delta)
    image = tf.image.adjust_contrast(image / 255, factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image

def random_adjust_hue(image,
                      max_delta=0.02):
    delta = tf.random_uniform([], -max_delta, max_delta)
    image = tf.image.adjust_hue(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image

def random_adjust_saturation(image,
                            min_delta=0.8,
                            max_delta=1.25):
    factor = tf.random_uniform([], min_delta, max_delta)
    image = tf.image.adjust_saturation(image / 255, factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image

def random_distort_color(image, color_orderding=0):
    if color_orderding == 0:
        image = random_adjust_brightness(
            image, max_delta=32. / 255.
            )
        image = random_adjust_saturation(
            image, min_delta=0.5, max_delta=1.5
        )
        image = random_adjust_hue(
            image, max_delta=0.2
        )
        image = random_adjust_contrast(
            image, min_delta=0.5, max_delta=1.5
        )
    elif color_orderding == 1:
        image = random_adjust_brightness(
            image, max_delta=32. / 255.
            )
        image = random_adjust_contrast(
            image, min_delta=0.5, max_delta=1.5
        )
        image = random_adjust_saturation(
            image, min_delta=0.5, max_delta=1.5
        )
        image = random_adjust_hue(
            image, max_delta=0.2
        )
    else:
        raise ValueError('color_ordering must be in {0, 1}')
    return image