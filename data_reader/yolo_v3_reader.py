import tensorflow as tf
import numpy as np
import functools
from data_reader import data_augment
from data_reader import util
import cv2
from PIL import Image

keys_to_features = {
    'image_raw':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'image/height':
        tf.FixedLenFeature((), tf.int64, 1),
    'image/width':
        tf.FixedLenFeature((), tf.int64, 1),
    # Object boxes and classes.
    'image/object/bbox/xmin':
        tf.VarLenFeature(tf.float32),
    'image/object/bbox/xmax':
        tf.VarLenFeature(tf.float32),
    'image/object/bbox/ymin':
        tf.VarLenFeature(tf.float32),
    'image/object/bbox/ymax':
        tf.VarLenFeature(tf.float32),
    'image/object/class/label':
        tf.VarLenFeature(tf.int64),
}


class YOLOV3Reader:
    """
    input pipline of the models
    """

    def __init__(self, 
                mode,
                anchors,
                resize_width,
                resize_height,
                num_object,
                num_class,
                grid_width,
                grid_height,
                random_flip,
                jitter_box,
                random_crop,
                distort_color):
        self.mode = mode
        self.anchors = anchors
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.num_object = num_object
        self.num_class = num_class
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.random_flip = random_flip
        self.jitter_box = jitter_box
        self.random_crop = random_crop
        self.distort_color = distort_color

    def image_labels(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                keys_to_features)
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image_height = tf.cast(features['image/height'], tf.float32)
        image_width = tf.cast(features['image/width'], tf.float32)

        x_min = tf.reshape(
            tf.sparse_tensor_to_dense(features['image/object/bbox/xmin']), 
            [-1, 1])
        y_min = tf.reshape(
            tf.sparse_tensor_to_dense(features['image/object/bbox/ymin']),
            [-1, 1])
        x_max = tf.reshape(
            tf.sparse_tensor_to_dense(features['image/object/bbox/xmax']),
            [-1, 1])
        y_max = tf.reshape(
            tf.sparse_tensor_to_dense(features['image/object/bbox/ymax']),
            [-1, 1])
        class_label = tf.cast(
            tf.reshape(
                tf.sparse_tensor_to_dense(
                    features['image/object/class/label']),
                [-1, 1]), tf.float32)


        scale = tf.reduce_min(
            [self.resize_height / image_height, 
            self.resize_width / image_width])
        new_height = tf.cast(image_height * scale, tf.int32)
        new_width = tf.cast(image_width * scale, tf.int32)
        dx = tf.cast((self.resize_width - new_width) / 2, 
        tf.float32) / self.resize_width
        dy = tf.cast((self.resize_height - new_height) / 2, 
            tf.float32) / self.resize_height
        image = tf.reshape(image, (new_height, new_width, 3), name="image_reshape")
        image = tf.image.resize_image_with_crop_or_pad(
            image, self.resize_height, self.resize_width)
        x_scale = tf.cast(new_width / self.resize_width, tf.float32)
        y_scale = tf.cast(new_height / self.resize_height, tf.float32)
        # tf.Print(x_scale,[x_min,x_max,y_min,y_max],message="ori: ")
        x_min= x_min * x_scale + dx
        x_max = x_max * x_scale + dx
        y_min = y_min * y_scale + dy
        y_max = y_max * y_scale + dy

        # to float
        x_min = x_min / tf.cast(new_width, tf.float32)
        x_max = x_max / tf.cast(new_width, tf.float32)
        y_min = y_min / tf.cast(new_height, tf.float32)
        y_max = y_max / tf.cast(new_height, tf.float32)
        # tf.Print(x_scale,[x_min,x_max,y_min,y_max],message="convert: ")
        # tf.Print(x_scale,[x_scale,dx,y_scale,dy],message="param: ")

        if self.mode == "train":
            if self.random_flip:
                image, box_list = tf.cond(
                    tf.random_uniform([], 0, 10, dtype=tf.int32) > 5,
                    lambda: (image, 
                    tf.concat([y_min, x_min, y_max, x_max], axis=1)),
                    lambda: (tf.image.flip_left_right(image),
                    tf.concat(
                        [y_min, 1.0 - x_max,  y_max, 1.0 - x_min], axis=1))
                    )
            else:
                box_list = tf.concat([y_min, x_min, y_max, x_max], axis=1)

            box_list = tf.clip_by_value(box_list, 0.0, 1.0)
            if self.jitter_box:
                def random_jitter_box(box, ratio=0.04):
                    """Randomly jitter box.

                    Args:
                    box: bounding box [4].
                    ratio: max ratio between jittered box and original box,
                    a number between [0, 0.5].
                    """
                    rand_numbers = tf.random_uniform(
                        [4], -ratio, ratio, dtype=tf.float32)
                    box_width = tf.subtract(box[3], box[1])
                    box_height = tf.subtract(box[2], box[0])
                    hw_coefs = tf.stack(
                        [box_height, box_width, box_height, box_width])
                    hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)
                    jittered_box = tf.add(box, hw_rand_coefs)
                    jittered_box = tf.clip_by_value(jittered_box, 0.0, 1.0)
                    return jittered_box

                boxes_shape = tf.shape(box_list)
                distorted_boxes = tf.map_fn(
                    lambda x: random_jitter_box(x, 0.04), 
                    box_list, dtype=tf.float32)
                box_list = tf.cond(
                    tf.random_uniform([], 0, 10, dtype=tf.int32) > 5,
                    lambda: box_list,
                    lambda: tf.reshape(distorted_boxes, boxes_shape),
                    )
            if self.random_crop:
                def random_crop(image, box_list, class_label):
                    image_shape = tf.shape(image)
                    # boxes are [N, 4]. Lets first make them [N, 1, 4].
                    boxes_expanded = tf.expand_dims(
                        tf.clip_by_value(
                            box_list, 
                            clip_value_min=0.0, clip_value_max=1.0), 1)
                    (im_box_begin,
                    im_box_size, 
                    im_box) = tf.image.sample_distorted_bounding_box(
                        image_shape,
                        bounding_boxes=boxes_expanded,
                        min_object_covered=0.3,
                        aspect_ratio_range=[0.75, 1.33],
                        area_range=[0.05, 1],
                        max_attempts=100,
                        use_image_if_no_bounding_boxes=True
                        )
                    new_image = tf.slice(image, im_box_begin, im_box_size)
                    # [1, 4]
                    im_box_rank2 = tf.squeeze(im_box, squeeze_dims=[0])
                    # [4]
                    im_box_rank1 = tf.squeeze(im_box)
                    (box_list, 
                    inside_window_ids) =(
                        data_augment.prune_completely_outside_window(
                            box_list, im_box_rank1))
                    class_label = tf.gather(class_label, inside_window_ids)
                    overlapping_boxlist, keep_ids =(
                        data_augment.prune_non_overlapping_boxes(
                            box_list, im_box_rank2))
                    new_class_label = tf.gather(class_label, keep_ids)
                    new_boxlist =(
                        data_augment.change_coordinate_frame(
                            overlapping_boxlist, im_box_rank1))
                    new_boxlist = tf.clip_by_value(
                        new_boxlist, clip_value_min=0.0, clip_value_max=1.0)
                    new_image = tf.image.resize_images(
                        new_image, [self.resize_height, self.resize_width])
                    return (new_image, 
                    tf.concat([new_boxlist, new_class_label], axis=1))

                image, box_list = tf.cond(
                    tf.random_uniform([], 0, 10, dtype=tf.int32) > 4,
                    lambda: (image, 
                    tf.concat([box_list, class_label], axis=1)),
                    lambda: random_crop(image, box_list, class_label)
                    )
            else:
                box_list = tf.concat([box_list, class_label], axis=1)
            image = tf.cast(image, tf.float32)
            if self.distort_color:
                image = tf.cond(
                    tf.random_uniform([], 0, 10, dtype=tf.int32) > 4,
                    lambda: image,
                    lambda: data_augment.random_distort_color(image)
                )
        else:
            box_list = tf.concat([y_min, x_min, y_max, x_max], axis=1)
            box_list = tf.clip_by_value(box_list, 0.0, 1.0)
            box_list = tf.concat([box_list, class_label], axis=1)
            image = tf.cast(image, tf.float32)
            # tf.Print(box_list, [box_list], message='box_list', summarize=100)

        def boxlist_to_map(box_list, height, width, start_idx, end_idx):
            label_map = np.zeros(
                [height, width, 
                self.num_object * (self.num_class + 5)], dtype=np.float32)
            object_map = np.zeros(
                [height, width, self.num_object], dtype=np.float32)
            for box in box_list:
                ymin, xmin, ymax, xmax, class_id = box
                center_x = xmin + np.abs(xmax - xmin) / 2
                center_y = ymin + np.abs(ymax - ymin) / 2
                y_index = min(int(np.floor(center_y * height)), height - 1)
                x_index = min(int(np.floor(center_x * width)), width - 1)
                class_id = int(class_id)-1
                ious = []
                for i in range(0, 6 * self.num_object, 2):
                    iou = util.overlap(
                        [0.0, 0.0,
                        self.anchors[i] / self.resize_width,
                        self.anchors[i + 1] / self.resize_height],
                        [0.0, 0.0, np.abs(xmax - xmin), np.abs(ymax - ymin)])
                    ious.append(iou)
                best_n = np.argmax(ious)
                if best_n >= start_idx and best_n <= end_idx:
                    best_n = best_n - start_idx
                    object_map[y_index, x_index, best_n] = 1.0
                    label_map[y_index,
                        x_index, best_n * (self.num_class + 5) ] = center_x
                    label_map[y_index,
                        x_index, best_n * (self.num_class + 5) + 1 ] = center_y
                    label_map[y_index,
                        x_index, best_n * (self.num_class + 5) + 2 ] = np.abs(xmax - xmin)
                    label_map[y_index,
                        x_index, best_n * (self.num_class + 5) + 3 ] = np.abs(ymax - ymin)
                    label_map[y_index,
                        x_index, best_n * (self.num_class + 5) + 4 ] = 1.0
                    label_map[y_index,
                        x_index, best_n * (self.num_class + 5) + 5 + class_id] = 1.0
                else:
                    label_map[y_index, x_index, 0] = center_x
                    label_map[y_index, x_index, 1] = center_y
                    label_map[y_index, x_index, 2] = np.abs(xmax - xmin)
                    label_map[y_index, x_index, 3] = np.abs(ymax - ymin)
                    label_map[y_index, x_index, 4] = 1.0
                    label_map[y_index, x_index, 5 + class_id] = 1.0
            return label_map, object_map

        generator_func1 = functools.partial(
            boxlist_to_map, 
            height=self.grid_height,
            width=self.grid_width,
            start_idx=0, end_idx=self.num_object - 1)
        label_map1, object_map1 = tf.py_func(
            generator_func1, [box_list], [tf.float32, tf.float32])
        label_map1 = tf.reshape(
            label_map1, 
            [self.grid_height, 
            self.grid_width, self.num_object * (self.num_class + 5)])
        object_map1 = tf.reshape(
            object_map1, [self.grid_height, self.grid_width, self.num_object])

        generator_func2 = functools.partial(
            boxlist_to_map,
            height=2 * self.grid_height, 
            width=2 * self.grid_width,
            start_idx=self.num_object, end_idx=self.num_object * 2 - 1)
        label_map2, object_map2 = tf.py_func(
            generator_func2, [box_list], [tf.float32, tf.float32])
        label_map2 = tf.reshape(
            label_map2, 
            [2 * self.grid_height, 
            2 * self.grid_width, self.num_object * (self.num_class + 5)])
        object_map2 = tf.reshape(
            object_map2, [2 * self.grid_height, 2 * self.grid_width, self.num_object])

        generator_func3 = functools.partial(
            boxlist_to_map, 
            height=4 * self.grid_height, 
            width=4 * self.grid_width,
            start_idx=self.num_object * 2, end_idx=self.num_object * 3 - 1)
        label_map3, object_map3 = tf.py_func(
            generator_func3, [box_list], [tf.float32,tf.float32])
        label_map3 = tf.reshape(
            label_map3, 
            [4 * self.grid_height, 
            4 * self.grid_width, self.num_object * (self.num_class + 5)])
        object_map3 = tf.reshape(
            object_map3, [4 * self.grid_height, 4 * self.grid_width, self.num_object])

        image = image / 255.0

        #tf.Print(object_map1,[tf.reduce_sum(tf.reduce_sum(object_map1,0),0),tf.reduce_sum(tf.reduce_sum(object_map2,0),0),tf.reduce_sum(tf.reduce_sum(object_map3,0),0)],message="mask")
        #tf.Print(object_map1,[x_min,x_max,y_min,y_max],message="xywh:")
        return (tf.reshape(image, [self.resize_height, self.resize_width, 3]), 
        label_map1, label_map2, label_map3, object_map1, object_map2, object_map3)
