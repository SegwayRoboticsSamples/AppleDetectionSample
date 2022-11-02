import tensorflow as tf
import numpy as np
from losses import util

EPSILON = 1e-8


class YOLOV3Loss():
    def __init__(self, grid_height, grid_width, start_end, total_anchors, 
    num_classes, num_object, ignore_threshold=0.5):
        self.anchors = []
        for i in range(2 * start_end[0], 2 * (start_end[1] + 1), 2):
            self.anchors.append(total_anchors[i] * grid_width)
            self.anchors.append(total_anchors[i + 1] * grid_height)
 
        self.total_anchors = total_anchors
        self.anchor_start_end = start_end
        self.num_classes = num_classes
        self.num_object = num_object
        self.ignore_threshold = ignore_threshold
        self.truth_threshold = 1.0

    def compute_loss(self, target_tensor, obj_tensor, prediction_tensor, 
    batch_size):
        (_, grid_height, grid_width, _
        ) = prediction_tensor.get_shape().as_list()
        grid2d_i, grid2d_j = np.meshgrid(
            np.linspace(0, grid_width - 1, grid_width, dtype=np.float32),
            np.linspace(0, grid_height - 1, grid_height, dtype=np.float32),
        )
        grid_i = np.reshape(np.tile(np.reshape(grid2d_i, [-1, 1]), 
        [batch_size, self.num_object]), [-1, 1])
        grid_j = np.reshape(np.tile(np.reshape(grid2d_j, [-1, 1]), 
        [batch_size, self.num_object]), [-1, 1])
    
        anchor = tf.reshape(tf.zeros([batch_size * grid_height * grid_width, 
            2 * self.num_object]) + self.anchors, [-1, 2])
        anchor_w, anchor_h = tf.split(anchor, num_or_size_splits=2, axis=1)
    
        coordinate_indice = []
        class_indice = []
        confidence_indice = []
        for i in range(self.num_object):
            coordinate_indice.append(i * (self.num_classes + 4 + 1)) #x
            coordinate_indice.append(i * (self.num_classes + 4 + 1) + 1) #y
            coordinate_indice.append(i * (self.num_classes + 4 + 1) + 2) #w
            coordinate_indice.append(i * (self.num_classes + 4 + 1) + 3) #h
            confidence_indice.append(i * (self.num_classes + 4 + 1) + 4) #objectness score
            for j in range(self.num_classes):
                class_indice.append(i*(self.num_classes + 4 + 1) + 5 + j)

        x, y, w, h = tf.split(
            tf.reshape(tf.gather(
                prediction_tensor, coordinate_indice, axis=-1), [-1, 4]),
            num_or_size_splits=4,
            axis=1
            )
        predict_coord = tf.concat(
            [(tf.sigmoid(x) + grid_i) / grid_width, 
            (tf.sigmoid(y) + grid_j) / grid_height, 
            tf.exp(w) * anchor_w / grid_width, 
            tf.exp(h) * anchor_h / grid_height],
            axis=1
            )
        
        predict_object = tf.reshape(tf.gather(prediction_tensor, 
        confidence_indice, axis=-1), [-1, 1])
        predict_class = tf.reshape(tf.gather(prediction_tensor, 
        class_indice, axis=-1), [-1, self.num_classes])

        truth_coord = tf.reshape(tf.gather(target_tensor, 
        coordinate_indice, axis=-1), [-1, 4])
        x2, y2, w2, h2 = tf.split(truth_coord, num_or_size_splits=4, axis=1)
        obj_map = tf.reshape(obj_tensor, [-1, 1])
        truth_object = tf.reshape(tf.gather(target_tensor, 
        confidence_indice, axis=-1), [-1, 1])
        truth_class = tf.reshape(tf.gather(target_tensor, 
        class_indice, axis=-1), [-1, self.num_classes])
        gt_bbox_indice = tf.boolean_mask(tf.range(0, tf.shape(truth_coord)[0]), 
        tf.squeeze(truth_object > 0.05, axis=1))
        gt_bbox_list = tf.gather(truth_coord, gt_bbox_indice, axis=0)
        gt_bbox_mask_indice = tf.boolean_mask(
            tf.range(0, tf.shape(truth_coord)[0]), 
            tf.squeeze(obj_map > 0.05, axis=1))
        gt_bbox_mask_list = tf.gather(truth_coord, gt_bbox_mask_indice, axis=0)
        ious = util.iou(predict_coord, gt_bbox_list)
        ious3 = util.matched_iou(gt_bbox_mask_list, 
        tf.gather(predict_coord, gt_bbox_mask_indice, axis=0))
        
        predict_class_map = tf.nn.sigmoid(predict_class)
        class_loss =obj_map*tf.reshape(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=truth_class, 
            logits=predict_class), axis=1), [-1, 1]) #* 5
        best_class = tf.reshape(tf.reduce_max(truth_class * predict_class_map, 
        axis=1), [-1, 1]) * obj_map
        class_score = tf.reduce_sum(best_class)
        x_diff = obj_map * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.clip_by_value(x2 * grid_width - grid_i, 0.0, 1.0), logits=x)
        y_diff = obj_map * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.clip_by_value(y2 * grid_height - grid_j, 0.0, 1.0), logits=y)
        w_diff = obj_map * (tf.log(w2 * grid_width / anchor_w + EPSILON) - w)
        h_diff = obj_map * (tf.log(h2 * grid_height / anchor_h + EPSILON) - h)
        xy_loss = tf.reduce_sum(((2.0 - w2 * h2) * tf.concat([x_diff, y_diff], 
        axis=1)), axis=1)
        wh_loss = tf.reduce_sum(tf.square((2.0 - w2 * h2) * tf.concat(
            [w_diff, h_diff], axis=1)), axis=1)
        
        sigmoid_object = tf.sigmoid(predict_object)
        ignore_mask = tf.reshape(tf.to_float(tf.reduce_max(ious, axis=1) > 
        self.ignore_threshold), [-1, 1])
        
        '''obj_loss = ((obj_map - 1.0) * (1.0 - ignore_mask) * 
            tf.pow(sigmoid_object, 2) * tf.log(
                tf.clip_by_value(1.0 - sigmoid_object, 1e-8, 1.0)) / 5 +
            obj_map * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_map, 
            logits=predict_object) * 20)
        pos_obj_loss=obj_map * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_map,
            logits=predict_object) * 20
        neg_obj_loss = obj_loss - pos_obj_loss'''

        neg_obj_loss = (obj_map - 1.0) * (1.0 - ignore_mask) * tf.pow(sigmoid_object, 2) * tf.log(
                tf.clip_by_value(1.0 - sigmoid_object, 1e-8, 1.0))
        #neg_obj_loss = tf.nn.top_k(tf.reshape(neg_obj_loss,[-1]), tf.reduce_max([tf.cast(tf.reduce_sum(obj_map), tf.int32),3]) * 3).values
        pos_obj_loss = obj_map * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_map,
            logits=predict_object)
        iou_score = tf.reduce_sum(ious3)
        recall_score = tf.reduce_sum(tf.to_float(ious3 > 0.5))
        recall75_score = tf.reduce_sum(tf.to_float(ious3 > 0.75))
        object_score = tf.reduce_sum(obj_map * tf.sigmoid(predict_object))
        avg_noobject = tf.reduce_mean(tf.sigmoid(predict_object))

        count = tf.reduce_sum(obj_map)
        total_loss = (tf.reduce_sum(xy_loss + wh_loss) + 
            tf.reduce_sum(class_loss) + tf.reduce_sum(neg_obj_loss) + tf.reduce_sum(pos_obj_loss)) / batch_size
        total_loss = tf.Print(total_loss, 
        #[class_score / count, iou_score / count, recall_score / count, 
        #recall75_score / count, object_score / count, avg_noobject], 
        [tf.reduce_sum(xy_loss), tf.reduce_sum(wh_loss), tf.reduce_sum(class_loss), tf.reduce_sum(neg_obj_loss), tf.reduce_sum(pos_obj_loss)],message='loss: ')
        return total_loss
