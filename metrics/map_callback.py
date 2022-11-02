from tensorflow import keras
from metrics import util
from boxes_parser import cornernet_parser
from metrics import average_precision
import tensorflow as tf

import numpy as np


def make_image(tag, image):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width, channel = image.shape
    from PIL import Image
    import io
    if channel <= 3:
        pil_img = Image.fromarray(image)
        output = io.BytesIO()
        pil_img.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return [tf.Summary.Value(tag=tag, image=tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string))]
    else:
        summary_images = []
        for i in range(channel):
            pil_image = Image.fromarray(image[:,:,i])
            output = io.BytesIO()
            pil_image.save(output, format='BMP')
            image_string = output.getvalue()
            output.close()
            summary_images.append(tf.Summary.Value(tag=tag+str(i), image=tf.Summary.Image(height=height,
                width=width, colorspace=1, encoded_image_string=image_string)))
        return summary_images

class MeanAveragePrecision(keras.callbacks.Callback):
    def __init__(self, sess, validation_data, num_classes, test_step, tensorboard, threshold=0.5):
        self.validation_data = validation_data
        self.maps = []
        self.num_classes = num_classes
        self.test_step = test_step
        self.tensorboard = tensorboard
        self.sess = sess

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x_val, y_val, tl_val_heatmap, br_val_heatmap = self.validation_data.make_one_shot_iterator().get_next()
        
        all_gt_dict = {}
        all_predict_boxes = {}
        for c in range(self.num_classes):
            all_gt_dict[c] = {}
            all_predict_boxes[c] = []
        stage_idx = 0
        for i in range(self.test_step):
            image, gt_boxes, gt_tl_heatmap, gt_br_heatmap = self.sess.run([x_val, y_val, tl_val_heatmap, br_val_heatmap])
            gt_data = util.visiualize_annotations(image[0], gt_boxes[0])
            #should use predict model here
            _, _, _, heatmaps, tags, offsets, _ = self.model.predict_on_batch(image)
            heatmaps = np.transpose(heatmaps, [0, 2, 3, 1])
            tags = np.transpose(tags, [0, 2, 3, 1])
            offsets = np.transpose(offsets, [0, 2, 3, 1])
            heatmaps = 1.0 / (1.0 + np.exp(-heatmaps))
            for b in range(heatmaps.shape[0]):
                pred_boxes = cornernet_parser.boxes_from_prediction(heatmaps[b], tags[b], offsets[b], stage_idx)
                test_data = util.visiualize_annotations(image[0], pred_boxes)
                for box in pred_boxes:
                    class_id = int(box[4])
                    all_predict_boxes[class_id].append(box)
                    
                for box in gt_boxes[b]:
                    formated_box = [box[0], box[1], box[2], box[3], box[4]]
                    if formated_box[0] < 0:
                        continue
                    class_id = int(formated_box[4]) - 1
                    if stage_idx in all_gt_dict[class_id]:
                        all_gt_dict[class_id][stage_idx].append(formated_box)
                    else:
                        all_gt_dict[class_id][stage_idx] = [formated_box]
                stage_idx = stage_idx +1
            '''for b in range(pred_boxes.shape[0]):
                formated_boxes = []
                
                for box in pred_boxes[b]:
                    if box[4] > 0.1:
                        formated_boxes.append([box[0], box[1], box[2], box[3], box[7], box[4], stage_idx])
                
                boxes = np.asarray(formated_boxes)
                if boxes.shape[0] > 0:
                    nms_box =util.nms(boxes, self.num_classes)
                    for box in nms_box:
                        class_id = int(box[4])
                        all_predict_boxes[class_id].append(box)
                
                for box in gt_boxes[b]:
                    formated_box = [box[1], box[0], box[3], box[2], box[4]]
                    if formated_box[0] < 0:
                        continue
                    class_id = int(formated_box[4]) - 1
                    if stage_idx in all_gt_dict[class_id]:
                        all_gt_dict[class_id][stage_idx].append(formated_box)
                    else:
                        all_gt_dict[class_id][stage_idx] = [formated_box]
                stage_idx = stage_idx +1'''
        
        mAP = 0.0
        for c in range(self.num_classes):
            ap = average_precision.calculate_averge_precision(
                all_predict_boxes[c], all_gt_dict[c])
            print(ap)
            mAP += ap
        mAP = mAP / self.num_classes
        print("map in test data: {:.3f}".format(mAP))
        logs['mAP'] = mAP
        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary = tf.Summary(value=make_image('gt', gt_data) + make_image('test', test_data))
            #summary = tf.Summary(value=(make_image('heatmap', (heatmaps[0] * 255).astype(np.uint8)) + make_image('gt', gt_data)
            #    + make_image('gt_tl', (gt_tl_heatmap[0]*255).astype(np.uint8)) + make_image('gt_br', (gt_br_heatmap[0]*255).astype(np.uint8))))
            summary_value = summary.value.add()
            summary_value.simple_value = mAP
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

                
