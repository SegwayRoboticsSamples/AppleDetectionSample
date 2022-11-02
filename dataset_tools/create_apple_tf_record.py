
r"""Convert T60 detection dataset (pedestrian box) to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_kitti_tf_record.py \
        --data_dir=/home/user/kitti \
        --output_path=/home/user/kitti.record
        ...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import io
import os
import cv2
import json
import math
import random
import hashlib
import numpy as np
import PIL.Image as pil
import tensorflow as tf

from PIL import Image
from skimage import transform

import email_sender
import dataset_util
import fisheye_img_util
import get_data_from_database


# custom_database_root_dir : '/dta/zhanghanjun/datasets/remote'

classes_name=["Pedestrian", "noPedestrian"]
# classes_label_map={"Object":0, "noObject":1}
classes_label_map={"Apple":0, "noApple":1}

tf.app.flags.DEFINE_string('data_dir', '/dta/zhanghanjun/datasets/remote', 'Location of root directory for the images.')
tf.app.flags.DEFINE_integer('from_database', '1', 'Data from database.')
tf.app.flags.DEFINE_string('data_family', 'testing', 'Class of data.')
# tf.app.flags.DEFINE_string('start_time', '2019-10-16 00:23:29', 'start time of database data.')
# tf.app.flags.DEFINE_string('end_time', '2021-10-17 00:23:29', 'end time of database data.')
tf.app.flags.DEFINE_string('start_time', '2021-01-12 16:45:29', 'start time of database data.')
tf.app.flags.DEFINE_string('end_time', '2021-04-15 12:50:29', 'end time of database data.')
tf.app.flags.DEFINE_string('output_path', '/dta/llx/data/mobileV1-YoloV3/testdata/20210112_20210415', 'Path to which TFRecord files'
                           'will be written. The TFRecord with the training set'
                           'will be located at: <output_path>_train.tfrecord.'
                           'And the TFRecord with the validation set will be'
                           'located at: <output_path>_val.tfrecord')
tf.app.flags.DEFINE_string('visual_dir', './visualization', 'Path to visualization')

tf.app.flags.DEFINE_integer('tfrecord_width', '512', 'image width to tfrecord')
tf.app.flags.DEFINE_integer('tfrecord_height', '512', 'image height to tfrecord')
FLAGS = tf.app.flags.FLAGS

'''
    Modification 20210419:
        1, resolution (608, 608) -> (512. 512)
        2, rotation
        3, crop the part above the scooter handle.
        4, mask

'''

'''
Mask Param
'''
front_mask_border_pt_idx = [74, 223, 77535, 77563, 184571, 184543, 261855, 261706] #small fov,modify at 2020-3-13
front_mask_border_pts_large_fov = [103, 254, 80126, 80142, 182030, 182014, 261886, 261735] #large fov, modify at 2020-4-29
front_mask_border_pts_large_fov_1005 = [147, 261, 76037, 76075, 186155, 186117, 261893, 261779] #large fov 1005, modify at 2020-7-17
front_mask_border_pts_large_fov_1007 = [147, 261, 76037, 76075, 186155, 186117, 261893, 261779] #large fov 1005, modify at 2020-7-17
front_mask_border_pts_large_fov_2000 = [147, 261, 76037, 76075, 186155, 186117, 261893, 261779] #large fov 1005, modify at 2020-7-17

def visualization(img, xmin, ymin, xmax, ymax, img_idx, img_path, visual_dir):
    img_res = img.copy()
    for bbid in range(len(xmin)):
        img_res = cv2.rectangle(img_res, (xmin[bbid], ymin[bbid]), (xmax[bbid], ymax[bbid]), (0,0,255))
    cv2.imwrite(os.path.join(visual_dir, str(img_idx) + '.jpg'), img_res)

def create_mask(front_mask_border_pt_idx, is_color, ori_img_width, ori_img_height):
    '''
        Front_mask_border_pts is calculated with resolution (512 ,512), need to transform.
    '''
    front_mask_border_pt = []
    for idx_t in range(len(front_mask_border_pt_idx)):
        temp_pt = []
        temp_pt.append(int(float(front_mask_border_pt_idx[idx_t] % 512) / 512 * ori_img_width))
        temp_pt.append(int(float(front_mask_border_pt_idx[idx_t] // 512) / 512 * ori_img_height))
        front_mask_border_pt.append(temp_pt)
    front_mask_border_pt = np.array(front_mask_border_pt)
    front_mask_border_pt = front_mask_border_pt.reshape((1, -1, 2))
    front_mask = np.zeros((ori_img_height, ori_img_width), np.uint8)
    cv2.polylines(front_mask, [front_mask_border_pt], True, 255)
    cv2.fillPoly(front_mask, [front_mask_border_pt], 255)
    # cv2.imwrite('./front_mask-gen-deploy.jpg', front_mask)
    # front_mask //= 255
    if np.max(front_mask) != 255:
        print('np.max(front_mask)!=255 ', np.max(front_mask))
        input()
    if np.min(front_mask) != 0:
        print('np.min(front_mask)!=0 ', np.min(front_mask))
        input()
    if is_color:
        front_mask = 255 - front_mask
        color_front_mask = np.stack([front_mask, front_mask, front_mask], axis=2)
        return color_front_mask
    else:
        return front_mask

def create_train_data():
    # Data Load
    i=0
    if FLAGS.from_database:
        img_dict_from_database_start = get_data_from_database.request_files(custom_root_dir=FLAGS.data_dir,
            data_family=FLAGS.data_family, time_filter=FLAGS.start_time)
        img_dict_from_database_end = get_data_from_database.request_files(custom_root_dir=FLAGS.data_dir,
            data_family=FLAGS.data_family, time_filter=FLAGS.end_time)
        data_path_list = []
        end_img_lst = [list(dic.keys())[0] for dic in img_dict_from_database_end]
        for sid, start in enumerate(img_dict_from_database_start):
            print('select date rate:', float(sid)/len(img_dict_from_database_start))
            img_pth = list(start.keys())[0]
            if img_pth in end_img_lst:
                continue
            data_path_list.append(start)
            i += 1
    else:
        # data_path_list = fisheye_img_util.get_files(FLAGS.data_dir)
        data_path_list = os.listdir(FLAGS.data_dir)

        
    print("total: ", i)
    # initial
    if not os.path.exists(FLAGS.visual_dir):
        os.mkdir(FLAGS.visual_dir)
    # TFRecord create
    if FLAGS.data_family == 'training' or FLAGS.data_family == 'pre_training':
      train_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%FLAGS.output_path)
    else:
      val_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord' % FLAGS.output_path)
    # Loop
    img_label_idx = 0
   
    for image_path in data_path_list:
        if image_path.endswith(('.jpg', '.png')):
            image_path = FLAGS.data_dir + '/' + image_path
            print("Write the tfrecord of [" + image_path + ']')
            try:
                image_np = cv2.imread(image_path)
            except FileNotFoundError:
                print("------WRONG IMAGE------ [", image_path, "]" )
                continue
            
            height, width, _ = image_np.shape
            tf_height, tf_width = FLAGS.tfrecord_height, FLAGS.tfrecord_width
            img_base, img_type = os.path.splitext(image_path)
            json_path = img_base + '.json'

            # resize
            image_object = cv2.resize(image_np, (tf_height, tf_width))

            try:
                with io.open(json_path, 'r', encoding='gbk') as load_f:
                    dict_inst = json.load(load_f)
            except Exception as e:
                print("------WRONG JSON------[", json_path, "]")
                continue

            if 'shapes' in dict_inst.keys():
                shapes = dict_inst['shapes']
            else:       
                print("------WRONG Dict Keys (shapes)------[", json_path, "]")
                continue
        
            xmin_norm = []
            ymin_norm = []
            xmax_norm = []
            ymax_norm = []
            class_ids = []
            for sid ,shp in enumerate(shapes):
                if (not 'label' in shp.keys()) or ():
                        print("------WRONG shapes Keys (label)------[", json_path, "]")
                        continue
                # if (not shp['label'] == FLAGS.label):
                #         continue
                if not 'points' in shp.keys():
                    print("------WRONG shapes Keys (points)------[", json_path, "]")
                    continue
                points = shp['points']
                if len(points) != 2 or len(points[0]) != 2:
                    print("------WRONG shapes Keys Value (points)------[", json_path, "]")
                    continue
                xmin = points[0][0]
                ymin = points[0][1]
                xmax = points[1][0]
                ymax = points[1][1]
                
                if(xmax - xmin <= 5):
                    print("------object is too small------[", json_path, "]")
                    continue
            
                # xmin_norm.append(int(float(xmin) / width_rot * tf_width))
                # ymin_norm.append(int(float(ymin) / height_rot * tf_height))
                # xmax_norm.append(int(float(xmax) / width_rot * tf_width))
                # ymax_norm.append(int(float(ymax) / height_rot * tf_height))
                xmin_norm.append(int(float(xmin) / width * tf_width))
                ymin_norm.append(int(float(ymin) / height * tf_height))
                xmax_norm.append(int(float(xmax) / width * tf_width))
                ymax_norm.append(int(float(ymax) / height * tf_height))
                target_label = shp['label']  # 'Object'
                class_ids.append(classes_label_map[target_label] + 1)
                
                cv2.line(image_object, (int(float(xmin) / width * tf_width), 
                                            int(float(ymin) / height * tf_height)), 
                                        (int(float(xmax) / width * tf_width), 
                                            int(float(ymin) / height * tf_height)), (0,255,255), 2)
                cv2.line(image_object, (int(float(xmin) / width * tf_width), 
                                            int(float(ymin) / height * tf_height)), 
                                        (int(float(xmin) / width * tf_width), 
                                            int(float(ymax) / height * tf_height)), (0,255,255), 2)
                cv2.line(image_object, (int(float(xmax) / width * tf_width), 
                                            int(float(ymin) / height * tf_height)), 
                                        (int(float(xmax) / width * tf_width), 
                                            int(float(ymax) / height * tf_height)), (0,255,255), 2)
                cv2.line(image_object, (int(float(xmin) / width * tf_width), 
                                            int(float(ymax) / height * tf_height)), 
                                        (int(float(xmax) / width * tf_width), 
                                            int(float(ymax) / height * tf_height)), (0,255,255), 2)
                cv2.imwrite('/dta/llx/yolov3_face_detect/get_require/res.png', image_object)
            
            img_label_idx += 1
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': dataset_util.bytes_feature(image_object.tostring()),
                'image/height': dataset_util.int64_feature(FLAGS.tfrecord_height),
                'image/width': dataset_util.int64_feature(FLAGS.tfrecord_width),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
                'image/object/class/label': dataset_util.int64_list_feature(class_ids),
                }))

            # visualization
            vis = True
            if vis:
                visualization(image_object, xmin_norm, ymin_norm, xmax_norm, ymax_norm, 
                            img_label_idx, image_path, FLAGS.visual_dir)

            if FLAGS.data_family == 'training' or FLAGS.data_family == 'pre_training':
                train_writer.write(example.SerializeToString())
            else:
                val_writer.write(example.SerializeToString())
            
            if img_label_idx % 100 == 0:
                print("Rate : ", float(img_label_idx) / len(data_path_list))

    if FLAGS.data_family == 'training' or FLAGS.data_family == 'pre_training':
      train_writer.close()
    else:
      val_writer.close()
    print("total image: ", img_label_idx)
#   print(width_heights)
#   anchors = k_means(np.asarray(width_heights), 9, 0.01)
#   anchors = anchors*[FLAGS.img_width, FLAGS.img_height]
#   print("Anchors = ", anchors)
#   areas = anchors[:,0]*anchors[:,1]
#   print("index =", np.argsort(areas))
    print("End the tfrecord creating.")


def main(_):
    create_train_data()

if __name__ == '__main__':
  tf.app.run()