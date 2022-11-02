#-*- coding:utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
import cv2
import os
from config import config 
import time
import shutil
import util
from math import pi
from numpy import cos, sin
import time
import time
import tensorflow as tf
import numpy as np

def cal_tp_fp(pred_c, gt, ovthresh, original_img_w, original_img_h):
    tp_image = 0
    fp_image = 0
    total_image = len(gt) 
   
    has_det = []
    for i in range(len(pred_c)):
        pred_box = pred_c[i]
        # if BBGT.size > 0:
        ptx1_resized = int(pred_box[0] * original_img_w)
        ptx2_resized = int(pred_box[2] * original_img_w)
        pty1_resized = int(pred_box[1] * original_img_h)
        pty2_resized = int(pred_box[3] * original_img_h)
        
        for j in range(len(gt)):
            if j in has_det:
                continue
            ixmin = np.maximum(ptx1_resized, gt[j][0])
            iymin = np.maximum(pty1_resized, gt[j][1])
            ixmax = np.minimum(ptx2_resized, gt[j][2])
            iymax = np.minimum(pty2_resized, gt[j][3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            # intersection set
            inters = iw * ih  
            # union set
            uni = ((gt[j][2] - gt[j][0] + 1.) * (gt[j][3] - gt[j][1] + 1.) +
                    (ptx2_resized - ptx1_resized + 1.) *
                    (pty2_resized - pty1_resized + 1.) - inters)
            
            # iou = intersection : union
            overlaps = inters / uni
            ovmax = np.max(overlaps) # 最大重叠
            jmax = np.argmax(overlaps) # 最大重合率对应的gt
            # 计算tp 和 fp个数
            if ovmax > ovthresh:
                has_det.append(j)
                tp_image += 1.
        if not has_det:
            fp_image += 1.
    return tp_image, fp_image, total_image

def sigmoid(data):
    return 1.0 / (1.0 + np.exp(-data))

def boxes_in_batch(predictions, origin_width, origin_height, 
    number_object, number_class, biase, conf_thresh, class_conf):
    """ predictions shape 
    [height, width, num_object*(num_class + 5)]
    """
    grid_width = predictions.shape[1]
    grid_height = predictions.shape[0]
    x, y = np.meshgrid(
        np.linspace(0, grid_width - 1, grid_width),
        np.linspace(0, grid_height - 1, grid_height))
    boxes = []
    confs = []
    cls_confs = []
    max_class= 0
    for b in range(number_object):
        classes = sigmoid(predictions[:, :, 
            b * (number_class + 5) + 5 : b * (number_class + 5) + 5 + number_class])
        # max_id = np.argmax(classes, axis = 2)
        # max_classes_conf = classes[:, :, max_id]
        # max_class = max_id if max_classes_conf > class_conf else max_class
        object_confidence = sigmoid(
            predictions[:, :, b * (number_class + 5) + 4])
        coord = predictions[:, :, b * (number_class + 5) : b * (number_class + 5) + 4]
        coord[:, :, 2] = np.exp(coord[:, :, 2]) * biase[2 * b]
        coord[:, :, 3] = np.exp(coord[:, :, 3]) * biase[2 * b + 1]
        coord[:, :, 0] = (x + sigmoid(coord[:, :, 0])) / grid_width - coord[:, :, 2] / 2
        coord[:, :, 1] = (y + sigmoid(coord[:, :, 1])) / grid_height - coord[:, :, 3] / 2
        for i in range(grid_height):
            for j in range(grid_width):
                box = [coord[i, j, 0], coord[i, j, 1], 
                    coord[i, j, 0] + coord[i, j, 2], 
                    coord[i, j, 1] + coord[i, j, 3]]
                box = interset(box, [0.0, 0.0, 1.0, 1.0])
                if object_confidence[i, j] > conf_thresh:
                    # box.append(object_confidence[i, j])
                    box.append(0) # 4: class_id
                    box.extend(classes[i, j, :] * object_confidence[i, j]) # 5: score
                    boxes.append(box)
                    confs.append(object_confidence[i, j])
                    # cls_confs.append(classes[i, j, max_id])
                    # print(box)
                    # print(object_confidence[i,j])
    return boxes, confs, cls_confs

def interset(box1, box2):
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    return [left, top, right, bottom]

#查找根目录下的所有文件夹
def dirAll(pathname):
    if os.path.exists(pathname) and os.path.isdir(pathname):
        filelist = os.listdir(pathname)
        for f in filelist:
            f = os.path.join(pathname, f)
            if os.path.isdir(f) and os.path.basename(f) == "fisheye":
                img_list = os.listdir(f)
                for img_f in img_list:
                    if img_f.endswith(".jpg") and len(img_f) == 20:
                        img_path = os.path.join(f, img_f)
                        file_list.append(img_path)

                        has_label = 0
                        for label_f in img_list:
                            if (img_f[:16] == label_f[:16]) and (img_f != label_f):
                                if (label_f[-17:] == '-person-boxes.txt'):
                                    has_label = 1
                                    label_f = os.path.join(f, label_f)
                                    label_list.append(label_f)
                                    # print(img_path)
                                    # print(label_f)
                                    break
                        if has_label == 1:
                            continue
                        else:
                            file_list.pop()
                
            else:
                dirAll(f)

if __name__ == '__main__':
    conf = config.get_params()
    os.environ['CUDA_VISIBLE_DEVICES']=conf['gpus']

    # Dataset Loader
    test_img_root_folder = conf['test_img_folder']
    custom_database_folder = conf['custom_database_folder']
    database_data_time_filter_start = conf['database_data_time_filter_start']
    database_data_time_filter_end = conf['database_data_time_filter_end']
    get_database_data = conf['get_database_data']
    test_img_folder_list = []
    database_data_dict = {}
    data_family = conf['data_family']

    # Parameter prepare
    seg_button = conf['seg_button']
    pen_button = conf['pen_button']
    net_input_h = conf['net_input_h']
    net_input_w = conf['net_input_w']
    output_dir_root = conf['output_dir']

    # Seg param init
    class_num = conf['class_num']
    # Pedestrian param init
    num_object = conf['num_object'] 
    num_class = conf['classes']
    anchors = conf['anchors']
    conf_thresh = conf['conf_thresh']
    class_thresh = conf['class_thresh']
    nms_thresh = conf['nms_thresh']

    valid_person_radius = conf['valid_person_radius']
    angles_circle = [i * pi/180 for i in range(-45, 45)]
    coord_circle = np.stack((cos(angles_circle), sin(angles_circle)), axis = 1) * valid_person_radius
    coord_circle_idx = np.linspace(1, 89, 30, dtype = int)
    coord_circle = coord_circle[coord_circle_idx].astype(np.int64)

    # anchor shrink
    for i in range(0, len(anchors), 2):
        anchors[i] = anchors[i] / net_input_w
        anchors[i + 1] = anchors[i + 1] / net_input_h

    if not os.path.exists(output_dir_root):
        os.mkdir(output_dir_root)
    else:
        if os.listdir(output_dir_root):
            print('output_dir_root is not empty, if need to remove: ', os.listdir(output_dir_root))
            print('press enter to remove: ', output_dir_root)
            input()
            shutil.rmtree(output_dir_root)
            os.mkdir(output_dir_root)
  
    path=r'./person_ground_truth/'
       
    file_list = []
    label_list = []
    dirAll(path)
  
    cur_img_idx = 0
    time_model_run = 0.0     
    tp_dataset = 0
    fp_dataset = 0
    total_dataset = 0
    total_valid_img_num = 0
    img_num = 0
 
    for img_index in range(len(file_list)):  
        img_num += 1      
        img_path = file_list[img_index]
        # Input image is placed horizontally. (H < W)
        np_array = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        filename = filename.split('.')[0]
        
        original_img_w = np_array.shape[1]
        original_img_h = np_array.shape[0]
        # ori_show_img = np.rot90(np_array)
        ori_show_img = np_array
        show_img = cv2.resize(np_array, (net_input_w, net_input_h)).copy()
 
        shape_t = np_array.shape
        crop_x = shape_t[1]
        np_array = np_array[0:shape_t[0], 0:crop_x].copy()

        batch_size = 1 #16
        target_size = (net_input_w, net_input_h)
                        
        if pen_button:
            # image_crop for pedestrian detection
            np_array_crop = np_array
            np_array_crop = cv2.resize(np_array_crop, target_size)
            np_array_crop = np.asarray(np_array_crop).astype(np.float32).copy()
    
        # For Segmentation
        sidewalk_flag = False # use for filter pedestrian (init)
        seg_choose = np.zeros((net_input_w, net_input_h), dtype = np.uint8)
                
        all_gt_dict = []           
        total_anchors = []
        for i in range(0, 6 * conf['num_object'], 2):
            total_anchors.append(conf['anchors'][i] / conf['img_width'])
            total_anchors.append(conf['anchors'][i + 1] / conf['img_height'])
                
        if pen_button:
            cv2.imwrite("/dta/llx/yolov3/get_require/res.png", np_array_crop)
            # do not quantize
            np_array_crop = np_array_crop/255.0
            np_array_crop = np.array(np_array_crop, dtype=np.float32)
            
            # quantize
            # np_array_crop = np.array(np_array_crop, dtype=np.uint8) 
            # interpreter = tf.contrib.lite.Interpreter(
            #     model_path="./best_distillation_float_model_folder/eval_graf_dont_aug_1.0_103300_quantize.tflite")
            
            interpreter = tf.contrib.lite.Interpreter(
                model_path="./best_distillation_float_model_folder/float-mbV1yoloV3-0.25-1task-103300-Aug-PretrainCoco-pedestrian_Aibox.tflite")

            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
        
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], [np_array_crop])
            interpreter.invoke()
            predictions1 = interpreter.get_tensor(output_details[0]['index'])
            predictions2 = interpreter.get_tensor(output_details[1]['index'])
            predictions3 = interpreter.get_tensor(output_details[2]['index'])   
            end_time = time.time()
            time_model_run += (end_time - start_time)
                           
            pedestrian_res, pedestrian_coord, pedestrian_bboxes, other_coord, \
                    other_bboxes, box_offsets, pedestrian_conf = [], [], [], [], [], [], []
            
            #predict_res
            boxes = []
            for b in range(predictions1.shape[0]):
                stage_idx = 1
                for pred_input in [predictions1, predictions2, predictions3]:
                    boxes_tmp, confs, cls_confs = boxes_in_batch(pred_input[b,:,:,:], 
                        net_input_w, net_input_h, num_object, num_class, 
                        anchors[(stage_idx - 1) * 2 * num_object : stage_idx * 2 * num_object],
                        conf_thresh, class_thresh)
                    boxes.extend(boxes_tmp)
                    stage_idx = stage_idx +1
                boxes = np.asarray(boxes)
                print("len",len(boxes))
                if boxes.shape[0] > 0:
                    nms_box =util.nms(boxes, num_class, nms_thresh)
                    print("after nms len",len(nms_box))
                    # pedestrian_res = copy.deepcopy(nms_box)
                    pedestrian_res = nms_box
                              
            #label res
            box_file = label_list[img_index]
            if(os.path.exists(box_file)):
                print('label file exists')
                for line in open(box_file):
                    sep_four = line.strip().split('],')
                    sep = sep_four[0].split(',')
                    sep[0] = sep[0][2:]
                    for i in range(len(sep)):
                        sep[i] = sep[i].strip()
                    
                    box=[]
                    box.append(float(sep[0]))   #left-up x1
                    box.append(float(sep[1]))   #left-up y1
                    box.append(float(sep[2]))   #right-down x2
                    box.append(float(sep[3]))   #right-down y2
                    box.append(float(sep[4]))
                    all_gt_dict.append(box)
            tp_image, fp_image , total_image = cal_tp_fp(pedestrian_res, all_gt_dict, 0.5, original_img_w, original_img_h)                   
            tp_dataset += tp_image
            fp_dataset += fp_image
            total_dataset += total_image
            print("[tp_dataset, fp_dataset, total_dataset]: ", tp_dataset, fp_dataset, total_dataset )
            
            for bid in range(len(pedestrian_res)):
                #--llx --20220319
                ptx1_resized = int(pedestrian_res[bid][0] * original_img_w)
                ptx2_resized = int(pedestrian_res[bid][2] * original_img_w)
                pty1_resized = int(pedestrian_res[bid][1] * original_img_h)
                pty2_resized = int(pedestrian_res[bid][3] * original_img_h)
                
                ptx1_resized, pty1_resized, ptx2_resized, pty2_resized = \
                    interset([ptx1_resized, pty1_resized, ptx2_resized, pty2_resized],
                                [0, 0, net_input_w - 1, net_input_h - 1])

                if (not sidewalk_flag):
                    b_offset = 0
                    pedestrian_coord.append([
                        ptx1_resized, 
                        int((pty1_resized + pty2_resized)/2)])
                    pedestrian_bboxes.append([
                        ptx1_resized, pty1_resized, 
                        ptx2_resized, pty2_resized])            
                    box_offsets.append(b_offset)
                    pedestrian_conf.append(pedestrian_res[bid][5])

                else:
                    # Debug
                    other_coord.append([
                        ptx1_resized, 
                        int((pty1_resized + pty2_resized) / 2)])
                    other_bboxes.append([ 
                        ptx1_resized, pty1_resized, 
                        ptx2_resized, pty2_resized])

            # Distance : to birdview (original H_Mat) -> Must calculate in resolution 512*512
            pedestrian_coord_to_birdview = []
            for pid, p_coord in enumerate(pedestrian_coord):
                pedestrian_coord_to_birdview.append(p_coord)
            pedestrian_coord_to_birdview = np.array(pedestrian_coord_to_birdview)

            # Pedestrian Show
            person_count_res = ori_show_img.copy()
            person_count_res_dis = person_count_res.copy()
            person_count_res_conf = person_count_res.copy()
            valid_color, invalid_color = (0, 0, 255), (0, 255, 0)
            for cid, dot_coord in enumerate(pedestrian_coord): # (x,y) in (512, 512)
                person_count_res = cv2.rectangle(
                    person_count_res,
                    (pedestrian_bboxes[cid][0],
                    pedestrian_bboxes[cid][1]),
                    (pedestrian_bboxes[cid][2], 
                    pedestrian_bboxes[cid][3]),
                    valid_color,
                    2, -1
                )
                
            for oid, oth_coord in enumerate(other_coord):
                person_count_res = cv2.circle(
                    person_count_res, 
                    (oth_coord[0] * original_img_w // net_input_w, 
                        oth_coord[1] * original_img_h // net_input_h),
                    5, invalid_color, -1)
                person_count_res = cv2.rectangle(
                    person_count_res,
                    (other_bboxes[oid][0] * original_img_w // net_input_w, 
                        other_bboxes[oid][1] * original_img_h // net_input_h),
                    (other_bboxes[oid][2] * original_img_w // net_input_w, 
                        other_bboxes[oid][3] * original_img_h // net_input_h),
                    invalid_color,
                    2, -1
                )
                 
        # show
        overlapping = np.zeros((shape_t[1], shape_t[0]), np.uint8)
    
        overlapping = person_count_res.copy()

        cv2.imwrite(output_dir_root + '/' + filename + '-res.png', overlapping)
        total_valid_img_num += 1  
       
    #cal ap and recall 
    rec = tp_dataset / float(total_dataset)
    pre = tp_dataset / np.maximum(tp_dataset + fp_dataset, np.finfo(np.float64).eps)  
    print("recall is : ", rec)
    print("pre is : ", pre)   
    print("mean per image time_model_run: ", time_model_run / img_num)