import numpy as np
from boxes_parser import util

import time

def sigmoid(data):
    return 1.0 / (1.0 + np.exp(-data))

def softmax(data):
    """ data shape [grid, grid, num_class] """
    max_value = np.amax(data, axis = 2)
    data = np.exp(data - max_value[:, :, np.newaxis])
    sum_value = np.sum(data, axis = 2)
    return data / sum_value[:,:,np.newaxis]

def boxes_in_yolo_v3_batch(predictions, image_id, number_object, number_class, 
    biase):
    """ predictions shape [height, width, num_object*(num_class + 5)]"""
    grid_height, grid_width, _ = predictions.shape
    x, y = np.meshgrid(
        np.linspace(0, grid_width - 1, grid_width), 
        np.linspace(0, grid_height - 1, grid_height))
    boxes = []
    for b in range(number_object):
        t1=time.time()
        classes = sigmoid(predictions[:, :, 
            b * (number_class + 5) + 5 : b * (number_class + 5) + 5 + number_class])
        object_confidence = sigmoid(predictions[:, :, 
            b * (number_class + 5) + 4 : b *(number_class + 5) + 5])
        coord = predictions[:, :, 
            b * (number_class + 5) : b * (number_class + 5) + 4]
        #print("c:",coord[:,:,2],coord[:,:,3])
        #print("b:",b,biase[2*b],biase[2*b+1])
        coord[:, :, 2] = np.exp(coord[:, :, 2]) * biase[2 * b]
        coord[:, :, 3] = np.exp(coord[:, :, 3]) * biase[2 * b + 1]
        #print("coord:",coord[:,:,2],coord[:,:,3])
        coord[:, :, 0] = ((x + sigmoid(coord[:, :, 0])) / grid_width - 
                        coord[:, :, 2] / 2)
        coord[:, :, 1] = ((y + sigmoid(coord[:, :, 1])) / grid_height - 
                        coord[:, :, 3] / 2)
        score = classes * object_confidence
        t2=time.time()
        for i in range(grid_height):
            for j in range(grid_width):
                if object_confidence[i,j,0] > 0.9 and max(classes[i,j,:] > 0.5):
                    if(coord[i,j,2] < 0.001 or coord[i,j,3] < 0.001):
                        continue
                    box = [coord[i, j, 0], coord[i, j, 1], 
                        coord[i, j, 0] + coord[i, j, 2], 
                        coord[i, j, 1] + coord[i, j, 3]]
                    box = util.interset(box, [0.0, 0.0, 1.0, 1.0])
                    #print("box:", box)
                #if object_confidence[i,j,0] > 0.9999:
                    box.append(image_id)
                    #box.extend(score[i, j, :])
                    box.append(max(score[i, j, :]))
                    box.append(np.argmax(score[i,j,:]))
                    boxes.append(box)
        t3=time.time()
        #print("exten:",t2-t1,t3-t2)
    return boxes

def boxes_in_yolo_v2_batch(predictions, image_id, number_object, number_class, 
    biase):
    """ predictions shape [batch_size, height, width, num_object*(num_class + 5)]"""
    grid_height, grid_width, _ = predictions.shape
    x, y = np.meshgrid(
        np.linspace(0, grid_width - 1, grid_width), 
        np.linspace(0, grid_height - 1, grid_height))
    boxes = []
    for b in range(number_object):
        if number_class > 1:
            classes = softmax(predictions[:, :, 
                b * (number_class + 5) + 5 : b * (number_class + 5) + 5 + number_class])
        else:
            classes = sigmoid(predictions[:, :, 
                b * (number_class + 5) + 5: b * (number_class + 5) + 6])
        object_confidence = sigmoid(
            predictions[:, :, b * (number_class + 5) + 4])
        coord = predictions[:, :, 
            b * (number_class + 5) : b * (number_class + 5) + 4]
        coord[:, :, 2] = np.exp(coord[:, :, 2]) * biase[2 * b] / grid_width
        coord[:, :, 3] = np.exp(coord[:, :, 3]) * biase[2 * b + 1] / grid_height
        coord[:, :, 0] = (x + 
            sigmoid(coord[:, :, 0])) / grid_width  - coord[:, :, 2] / 2
        coord[:, :, 1] = (y + 
            sigmoid(coord[:, :, 1])) / grid_height - coord[:, :, 3] / 2
        class_id = np.argmax(classes, axis=2)
        score = object_confidence * np.amax(classes, axis=2)
        for i in range(grid_height):
            for j in range(grid_width):
                box = [coord[i, j, 0], coord[i, j, 1], 
                    coord[i, j, 0] + coord[i, j, 2], 
                    coord[i, j, 1] + coord[i, j, 3]]
                box = util.interset(box, [0, 0, 1.0, 1.0])
                box.append(image_id)
                box.append(score[i, j])
                box.append(class_id[i,j])
                if score[i,j]> 0.3:
                    boxes.append(box)
    return np.asarray(boxes)

def boxes_in_label(label, num_object, number_class):
    grid_width = label.shape[1]
    grid_height = label.shape[0]
    boxes= []
    for i in range(grid_height):
        for j in range(grid_width):
            for k in range(num_object):
                #print("label:",i,j,k,label[i,j,k*(number_class+5)+4])
                if label[i, j, k * (number_class + 5) + 4] > 0.5:
                    h = label[i, j, k * (number_class + 5) + 3]
                    w = label[i, j, k * (number_class + 5) + 2]
                    y = label[i, j, k * (number_class + 5) + 1] - h / 2
                    x = label[i, j, k * (number_class + 5) + 0] - w / 2
                    for c in range(number_class):
                        if label[i, j, k * (number_class + 5) + 5 + c] > 0.5:
                            boxes.append([x, y, x + w, y + h, c])
                            #print("truth:",[x,y,x+w,y+h,c])
    return boxes


def collect_predict_gt_in_yolo_v2_batch(image_index, input, label, all_gt_dict, 
    all_predict_boxes, number_object, number_class, biase):
    for b in range(input.shape[0]):
        boxes = boxes_in_yolo_v2_batch(input[b, :, :, :], image_index, 
            number_object, number_class, biase)
        nms_box = util.nms(boxes, number_class, has_class_id=True)
        gt_box = boxes_in_label(label[b, :, :, :], number_object, number_class)
        for box in gt_box:
            class_id = int(box[4])
            if image_index in all_gt_dict[class_id]:
                all_gt_dict[class_id][image_index].append(box)
            else:
                all_gt_dict[class_id][image_index] = [box]
        for box in nms_box:
            class_id = int(box[4])
            all_predict_boxes[class_id].append(box)
        image_index = image_index + 1
    return image_index

def collect_predict_gt_in_yolo_v3_batch(image_index, inputs, label, all_gt_dict, 
    all_predict_boxes, number_object, number_class, biase):
    for b in range(inputs[0].shape[0]):
        t1=time.time()
        boxes = []
        stage_idx = 1
        for input in inputs:
            #if stage_idx != 2:
                #stage_idx = stage_idx +1
                #continue
            #print("parse", stage_idx)
            boxes.extend(boxes_in_yolo_v3_batch(input[b, :, :, :], image_index, 
                number_object, number_class, 
                biase[(stage_idx - 1) * 2 * number_object : stage_idx * 2 * number_object]))
            stage_idx = stage_idx +1
        boxes = np.asarray(boxes)
        t2=time.time()
        if boxes.shape[0] > 0:
            nms_box =util.nms(boxes, number_class,  has_class_id=True)
            for box in nms_box:
                class_id = int(box[4])
                all_predict_boxes[class_id].append(box)
        t3=time.time()
        gt_box = boxes_in_label(label[b, :, :, :], number_object, number_class)
        for box in gt_box:
            class_id = int(box[4])
            if image_index in all_gt_dict[class_id]:
                all_gt_dict[class_id][image_index].append(box)
            else:
                all_gt_dict[class_id][image_index] = [box]

        image_index = image_index + 1
        t4=time.time()
        #print("parse:",t2-t1,t3-t2,t4-t3)
    return image_index

def collect_predict_gt_in_trident_batch(image_index, inputs, label, 
    all_gt_dict, all_predict_boxes, number_object, number_class, biase):
    for b in range(inputs[0].shape[0]):
        boxes = []
        stage_idx =1
        for input in inputs:
            boxes.extend(boxes_in_batch_v2(input[b,:,:,:], image_index, 
                number_object, number_class, 
                biase[(stage_idx - 1) * 2 * number_object : stage_idx * 2 * number_object]))
            stage_idx = stage_idx + 1
        boxes = np.asarray(boxes)
        if boxes.shape[0] > 0:
            nms_box =util.nms(boxes, number_class, has_class_id=True)
            for box in nms_box:
                class_id = int(box[4])
                all_predict_boxes[class_id].append(box)
        gt_box = boxes_in_label(label[b, :, :, :], number_object, number_class)
        for box in gt_box:
            class_id = int(box[4])
            if image_index in all_gt_dict[class_id]:
                all_gt_dict[class_id][image_index].append(box)
            else:
                all_gt_dict[class_id][image_index] = [box]
        image_index = image_index + 1
    return image_index
