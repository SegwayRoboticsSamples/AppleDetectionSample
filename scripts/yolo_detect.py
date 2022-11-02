#!/usr/bin/env python3
import sys
import time
import cv2

sys.path.append("../")

import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from boxes_parser import util
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
anchors= [169.54, 131.54, 228.0, 289.38, 545.15, 476.46, 43.85, 89.15, 90.62, 65.77, 86.23, 173.92, 14.62, 19.0, 23.38, 43.85, 48.23, 33.62]
#anchors= [0.1,0.1,0.1,0.1,0.1,0.1, 7.29,37.2, 25.70,143.34, 69.49,349.21, 0.1,0.1,0.1,0.1,0.1,0.1]
class_names = ["Pedestrian"]
#anchors=[5.65, 10.10, 13.95, 26.66, 22.72, 64.96, 45.83, 31.11, 39.84, 125.39, 
    #66.85, 73.67, 84.02, 162.85, 154.88, 87.43, 172.73, 185.79]
#class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
    #'traffic light', 'bench', 'cat', 'dog']

def sigmoid(data):
    return 1.0 / (1.0 + np.exp(-data))

def softmax(data):
    """ data shape [grid, grid, num_class] """
    max_value = np.amax(data, axis = 2)
    data = np.exp(data - max_value[:, :, np.newaxis])
    sum_value = np.sum(data, axis = 2)
    return data / sum_value[:, :, np.newaxis]
def interset(box1, box2):
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    return [left, top, right, bottom]

def boxes_in_batch(predictions, image_id, origin_width, origin_height, 
    number_object, number_class, biase):
    """ predictions shape 
    [height, width, num_object*(num_class + 5)]
    """
    grid_width = predictions.shape[1]
    grid_height = predictions.shape[0]
    x, y = np.meshgrid(
        np.linspace(0, grid_width - 1, grid_width), 
        np.linspace(0, grid_height - 1, grid_height))
    boxes = []
    for b in range(number_object):
        classes = sigmoid(predictions[:, :, 
            b * (number_class + 5) + 5 : b * (number_class + 5) + 5 + number_class])
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
                if object_confidence[i, j] > 0.9:
                    box.append(image_id)
                    box.extend(classes[i, j, :] * object_confidence[i, j])
                    boxes.append(box)
                    print(box)
    return boxes



def make_square(oring_im, desired_size=608, fill_color=(0, 0, 0)):
    old_size = [oring_im.shape[1],oring_im.shape[0]]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = cv2.resize(oring_im, new_size)
    #new_im = Image.new('RGB', (desired_size, desired_size), fill_color)
    #new_im.paste(im, 
    #    (int((desired_size - new_size[0]) / 2), 
    #    int((desired_size - new_size[1]) / 2)))
    dx = (desired_size - new_size[0]) / 2;
    dy = (desired_size - new_size[1]) / 2;
    new_im = cv2.copyMakeBorder(im, int(dy), int(dy), int(dx), int(dx), cv2.BORDER_CONSTANT, value=0)
    dx = dx / desired_size
    dy = dy / desired_size
    
    x_scale = new_size[0] / desired_size
    y_scale = new_size[1] / desired_size
    print(dx,dy,x_scale,y_scale,new_im.shape)
    return new_im, dx, dy, x_scale, y_scale

def main():
    graph_def = tf.GraphDef()
    model_width = 608
    model_height = 608
    num_object = 3
    num_class =1
    with tf.gfile.FastGFile('./cocoperson1.0_normal.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    origin = Image.open('post2.jpeg')
    origin_img = cv2.imread('post2.jpeg')
    origin_height, origin_width = origin_img.shape[0:2]
    print(origin_width, origin_height)
    raw_img, dx, dy, x_scale, y_scale = make_square(origin_img)
    img = np.asarray(raw_img).astype(np.float32)
    print(np.max(img),np.min(img))

    draw = ImageDraw.Draw(origin)
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/freefont/FreeMono.ttf', 34)
    for i in range(0, len(anchors), 2):
        anchors[i] = anchors[i] / model_width
        anchors[i + 1] = anchors[i + 1] / model_height
    with tf.Session() as sess:
        output_tensor1 = sess.graph.get_tensor_by_name('Conv_6/BiasAdd:0')
        output_tensor2 = sess.graph.get_tensor_by_name('Conv_14/BiasAdd:0')
        output_tensor3 = sess.graph.get_tensor_by_name('Conv_22/BiasAdd:0')
        predictions1, predictions2, predictions3 = sess.run(
            [output_tensor1, output_tensor2, output_tensor3], 
            {'Placeholder:0': [img / 255.0]})
        image_index = 0
   
        for b in range(predictions2.shape[0]):
            boxes = []
            stage_idx = 2
            for input in [predictions2]:
                boxes.extend(boxes_in_batch(input[b,:,:,:], image_index, 
                    model_width, model_height, num_object, num_class, 
                    anchors[(stage_idx - 1) * 6 : stage_idx * 6]))
                stage_idx = stage_idx +1
            boxes = np.asarray(boxes)
            print("len",len(boxes))
            if boxes.shape[0] > 0:
                nms_box =util.nms(boxes, num_class)
                for box1 in nms_box:
                    if box1[5] < 0.5:
                        continue
                    box1 = np.asarray(box1)
                    box=box1[0 : 5]
                    print(box)
                    
                    box[0] = int((box[0] - dx) / x_scale * origin_width)
                    box[1] = int((box[1] - dy) / y_scale * origin_height)
                    box[2] = int((box[2] - dx) / x_scale * origin_width)
                    box[3] = int((box[3] - dy) / y_scale * origin_height)
                    box = interset(box,[0.0,0.0, origin_width,origin_height])
                    print(box)
                    #draw.text((box[0], box[1]), class_names[int(box1[4])],
                        #(255, 255, 0),font)
                    draw.line((box[0], box[1], box[2], box[1]), fill=128, 
                        width=6)
                    draw.line((box[0], box[3], box[2], box[3]), fill=128, 
                        width=6)
                    draw.line((box[0], box[1], box[0], box[3]), fill=128, 
                        width=6)
                    draw.line((box[2], box[1], box[2], box[3]), fill=128, 
                        width=6)
                origin.save('test2.jpg')

if __name__ == '__main__':
    main()
