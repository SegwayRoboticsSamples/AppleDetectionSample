#!/usr/bin/env python3

import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import util
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
anchors=[14.18049792, 25.83762102, 31.14735724, 63.13654239, 63.51648073,
    78.50980392, 61.67917633, 138.83387471, 135.43246753, 175.65281385]
class_names = ["elvator", "escalator", "GX", "turnstile"]

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

def boxes_in_batch(predictions, image_id, number_object, number_class, biase):
    """predictions shape:
        [batch_size, height, width, num_object * (num_class + 5)]
    """
    grid_height, grid_width,_ = predictions.shape[1]
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
            b * (number_class + 5) + 5:b * (number_class + 5) + 6])
        object_confidence = sigmoid(
            predictions[:, :, b * (number_class + 5) + 4])
        coord = predictions[:, :, b * (number_class + 5)  : b * (number_class + 5) + 4]
        coord[:, :, 2] = np.exp(coord[:, :, 2]) * biase[2 * b] / grid_width
        coord[:, :, 3] = np.exp(coord[:, :, 3]) * biase[2 * b + 1] /grid_height
        coord[:, :, 0] = (x + sigmoid(coord[:, :, 0])) / grid_width  - coord[:, :, 2] / 2
        coord[:, :, 1] = (y + sigmoid(coord[:, :, 1])) / grid_height - coord[:, :, 3] / 2
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
                boxes.append(box)
    return np.asarray(boxes)

def resize_image_keep_ratio(origin_im, desired_size=(320, 320), 
    fill_color=(0, 0, 0)):
    old_width, old_height = origin_im.size

    resize_width, resize_height = desired_size
    ratio = min(float(resize_height) / old_height, 
        float(resize_width) / old_width)
    new_size = tuple([int(old_width * ratio), int(old_height*  ratio)])
    im = origin_im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new('RGB', desired_size, fill_color)
    new_im.paste(im, (int((resize_width - new_size[0]) / 2), 
        int((resize_height - new_size[1]) / 2)))
    dx = (float(resize_width - new_size[0]) / 2) / float(resize_width)
    dy = (float(resize_height - new_size[1]) / 2) / float(resize_height)

    x_scale = float(new_size[0]) / float(resize_width)
    y_scale = float(new_size[1]) / float(resize_height)

    return new_im, dx, dy, x_scale, y_scale

def main():
    graph_def = tf.GraphDef()
    model_width = 224
    model_height = 224
    num_class = 4
    num_object = 5
    interpreter = tf.contrib.lite.Interpreter(
        model_path='gx_v2_detector_quant.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    gray_img = Image.open('GX10.jpg')
    origin_img = Image.new('RGB', gray_img.size)
    origin_img.paste(gray_img)
    origin_width, origin_height= origin_img.size
    print(origin_img.size)
    raw_img, dx, dy, x_scale, y_scale = resize_image_keep_ratio(
        origin_img, (model_height, model_width))
    img = np.asarray(raw_img)
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, 0))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    s, z = output_details[0]['quantization']
    r = s*(output_data.astype(np.float32)-z)
    print(r.shape)
    draw = ImageDraw.Draw(origin_img)
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/freefont/FreeMono.ttf', 34)
    total_anchors=[]
    for i in range(0, 2 * num_object, 2):
        total_anchors.append(anchors[i] * 7 / model_width)
        total_anchors.append(anchors[i + 1] * 7 / model_height)
    print(total_anchors)

    image_index = 0
    for b in range(r.shape[0]):
        boxes = boxes_in_batch(
            r[b,:,:,:], image_index, num_object, num_class, total_anchors)
        if boxes.shape[0] > 0:
            nms_box =util.nms(boxes, num_class, has_class_id=True)
            for box1 in nms_box:
                if box1[5] < 0.1:
                    continue
                box1 = np.asarray(box1)
                box=box1[0: 4]
                box[0] = int((box[0] - dx) / x_scale * origin_width)
                box[1] = int((box[1] - dy) / y_scale * origin_height)
                box[2] = int((box[2] - dx) / x_scale * origin_width)
                box[3] = int((box[3] - dy) / y_scale * origin_height)
                #box = interset(box,[0.0,0.0, origin_width,origin_height])
                print(box, box1[5])
                draw.text((box[0], box[1]), 
                    class_names[int(box1[4])] + str(box1[5]) ,(255), font)
                draw.line((box[0],box[1], box[2], box[1]), fill=128, width=6)
                draw.line((box[0],box[3], box[2], box[3]), fill=128, width=6)
                draw.line((box[0],box[1], box[0], box[3]), fill=128, width=6)
                draw.line((box[2],box[1], box[2], box[3]), fill=128, width=6)
            origin_img.save('test4.jpg')

if __name__ == '__main__':
    main()
