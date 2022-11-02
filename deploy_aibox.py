#-*- coding:utf-8 -*-
from __future__ import division
import tensorflow as tf
from matplotlib import gridspec
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
from PIL import Image
#from scipy import misc
import cv2
import os
import ast
import math
from config import config 
from skimage.feature import peak_local_max
from scipy.spatial.distance import pdist, squareform
import time
#from PIL import ImageDraw
#from PIL import ImageFont
from dataset import get_data_from_database
import shutil
import util
import copy
from math import pi
from numpy import cos, sin
import time

def sigmoid_fun(x):
  return 1 / (1.0 + np.exp(-x))

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

def rotation_img(img, degree):
    w = img.shape[1]
    h = img.shape[0]
    temp_length = math.sqrt(w*w+h*h) + 10
    temp_x = int((temp_length - w)/2.0+0.5)
    temp_y = int((temp_length - h)/2.0+0.5)
    temp_length = int(temp_length+0.5)
    temp_img = np.zeros((temp_length,temp_length,3), dtype=np.uint8)
    temp_img[temp_y:temp_y+h,temp_x:temp_x+w,:] = img
    dst_sz = (temp_length,temp_length)
    rotation_center = (temp_length/2.0, temp_length/2.0)
    rataion_matrix = cv2.getRotationMatrix2D(rotation_center, degree, 1.0)
    rot_img = cv2.warpAffine(temp_img, rataion_matrix, dst_sz)
    rot_mat_list=[]
    rot_mat_list.append(rataion_matrix[0,0])
    rot_mat_list.append(rataion_matrix[0,1])
    rot_mat_list.append(rataion_matrix[0,0] * temp_x + rataion_matrix[0,1] * temp_y + rataion_matrix[0,2])
    rot_mat_list.append(rataion_matrix[1,0])
    rot_mat_list.append(rataion_matrix[1,1])
    rot_mat_list.append(rataion_matrix[1,0] * temp_x + rataion_matrix[1,1] * temp_y + rataion_matrix[1,2])
    return rot_img, rot_mat_list

#def undistort_fisheye_points(pts, fisheyeParameter=np.array([693.1535889533948/1.5, 697.6762323982881/1.5,
#  1014.5700737950523/1.5, 549.2067202024954/1.5, 1.0120113034987892])): #760.16,763.28,946.14,551.73,1.0171363
def undistort_fisheye_points(pts, fisheyeParameter=np.array([467.3418184973979 * 512 / 1280,
 466.9720339209149 * 512 / 720, 651.5441842425998 * 512 / 1280, 371.1062915055246 * 512 / 720, 0.995537338859336])): 
    fu = fisheyeParameter[0]
    fv = fisheyeParameter[1]
    cu = fisheyeParameter[2]
    cv =  fisheyeParameter[3]
    omega = fisheyeParameter[4]
    v_prime = (pts[:,1] - cv)/fv
    u_prime = (pts[:,0] - cu)/fu
    r_prime = np.sqrt(u_prime*u_prime + v_prime*v_prime)
    r = np.tan(r_prime*omega)/(2*np.tan(0.5*omega))
    u = u_prime*r*fu/r_prime + cu
    v = v_prime*r*fv/r_prime + cv
    return np.stack([u,v]).T

#def distort_fisheye_points(pts, fisheyeParams=np.array([693.1535889533948/1.5, 697.6762323982881/1.5,
#  1014.5700737950523/1.5, 549.2067202024954/1.5, 1.0120113034987892])): #760.16,763.28,946.14,551.73,1.0171363
def distort_fisheye_points(pts, fisheyeParams=np.array([467.3418184973979 * 512 / 1280,
 466.9720339209149 * 512 / 720, 651.5441842425998 * 512 / 1280, 371.1062915055246 * 512 / 720, 0.995537338859336])): 
    fu = fisheyeParams[0]
    fv = fisheyeParams[1]
    cu = fisheyeParams[2]
    cv = fisheyeParams[3]
    omega = fisheyeParams[4]
    v_prime = (pts[:,1] - cv)/fv
    u_prime = (pts[:,0] - cu)/fu    
    r_prime = np.sqrt(u_prime*u_prime + v_prime*v_prime)
    r = np.arctan(2*r_prime*np.tan(0.5*omega))/omega
    u = u_prime*r*fu/r_prime + cu
    v = v_prime*r*fv/r_prime + cv
    return np.stack([u,v]).T

def transform_pts_to_distance(pts, bboxes, valid_dis):
    distance = []
    new_bboxes = []
    new_pts = []
    for i in range(pts.shape[0]):
        dis = math.sqrt(pow(pts[i][0], 2) + pow(pts[i][1], 2))
        if ids <= valid_dis:
            distance.append(dis)
            new_pts.append(pts[i])
            new_bboxes.append(bboxes[i])
    return distance, new_bboxes

def transform_pts_to_birdview(pts, H, fisheyeParams):
    pts = np.array(pts).reshape(-1, 2)
    pts_undist = undistort_fisheye_points(pts, fisheyeParams)
    pts_birdview = cv2.perspectiveTransform(pts_undist.reshape(-1,1,2), H)
    return pts_birdview.reshape(-1,2)

'''
def transform_pts_to_birdview(pts, H):
    pts = pts.reshape(-1, 2)
    pts_undist = undistort_fisheye_points(pts)
    pts_undist = np.split(pts_undist, 2, axis=1)
    pts_undist_x = np.squeeze(pts_undist[0], axis=1)
    pts_undist_y = np.squeeze(pts_undist[1], axis=1)
    src = np.stack([pts_undist_x, pts_undist_y, np.ones_like(pts_undist_x)], axis=0) #(3,?)
    dst = np.matmul(H, src) #(3, ?)
    tmp = np.split(dst, [2,3], axis=0)
    xy = tmp[0]
    scale = tmp[1]
    scale = np.tile(scale, [2,1]) #(2, ?)
    pts_birdview = np.divide(xy, scale)#(2, ?)
    #pts_birdview.reshape(-1,2)
    return pts_birdview.transpose()
'''

def transform_pts_to_fishview(pts, H, distortion=True, fisheye_distor_paras=None):
    pts = pts.reshape(-1, 2)
    pts = np.split(pts, 2, axis=1)
    pts_x = np.squeeze(pts[0], axis=1)
    pts_y = np.squeeze(pts[1], axis=1)
    src = np.stack([pts_x, pts_y, np.ones_like(pts_x)], axis=0) #(3,?)
    dst = np.matmul(H, src) #(3, ?)
    tmp = np.split(dst, [2,3], axis=0)
    xy = tmp[0]
    scale = tmp[1]
    scale = np.tile(scale, [2,1]) #(2, ?)
    pts_fishview = np.divide(xy, scale)#(2, ?)

    pts_t = pts_fishview.transpose()
    shape_t = pts_t.shape
    if distortion:
        distort_pts = distort_fisheye_points(pts_t, fisheyeParams=fisheye_distor_paras)
        return distort_pts
    else:
        return pts_t

'''
def create_obsmap(img):
    # left to right -200, 200, near to far 0 600, H added the offset of 200 cm
    H = np.array([[-0.83975343,-0.02348891, 1035.4847692],
                  [ 0.83168454, 0.82410012,-240.00806555],
                  [ 0.00424073,-0.00020651, 1.          ]], dtype='float32')
    ori_img = np.zeros([1080,1920])
    ori_cropped = cv2.resize(img, (1320,1080))
    ori_img[:,0:1320] = ori_cropped
    # missing undistortion part
    bird_view = cv2.warpPerspective(ori_img, H, (600, 400))
    bird_view_rot = cv2.rotate(bird_view, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bird_view_rot

def create_obsmap_hole(img, bird_h, bird_w):
    ori_cropped = cv2.resize(img, (1320,1080))
    H = np.array([[5.0804817,   -0.18302056, -35.48475061],
    [2.10390302,   0.76214727,  59.99194269],
    [0.00424073, -0.00020651,   1.        ]], dtype='float32')
    bird_view = cv2.warpPerspective(ori_cropped, H, (bird_w, bird_h))
    #bird_view_rot = cv2.rotate(bird_view, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bird_view_rot = np.rot90(bird_view,-1).copy()
    return bird_view_rot

def create_obsmap_new(img, ud_map, bird_h, bird_w):
    ss_map = np.zeros((bird_h, bird_w), dtype=img.dtype)
    for xx in range(img.shape[1]):
        for yy in range(img.shape[0]):
            u_pt = ud_map[img.shape[1]*yy + xx]
            if u_pt[1] >= 0 and u_pt[1] < bird_h and u_pt[0] >= 0 and u_pt[0] < bird_w:
                ss_map[u_pt[1], u_pt[0]] = img[yy, xx]
    #bird_view_rot = cv2.rotate(ss_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bird_view_rot = np.rot90(ss_map,-1).copy()
    return bird_view_rot
'''

def create_dense_obsmap(img, birdview_to_fishview_map, bird_h, bird_w, colormap):
    ss_map = np.zeros((bird_h, bird_w, 3), dtype=img.dtype)
    for xx in range(bird_w):
        for yy in range(bird_h):
            pt = birdview_to_fishview_map[yy, xx]
            x=int(pt[0])
            y=int(pt[1])
            if x >= 0 and y >= 0 and y < img.shape[0] and x < img.shape[1]:
                ss_map[yy, xx] = colormap[img[y, x]]
    #bird_view_rot = cv2.rotate(ss_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #bird_view_rot = np.rot90(ss_map,-1).copy()
    return ss_map

'''
def create_obsmap_ori_img(img, ud_map, bird_h, bird_w):
    ss_map = np.zeros((bird_h, bird_w, 3), dtype=img.dtype)
    valid_num=0
    for xx in range(img.shape[1]):
        for yy in range(img.shape[0]):
            u_pt = ud_map[img.shape[1]*yy + xx]
            if u_pt[1] >= 0 and u_pt[1] < bird_h and u_pt[0] >= 0 and u_pt[0] < bird_w:
                ss_map[u_pt[1], u_pt[0]] = img[yy, xx]
                valid_num += 1
    #bird_view_rot = cv2.rotate(ss_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print('valid_num')
    print(valid_num, 512*512)
    return ss_map
'''

def generate_birdview_to_fishview_map(birdview_min_x, birdview_min_y,
    birdview_max_x, birdview_max_y, H_birdview_to_fisheye, fisheye_distor_paras):
    birdview_map = np.zeros((birdview_max_y-birdview_min_y, birdview_max_x-birdview_min_x, 2), dtype=np.float32)
    birdview_w = birdview_max_x - birdview_min_x
    birdview_h = birdview_max_y - birdview_min_y
    for x in range(birdview_w):
        for y in range(birdview_h):
            birdview_map[y,x,0] = x + birdview_min_x
            birdview_map[y,x,1] = y + birdview_min_y
    return transform_pts_to_fishview(birdview_map.reshape(-1,2), H_birdview_to_fisheye, True, fisheye_distor_paras)

def read_H_matrix(filename):
    filein = open(filename, 'r')
    pt_str = filein.read()
    fisheye_to_birdview_H = np.array(eval(pt_str))
    return fisheye_to_birdview_H
    
def create_mask(front_mask_border_pt_idx, is_color, record_img_width=512,record_img_height=512):
    front_mask_border_pt = []
    for idx_t in range(len(front_mask_border_pt_idx)):
        temp_pt = []
        temp_pt.append(front_mask_border_pt_idx[idx_t] % record_img_width)
        temp_pt.append(front_mask_border_pt_idx[idx_t] // record_img_width)
        front_mask_border_pt.append(temp_pt)
    front_mask_border_pt = np.array(front_mask_border_pt)
    front_mask_border_pt = front_mask_border_pt.reshape((1, -1, 2))
    front_mask = np.zeros((record_img_height, record_img_width), np.uint8)
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

def read_basetype(basetype_path):
    # if not os.path.exists(basetype_path):
    #     print("basetype_path does not exist...", basetype_path)
    #     basetype_id = 1004
    # else:
    #     f_tmp_in = open(basetype_path, 'r')
    #     basetype_info_paras = f_tmp_in.readline()
    #     f_tmp_in.close()
    #     basetype_id = int(basetype_info_paras)
    #     print("basetype_id ", basetype_id)
    # if basetype_id != 1004 and basetype_id != 1005 and basetype_id != 1006:
    #     print('basetype_id err: ', basetype_id)
    #     input()
    return 1005 #basetype_id

def calculate_fov(fx, fy, cx, cy, distortion, w, h):
    undistort_pts = undistort_fisheye_points(np.array([[cx, 0], [cx, h-1]]),
        fisheyeParameter=np.array([fx, fy, cx, cy, distortion]))
    #h_fov = math.atan2(cx - undistort_pts[0, 0], fx) + math.atan2(undistort_pts[1, 0] - cx, fx)
    v_fov = math.atan2(cy - undistort_pts[0, 1], fy) + math.atan2(undistort_pts[1, 1] - cy, fy)
    return v_fov/math.pi*180

def is_large_fov(cam_info_path, large_v_fov, small_v_fov, img_shape):
    cam_info_exist = False
    if not os.path.exists(cam_info_path):
        print("cam_info_path does not exist...", cam_info_path)
        input()
        cam_info_exist = False
    else:
        cam_info_exist = True
        f_tp_in = open(cam_info_path, 'r')
        cam_paras = f_tp_in.readline()
        f_tp_in.close()
        cam_paras = cam_paras.split(',')
        if len(cam_paras) != 9:
            print('camerainfo.txt format err...')
            input()
        fx = float(cam_paras[0])
        fy = float(cam_paras[1])
        cx = float(cam_paras[2])
        cy = float(cam_paras[3])
        distortion = float(cam_paras[4])
    fisheye_distor_paras = None
    large_fov = False
    if cam_info_exist:
        if fx < 1:
            print(fx, fy, cx, cy, distortion, img_shape)
            input()
            return False, np.zeros((5,), np.float32)

        v_fov = calculate_fov(fx=fx, fy=fy, cx=cx, cy=cy, distortion=distortion, w=img_shape[1], h=img_shape[0])
        if abs(v_fov - small_v_fov) > 7 and abs(v_fov - large_v_fov) > 10:
            print('abs(v_fov - small_v_fov) ', abs(v_fov - small_v_fov), abs(v_fov - large_v_fov),
             v_fov, fx, fy, cx, cy, distortion, img_shape)
            # input()
        
        if abs(v_fov - small_v_fov) > abs(v_fov - large_v_fov):
            large_fov = True
        else:
            large_fov = False

        fisheye_distor_paras = np.array([fx, fy, cx, cy, distortion])
    else:
        large_fov = False
        print("cam_info_path does not exist...", cam_info_path)
        input()
    return large_fov, fisheye_distor_paras

def generate_min_distance_mat(coord, img_w, img_h):
    min_distance_mat = np.zeros((coord.shape[0],coord.shape[0]))
    coord_lst = list(coord)
    for i in range(len(coord_lst)):
        for j in range(len(coord_lst)):
            if i == j:
                continue
            dis_i = img_w - coord_lst[i][1]
            dis_j = img_w - coord_lst[j][1]
            avg_y = (dis_i + dis_j) / 2.0
            if avg_y < 56.8861:
                min_distance_mat[i][j] = 10
            else:
                min_distance_mat[i][j] = 10#0.439 * avg_y - 14.973
    return min_distance_mat
            
def coord_filter(coord, coord_conf, img_w, img_h):
    '''
    The coorresponding 'min_person_dis' is obtained based on the distance of the t60 
    to filter invalid response points.

    Attention:
        The param 'min_distance' of function 'peak_local_max()' should be the min output value
    of this function.
    '''
    # coord_order = np.array(coord_conf).argsort()[::-1]
    # coord = coord[coord_order]
    # coord_conf = np.array(coord_conf)[coord_order]
    dis = squareform(pdist(coord))
    min_dis = generate_min_distance_mat(coord,img_w, img_h) 

    subtract_dis = dis - min_dis

    invalid_coord = np.argwhere(subtract_dis<0)
    invalid_ind = []
    for i in range(len(invalid_coord)):
        if invalid_coord[i][0] < invalid_coord[i][1]:
            invalid_ind.append(invalid_coord[i][1])
    res = []
    res_conf = []
    for j in range(len(coord)):
        if j in invalid_ind:
            continue
        res.append(coord[j])
        res_conf.append(list(coord_conf)[j])

    return np.array(res), np.array(res_conf)

def human_single_limited(pts, value, w, h):
    # 所有点互相求距离，留下和剩下所有点距离大于0.8*max(h,w)的点，或者值大于其它所有点的点
    # 同nms
    index = value.argsort()[::-1]
    # # limited value
    # peak_val = value[index[0]]
    # idx = np.where(value > peak_val * 0.8)[0]
    # pts = pts[idx]

    # limited distance
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i) # save index of pts
        dis = np.sqrt(np.sum((pts[i] - pts[index[1:]])**2))
        idx = np.where(dis > max(w,h)*0.7)[0] # max(w,h) / num of coord
        index = index[idx + 1]
    return pts[keep]

def distance(pts1, pts2):
    return math.sqrt((pts1[0] - pts2[0]) ** 2 + (pts1[1] - pts2[1]) ** 2)

def roi_compare(sidewalk_area, road_area, thresh):
    if sidewalk_area.shape != road_area.shape:
        print("roi shape is error.")
        input()
    road_area_sum = road_area.sum()
    sidewalk_area_sum = sidewalk_area.sum()
    _sum = sidewalk_area.shape[0] * sidewalk_area.shape[1]
    sidewalk_flag = sidewalk_area_sum > road_area_sum
    valid_flag = sidewalk_area_sum / _sum > thresh if sidewalk_flag else road_area_sum / _sum > thresh
    return sidewalk_flag, valid_flag 

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

def make_square(oring_im, desired_size=608, fill_color=(0, 0, 0)):
    old_size = [oring_im.shape[1],oring_im.shape[0]]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = cv2.resize(oring_im, new_size)
    #new_im = Image.new('RGB', (desired_size, desired_size), fill_color)
    #new_im.paste(im, 
    #    (int((desired_size - new_size[0]) / 2), 
    #    int((desired_size - new_size[1]) / 2)))
    dx = (desired_size - new_size[0]) / 2
    dy = (desired_size - new_size[1]) / 2
    new_im = cv2.copyMakeBorder(im, int(dy), int(dy), int(dx), int(dx), cv2.BORDER_CONSTANT, value=0)
    dx = dx / desired_size
    dy = dy / desired_size
    
    x_scale = new_size[0] / desired_size
    y_scale = new_size[1] / desired_size
    print(dx,dy,x_scale,y_scale,new_im.shape)
    return new_im, dx, dy, x_scale, y_scale

def interset(box1, box2):
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    return [left, top, right, bottom]

def frame_coverage(seg_roi, fbox):
    tmp_roi = seg_roi.copy()
    # tmp_color = cv2.merge([tmp_roi, np.zeros(tmp_roi.shape), tmp_roi])
    # box : [x1, y1, x2, y2]
    side_len = (fbox[2] - fbox[0] + fbox[3] - fbox[1]) * 2
    # horizontal
    h_len = 0
    for h in range(fbox[0], fbox[2] + 1):
        if seg_roi[fbox[1], h] == 1:
            h_len += 1
            # tmp_color = cv2.circle(
            #     tmp_color,
            #     (fbox[1], h),
            #     1,
            #     (0,0,255)
            # )
        if seg_roi[fbox[3], h] == 1:
            h_len += 1
            # tmp_color = cv2.circle(
            #     tmp_color,
            #     (fbox[3], h),
            #     1,
            #     (0,0,255)
            # )
    # vertical
    v_len = 0
    for v in range(fbox[1] + 1, fbox[3]):
        if seg_roi[v, fbox[0]] == 1:
            v_len += 1
            # tmp_color = cv2.circle(
            #     tmp_color,
            #     (v, fbox[0]),
            #     1,
            #     (0,0,255)
            # )
        if seg_roi[v, fbox[2]] == 1:
            v_len += 1
    #         tmp_color = cv2.circle(
    #             tmp_color,
    #             (v, fbox[2]),
    #             1,
    #             (0,0,255)
    #         )
    # cv2.imwrite("temp.jpg", tmp_color)
    return float(h_len + v_len) / side_len 

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
    if get_database_data:
        img_dict_from_database_start = get_data_from_database.request_files(custom_root_dir=custom_database_folder,
         data_family=data_family, time_filter=database_data_time_filter_start)
        img_dict_from_database_end = get_data_from_database.request_files(custom_root_dir=custom_database_folder,
         data_family=data_family, time_filter=database_data_time_filter_end)
       
        data_path_list = []
        end_img_lst = [list(dic.keys())[0] for dic in img_dict_from_database_end]
        for sid, start in enumerate(img_dict_from_database_start):
            print('select date rate:', float(sid)/len(img_dict_from_database_start))
            img_pth = list(start.keys())[0]
            if img_pth in end_img_lst:
                continue
            data_path_list.append(start)

        print('get database data size ', len(data_path_list))
        for tp_dict_name in data_path_list:
            tp_name = list(tp_dict_name.keys())[0]
            tp_l = tp_name[ : tp_name.find('/fisheye')+len('fisheye/')]
            tp_ll = tp_name[tp_name.find('/fisheye')+len('fisheye/')+1: ]
            if tp_l not in test_img_folder_list:
                test_img_folder_list.append(tp_l)
                database_data_dict[tp_l] = [tp_ll]
            else:
                database_data_dict[tp_l].append(tp_ll)
    else:
        tmp_list = os.listdir(test_img_root_folder)
        for dir_t in tmp_list:
            if os.path.isdir(test_img_root_folder + '/' + dir_t):
                test_img_folder_list.append(test_img_root_folder + '/' + dir_t + '/fisheye')


    # Parameter prepare
    seg_button = conf['seg_button']
    pen_button = conf['pen_button']
    deploy_mode = conf['deploy_mode']
    net_input_h = conf['net_input_h']
    net_input_w = conf['net_input_w']
    normalize_flag = conf['normalize']
    output_dir_root = conf['output_dir']
    R_MEAN = conf['r_mean']
    G_MEAN = conf['g_mean']
    B_MEAN = conf['b_mean']
    SCALE = conf['standard_deviation_reciprocal']
    front_mask_border_pt_idx = conf['front_mask_border_pt_idx']
    front_mask_border_pts_large_fov = conf['front_mask_border_pts_large_fov']
    front_mask_border_pts_large_fov_1005 = conf['front_mask_border_pts_large_fov_1005']
    front_mask_border_pts_large_fov_1007 = conf['front_mask_border_pts_large_fov_1007']
    front_mask_border_pts_large_fov_2000 = conf['front_mask_border_pts_large_fov_2000']

    front_mask_border_pt_idx_post_process = conf['front_mask_border_pt_idx_post_process']
    front_mask_border_pts_post_process_large_fov = conf['front_mask_border_pts_post_process_large_fov']
    front_mask_border_pts_post_process_large_fov_1005 = conf['front_mask_border_pts_post_process_large_fov_1005']
    front_mask_border_pts_post_process_large_fov_1007 = conf['front_mask_border_pts_post_process_large_fov_1007']
    front_mask_border_pts_post_process_large_fov_2000 = conf['front_mask_border_pts_post_process_large_fov_2000']

    # Seg param init
    class_num = conf['class_num']
    # Pedestrian param init
    num_object = conf['num_object'] 
    num_class = conf['classes']
    anchors = conf['anchors']
    h_lite = conf['h_lite']
    distor_lite = conf['distor_lite']
    conf_thresh = conf['conf_thresh']
    class_thresh = conf['class_thresh']
    nms_thresh = conf['nms_thresh']
    filter_scale = conf['filter_scale']
    dis_debug = conf['pedes_dis_debug']
    conf_debug = conf['pedes_conf_debug']
    no_filter_debug = conf['no_filter_debug']
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

    # create mask
    record_img_width = net_input_w
    record_img_height = net_input_h
    
    color_front_mask = create_mask(front_mask_border_pt_idx, is_color=True,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask = create_mask(front_mask_border_pt_idx_post_process, is_color=False,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask //= 255

    color_front_mask_large_fov = create_mask(front_mask_border_pts_large_fov, is_color=True,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask_large_fov = create_mask(front_mask_border_pts_post_process_large_fov, is_color=False,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask_large_fov //= 255
    
    color_front_mask_large_fov_1005 = create_mask(front_mask_border_pts_large_fov_1005, is_color=True,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask_large_fov_1005 = create_mask(front_mask_border_pts_post_process_large_fov_1005, is_color=False,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask_large_fov_1005 //= 255

    color_front_mask_large_fov_1007 = create_mask(front_mask_border_pts_large_fov_1007, is_color=True,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask_large_fov_1007 = create_mask(front_mask_border_pts_post_process_large_fov_1007, is_color=False,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask_large_fov_1007 //= 255

    color_front_mask_large_fov_2000 = create_mask(front_mask_border_pts_large_fov_2000, is_color=True,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask_large_fov_2000 = create_mask(front_mask_border_pts_post_process_large_fov_2000, is_color=False,
    record_img_width=record_img_width, record_img_height=record_img_height)
    front_mask_large_fov_2000 //= 255

    # ================================================  Deploy ====================================================
    
    # Segmentation Graph
    if seg_button:
        graph_sidewalk = load_graph(conf['frozen_model_sidewalk_filename'])
        img_sidewalk = graph_sidewalk.get_tensor_by_name(u'prefix/input:0')
        y_sidewalk = graph_sidewalk.get_tensor_by_name(u'prefix/add_on_mobilenet_v1/road_heatmap_argmax:0')
        graph_road = load_graph(conf['frozen_model_road_filename'])
        img_road = graph_road.get_tensor_by_name(u'prefix/input:0')
        y_road = graph_road.get_tensor_by_name(u'prefix/add_on_mobilenet_v1/road_heatmap_argmax:0')

        sess_sidewalk = tf.Session(graph=graph_sidewalk)
        sess_road = tf.Session(graph=graph_road)
        

    # Pedestrian Graph
    if pen_button:
        graph_pen = load_graph(conf['frozen_model_pen_filename'])
        img_pen = graph_pen.get_tensor_by_name(u'prefix/Placeholder:0')
        y_pen_13 = graph_pen.get_tensor_by_name(u'prefix/head/reg13x13_output/BiasAdd:0')
        y_pen_26 = graph_pen.get_tensor_by_name(u'prefix/head/reg26x26/BiasAdd:0')
        y_pen_52 = graph_pen.get_tensor_by_name(u'prefix/head/reg52x52/BiasAdd:0')

        sess_pen = tf.Session(graph=graph_pen)

    total_valid_img_num = 0
    test_folder_num = len(test_img_folder_list)
    for test_folder_idx, test_img_folder in enumerate(test_img_folder_list):
        basetype = read_basetype(test_img_folder + '/robot_base_type.txt')
        cam_info_path = test_img_folder + '/camrainfo.txt'
        if get_database_data:
            file_list = database_data_dict[test_img_folder]
        else:
            file_list = os.listdir(test_img_folder)
        print('find files: ', len(file_list))
        img_num = 0
        end_str = '.jpg'

        print(test_img_folder)
        output_dir = output_dir_root + '/' + test_img_folder.split('/')[-2]
        print(output_dir)
        original_img_h = 720
        original_img_w = 1280
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            if os.listdir(output_dir):
                print('output_dir is not empty, ', output_dir)
                input()
        
        for filename in file_list:
            _, file_type = os.path.splitext(test_img_folder + '/' + filename)
            if (file_type == ".jpg" or file_type == ".png") and filename.find('mask') < 0:
                img_num += 1
                if img_num == 1:
                    sp = cv2.imread(test_img_folder + '/' + filename).shape
                    original_img_w = sp[1]
                    original_img_h = sp[0]

        resize_x = original_img_w / net_input_w #crop_x
        resize_y = original_img_h / net_input_h

        cur_img_idx = 0
        time_model_run = 0.0
        for filename in file_list:
            img_path = os.path.join(test_img_folder, filename)
            # # Debug
            # if not img_path.find("1617165842963076"):
            #     continue
            file_basepath, file_type = os.path.splitext(img_path)
            if file_type != end_str or filename.find('mask') >= 0: 
                continue
            print(img_path)
            cur_img_idx += 1
            print('*******************************Current folder Processing: ', cur_img_idx / img_num,
                '------ folder: ', (test_folder_idx+1) / test_folder_num)
            H_path = file_basepath + '-H.txt'
            if deploy_mode == 't60lite':
                H_mat = np.array(h_lite).reshape((3,3))
            elif deploy_mode == 't60':
                if not os.path.exists(H_path):
                    print("no : ", H_path)
                    continue
                else:
                    H_mat = read_H_matrix(H_path).reshape((3,3))
            else:
                print("Error Deploy Mode, please check out.")
                continue
            # Calculate inverse matrix to get real distance. 
            try:
                H_mat_inv = np.linalg.inv(H_mat)
            except np.linalg.LinAlgError:
                print("H mat inv solving failed..")
                continue
            
            # Input image is placed horizontally. (H < W)
            np_array = cv2.imread(img_path)
            # ori_show_img = np.rot90(np_array)
            ori_show_img = np_array
            show_img = cv2.resize(np_array, (net_input_w, net_input_h)).copy()
            small_v_fov = 79
            large_v_fov = 100
            large_fov, fisheye_distor_paras = is_large_fov(cam_info_path, large_v_fov, small_v_fov, np_array.shape)
            if deploy_mode =='t60lite':
                fisheye_distor_paras = distor_lite.copy()
            fisheye_distor_paras[0] /= resize_x
            fisheye_distor_paras[1] /= resize_y
            fisheye_distor_paras[2] /= resize_x
            fisheye_distor_paras[3] /= resize_y
            shape_t = np_array.shape
            crop_x = shape_t[1]
            np_array = np_array[0:shape_t[0], 0:crop_x].copy()

            batch_size = 1 #16
            target_size = (net_input_w, net_input_h)
            if seg_button:
                # no need cropping for segmentation
                np_array_nocrop = cv2.resize(np_array, target_size)
                need_flip = False
                if need_flip:
                    np_array = cv2.flip(np_array, 1)
            
                if normalize_flag:
                    np_array_nocrop = (np_array_nocrop - [[[B_MEAN, G_MEAN, R_MEAN]]])*SCALE
                    #print('current pattern, normalized...')
                else:
                    print('current pattern, not normalized, press enter to confirm.')
                    input()
                
                np_array_nocrop = np_array_nocrop.astype(np.float32)
                image_in_nocrop = np_array_nocrop.reshape([-1,net_input_h,net_input_w,3])
                image_in_nocrop = [image_in_nocrop] * batch_size
                image_in_nocrop = np.concatenate(image_in_nocrop, axis=0)
                
            if pen_button:
                # Calculate handle location with resolution 512*512
                if large_fov:
                    if basetype < 1005: front_mask_pts = front_mask_border_pts_large_fov
                    elif basetype < 1007:front_mask_pts = front_mask_border_pts_large_fov_1005
                    elif basetype < 2000:front_mask_pts = front_mask_border_pts_large_fov_1007
                    else: front_mask_pts = front_mask_border_pts_large_fov_2000
                else:
                    front_mask_pts = front_mask_border_pt_idx
                    
                # with original shape
                # handle_top_edge = front_mask_pts[1] % net_input_w * original_img_w / net_input_w
                handle_top_edge = int(front_mask_pts[3] % net_input_w * original_img_w / net_input_w)
                
                # with net_input shape
                roi_x = front_mask_pts[3] % net_input_w 
                roi_y = front_mask_pts[3] // net_input_w
                roi_h = front_mask_pts[4] // net_input_w - roi_y
                roi_w = int(roi_h * 0.5)
                roi_x_ori = roi_x * original_img_w // net_input_w 
                roi_y_ori = roi_y * original_img_h // net_input_h
                roi_h_ori = roi_h * original_img_h // net_input_h 
                roi_w_ori = int(roi_h_ori * 0.5)

                # image_crop for pedestrian detection
                # np_array_crop = np_array[:, handle_top_edge:, :]
                np_array_crop = np_array
                np_array_rotated = np.rot90(np_array_crop)
                cropped_height, cropped_width, _ = np_array_rotated.shape
                np_array_resized = cv2.resize(np_array_rotated, target_size)
                raw_img, dx, dy, x_scale, y_scale = make_square(np_array_resized, target_size[0])
                # np_array_crop = np.asarray(raw_img).astype(np.float32).copy()
                
                #--llx --20220319
                np_array_crop = cv2.resize(np_array_crop, target_size)
                np_array_crop = np.asarray(np_array_crop).astype(np.float32).copy()

                # image_in_crop = np_array_crop.reshape([-1,net_input_h,net_input_w,3])
                # image_in_crop = [image_in_crop] * batch_size
                # image_in_crop = np.concatenate(image_in_crop, axis=0)

            

            # For Segmentation
            sidewalk_flag = False # use for filter pedestrian (init)
            # mask_filter_pen = np.zeros((original_img_w, original_img_h), dtype = np.uint8)
            seg_choose = np.zeros((net_input_w, net_input_h), dtype = np.uint8)
            if seg_button:
                y_sidewalk_pred = sess_sidewalk.run([y_sidewalk], feed_dict={img_sidewalk: image_in_nocrop})
                y_road_pred = sess_road.run([y_road], feed_dict={img_road: image_in_nocrop})

                y_out_sidewalk = y_sidewalk_pred[0]
                y_out_sidewalk = np.reshape(y_out_sidewalk, (net_input_h, net_input_w))
                
                if np.max(y_out_sidewalk) >= class_num:
                    print('np.max(y_out_sidewalk) >= class_num', np.max(y_out_sidewalk))
                    input()
                
                y_out_road = y_road_pred[0]
                y_out_road = np.reshape(y_out_road, (net_input_h, net_input_w))
                
                if np.max(y_out_road) >= class_num:
                    print('np.max(y_out_road) >= class_num', np.max(y_out_road))
                    input()

                colormap_sidewalk = np.array([[0, 0, 0], [255, 0, 255], [0,255,255]], dtype=y_out_sidewalk.dtype)
                colormap_road = np.array([[0, 0, 0], [0, 255, 0], [0,255,255]], dtype=y_out_road.dtype)
                if large_fov:
                    if basetype < 1005:
                        y_out_sidewalk = np.where(front_mask_large_fov, front_mask_large_fov, y_out_sidewalk)
                        y_out_road = np.where(front_mask_large_fov, front_mask_large_fov, y_out_road)
                    elif basetype < 1007:
                        y_out_sidewalk = np.where(front_mask_large_fov_1005, front_mask_large_fov_1005, y_out_sidewalk)
                        y_out_road = np.where(front_mask_large_fov_1005, front_mask_large_fov_1005, y_out_road)
                    elif basetype < 2000:
                        y_out_sidewalk = np.where(front_mask_large_fov_1007, front_mask_large_fov_1007, y_out_sidewalk)
                        y_out_road = np.where(front_mask_large_fov_1007, front_mask_large_fov_1007, y_out_road)
                    else:
                        y_out_sidewalk = np.where(front_mask_large_fov_2000, front_mask_large_fov_2000, y_out_sidewalk)
                        y_out_road = np.where(front_mask_large_fov_2000, front_mask_large_fov_2000, y_out_road)
                else:
                    y_out_sidewalk = np.where(front_mask, front_mask, y_out_sidewalk)
                    y_out_road = np.where(front_mask, front_mask, y_out_road)

                segmentaion_sidewalk = colormap_sidewalk[y_out_sidewalk].astype('uint8')
                segmentaion_road = colormap_road[y_out_road].astype('uint8')
                # segmentaion_mask = cv2.addWeighted(segmentaion_sidewalk, 0.5, segmentaion_road, 0.5, 0)
                
                # segmentation result
                # segmentation result choice
                sidewalk_flag, valid_flag = roi_compare(y_out_sidewalk[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], 
                                            y_out_road[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy(), 0.8)
                path_text = 'Sidewalk' if sidewalk_flag else 'Road'
                valid_text = 'valid' if valid_flag else 'invalid'
                segmentaion_mask = segmentaion_sidewalk if sidewalk_flag else segmentaion_road
                segmentaion_mask = cv2.resize(segmentaion_mask,  (shape_t[1], shape_t[0]))
                # filter pedestrian
                # y_out_road_original = cv2.resize(y_out_road.astype('uint8'), (original_img_w, original_img_h))
                # mask_filter_pen[:, :] = np.rot90(y_out_road_original).copy()
                seg_choose = y_out_road.astype('uint8')
            # For pedestrian
            if pen_button:
                
                T1 = time.perf_counter()   
                
                cv2.imwrite("./get_require/res.png", np_array_crop)
                      
                predictions1, predictions2, predictions3 = sess_pen.run(
                    [y_pen_13, y_pen_26, y_pen_52], 
                    feed_dict={img_pen: [np_array_crop/255.0]})
                T2 =time.perf_counter()
                time_model_run += ((T2 - T1)*1000)
                print("model run time (ms):", ((T2 - T1)*1000))
  
                pedestrian_res, pedestrian_coord, pedestrian_bboxes, other_coord, \
                    other_bboxes, box_offsets, pedestrian_conf = [], [], [], [], [], [], []
                for b in range(predictions1.shape[0]):
                    # print("predictions1.shape:", predictions1.shape)
                    # print("predictions1.shape:", predictions2.shape)
                    # print("predictions1.shape:", predictions3.shape)
                    # print("predictions1.shape[0]:", predictions1.shape[0])
                    boxes = []
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
                
                for bid in range(len(pedestrian_res)):
                    # back to the resolution of cropped image. (Attention: pedestrian_res has been rotated!)
                    # ptx1 = pedestrian_res[bid][0] * original_img_w
                    # ptx2 = pedestrian_res[bid][2] * original_img_w
                    # pty1 = pedestrian_res[bid][1] * original_img_h
                    # pty2 = pedestrian_res[bid][3] * original_img_h
                    # back to no rotate image coordination.
                    # ptx1_rot = original_img_w - pty2
                    # ptx2_rot = original_img_w - pty1
                    # pty1_rot = ptx1 
                    # pty2_rot = ptx2
                    
                    # resize to net_input_size
                    # ptx1_resized = int(ptx1 * net_input_w / original_img_w)
                    # ptx2_resized = int(ptx2 * net_input_w / original_img_w)
                    # pty1_resized = int(pty1 * net_input_h / original_img_h)
                    # pty2_resized = int(pty2 * net_input_h / original_img_h)
                    ptx1_resized = int(pedestrian_res[bid][0] * original_img_w)
                    ptx2_resized = int(pedestrian_res[bid][2] * original_img_w)
                    pty1_resized = int(pedestrian_res[bid][1] * original_img_h)
                    pty2_resized = int(pedestrian_res[bid][3] * original_img_h)
                    # clip
                    ptx1_resized, pty1_resized, ptx2_resized, pty2_resized = \
                        interset([ptx1_resized, pty1_resized, ptx2_resized, pty2_resized],
                                 [0, 0, net_input_w - 1, net_input_h - 1])
                        
                    sidewalk_flag = False
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
                # pedestrian_coord_to_birdview = []
                # for pid, p_coord in enumerate(pedestrian_coord):
                #     pedestrian_coord_to_birdview.append(p_coord)
                # pedestrian_coord_to_birdview = np.array(pedestrian_coord_to_birdview)
                # birdview_coord_list = []
                # if len(pedestrian_coord) > 0:
                #     birdview_coord = transform_pts_to_birdview(pedestrian_coord_to_birdview, H_mat, fisheye_distor_paras)
                #     birdview_coord_list.append(birdview_coord)
                # else:
                #     birdview_coord_list = pedestrian_coord_to_birdview
                # pts_distance, pedestrian_bboxes = transform_pts_to_distance(birdview_coord, pedestrian_bboxes, 800)

                # Pedestrian Show
                person_count_res = ori_show_img.copy()
                person_count_res_dis = person_count_res.copy()
                person_count_res_conf = person_count_res.copy()
                valid_color, invalid_color = (0, 0, 255), (0, 255, 0)
                for cid, dot_coord in enumerate(pedestrian_coord): # (x,y) in (512, 512)
                    # person_count_res = cv2.circle(
                    #     person_count_res, 
                    #     (dot_coord[0] * original_img_w // net_input_w, 
                    #      dot_coord[1] * original_img_h // net_input_h),
                    #     5, valid_color, -1)
                    # person_count_res = cv2.circle(
                    #     person_count_res,
                    #     ((dot_coord[0] - box_offsets[cid]) * original_img_w // net_input_w,
                    #       dot_coord[1] * original_img_h // net_input_h),
                    #     5, valid_color, -1)
                    # person_count_res = cv2.circle(
                    #     person_count_res,
                    #     ((dot_coord[0] - box_offsets[cid]) * original_img_w // net_input_w,
                    #       dot_coord[1] * original_img_h // net_input_h),
                    #     2, (0, 0, 0), -1)
                    person_count_res = cv2.rectangle(
                        person_count_res,
                        (pedestrian_bboxes[cid][0],
                         pedestrian_bboxes[cid][1]),
                        (pedestrian_bboxes[cid][2], 
                         pedestrian_bboxes[cid][3]),
                        valid_color,
                        2, -1
                    )
                    # if dis_debug:
                    #     # dis visualization
                    #     person_dis = int(math.sqrt(pow(birdview_coord_list[0][cid, 0], 2) + pow(birdview_coord_list[0][cid, 1], 2)))
                    #     person_count_res_dis = cv2.putText(
                    #         person_count_res_dis.copy(),
                    #         #'('+str(int(birdview_coord_list[0][cid, 1]))+','+str(int(birdview_coord_list[0][cid, 0]))+')'+ \
                    #         #'-'+
                    #         str(person_dis)[:4], 
                    #         ((dot_coord[0] - box_offsets[cid]) * original_img_w // net_input_w - 50,
                    #         dot_coord[1] * original_img_h // net_input_h),
                    #         cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255)
                    #     )
                    #     person_count_res_dis = cv2.rectangle(
                    #         person_count_res_dis,
                    #         (pedestrian_bboxes[cid][0] * original_img_w // net_input_w,
                    #         pedestrian_bboxes[cid][1] * original_img_h // net_input_h),
                    #         (pedestrian_bboxes[cid][2] * original_img_w // net_input_w, 
                    #         pedestrian_bboxes[cid][3] * original_img_h // net_input_h),
                    #         valid_color,
                    #         2, -1
                    #     )
                    # if conf_debug:
                    #     # conf visualization
                    #     person_count_res_conf = cv2.putText(
                    #         person_count_res_conf.copy(),  str(pedestrian_conf[cid])[:4], 
                    #         ((dot_coord[0] - box_offsets[cid]) * original_img_w // net_input_w - 50,
                    #         dot_coord[1] * original_img_h // net_input_h),
                    #         cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255))
                    #     person_count_res_conf = cv2.rectangle(
                    #         person_count_res_conf,
                    #         (pedestrian_bboxes[cid][0] * original_img_w // net_input_w,
                    #         pedestrian_bboxes[cid][1] * original_img_h // net_input_h),
                    #         (pedestrian_bboxes[cid][2] * original_img_w // net_input_w, 
                    #         pedestrian_bboxes[cid][3] * original_img_h // net_input_h),
                    #         valid_color,
                    #         2, -1
                    #     )
                    
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

                            
            # 10 meter line(旋转)
            # (x,y)
            ten_meter_pts = coord_circle #np.array([[500, 866], [707, 707], [866, 500], [1000, 0], [866, -500], [707, -707], [500, -866]])
            # np.add.at(ten_meter_pts, [[np.arange(ten_meter_pts.shape[0])], [0]], -1 * handle_top_edge)
            ten_meter_pts_fisheye = transform_pts_to_fishview(ten_meter_pts, H_mat_inv, True, fisheye_distor_paras)
            np.multiply.at(ten_meter_pts_fisheye, [[np.arange(ten_meter_pts_fisheye.shape[0])], [0]], resize_x)
            np.multiply.at(ten_meter_pts_fisheye, [[np.arange(ten_meter_pts_fisheye.shape[0])], [1]], resize_y)
            # for mid in range(1, ten_meter_pts.shape[0]):
            #     person_count_res = cv2.line(person_count_res, 
            #         (int(ten_meter_pts_fisheye[mid-1,1]),
            #          int(person_count_res.shape[0] - ten_meter_pts_fisheye[mid-1,0])), \
            #         (int(ten_meter_pts_fisheye[mid,1]),
            #          int(person_count_res.shape[0] - ten_meter_pts_fisheye[mid,1])), (0,255,255), 2)
            
            # draw the distance line
            # for mid in range(1, ten_meter_pts.shape[0]):
            #     person_count_res = cv2.line(person_count_res, 
            #         (int(ten_meter_pts_fisheye[mid-1,0]),
            #          int(ten_meter_pts_fisheye[mid-1,1])), \
            #         (int(ten_meter_pts_fisheye[mid,0]),
            #          int(ten_meter_pts_fisheye[mid,1])), (0,255,255), 2)
                
            # show
            overlapping = np.zeros((shape_t[1], shape_t[0]), np.uint8)
            if seg_button and pen_button:
                overlapping = cv2.addWeighted(person_count_res, 0.8, segmentaion_mask, 0.2, 0)
                overlapping = cv2.rectangle(overlapping, (roi_x_ori, roi_y_ori), (roi_x_ori+roi_w_ori, roi_y_ori+roi_h_ori), (0,0,0), 2)
                overlapping = cv2.putText(overlapping, path_text, (20,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 2)
                overlapping = cv2.putText(overlapping, valid_text, (20,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 2)
            elif seg_button:
                overlapping = cv2.addWeighted(segmentaion_mask, 0.5, ori_show_img, 0.5, 0)
                overlapping = cv2.rectangle(overlapping, (roi_x_ori, roi_y_ori), (roi_x_ori+roi_w_ori, roi_y_ori+roi_h_ori), (0,0,0), 2)
                overlapping = cv2.putText(overlapping, path_text, (20,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 2)
                overlapping = cv2.putText(overlapping, valid_text, (20,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 2)
            elif pen_button:
                overlapping = person_count_res.copy()
            else:
                print("You need to choose one visualization style.")
            
            if dis_debug and conf_debug:
                overlapping = np.vstack((overlapping, person_count_res_dis, person_count_res_conf))
            elif dis_debug:
                overlapping = np.vstack((overlapping, person_count_res_dis))
            elif conf_debug:
                overlapping = np.vstack((overlapping, person_count_res_conf))
            # cv2.imwrite(output_dir + '/' + filename + '-res.png', np.rot90(overlapping))
            cv2.imwrite(output_dir + '/' + filename + '-res.png', overlapping)
            total_valid_img_num += 1    
        print("mean per image time_model_run: ", time_model_run / img_num)
    if seg_button:
        sess_sidewalk.close()
        sess_road.close()
    if pen_button:
        sess_pen.close()
    print('********************************Total_valid_img_num is ', total_valid_img_num)    
    
    
    
    
    
   


