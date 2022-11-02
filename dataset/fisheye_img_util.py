import numpy as np
import os
import cv2
import math

def create_front_mask(front_mask_border_pt_idx, record_img_width, record_img_height):
  front_rect0 = [
    front_mask_border_pt_idx[0] % record_img_width,
    front_mask_border_pt_idx[1] % record_img_width,
    front_mask_border_pt_idx[0] // record_img_width,
    front_mask_border_pt_idx[7] // record_img_width
  ]
  front_rect1 = [
    front_mask_border_pt_idx[2] % record_img_width,
    front_mask_border_pt_idx[3] % record_img_width,
    front_mask_border_pt_idx[2] // record_img_width,
    front_mask_border_pt_idx[5] // record_img_width
  ]
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
  # cv2.imwrite('./front_mask-gen.jpg', front_mask)
  if np.max(front_mask) != 255:
    print('np.max(front_mask)!=255 ', np.max(front_mask))
    input()
  if np.min(front_mask) != 0:
    print('np.min(front_mask)!=0 ', np.min(front_mask))
    input()
  front_mask = 255 - front_mask
  color_front_mask = np.stack([front_mask, front_mask, front_mask], axis=2)
  return front_mask, color_front_mask, front_rect0, front_rect1

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

def calculate_fov(fx, fy, cx, cy, distortion, w, h):
  undistort_pts = undistort_fisheye_points(np.array([[cx, 0], [cx, h-1]]),
    fisheyeParameter=np.array([fx, fy, cx, cy, distortion]))
  #h_fov = math.atan2(cx - undistort_pts[0, 0], fx) + math.atan2(undistort_pts[1, 0] - cx, fx)
  v_fov = math.atan2(cy - undistort_pts[0, 1], fy) + math.atan2(undistort_pts[1, 1] - cy, fy)
  #print(math.atan2(cx, fx)/math.pi*180, math.atan2(undistort_pts[1, 0]-cx, fx)/math.pi*180)
  #print(math.atan2(cy, fy)/math.pi*180, math.atan2(undistort_pts[3, 1]-cy, fy)/math.pi*180)
  return v_fov/math.pi*180

def get_files(data_folder):
  file_list_pri = os.listdir(data_folder)
  data_path_list = []
  for file_sub in file_list_pri:
    cur_path = data_folder + '/' + file_sub
    if os.path.isdir(cur_path):
      print(cur_path)
      cur_folder = cur_path + '/fisheye'
      file_list = os.listdir(cur_folder)
      data_path_list += data_folder_parse(cur_folder, file_list)
  return data_path_list

def data_folder_parse(cur_folder, file_list):
  small_h_fov = 152
  small_v_fov = 79
  large_h_fov = 210
  large_v_fov = 100

  basetype_info = cur_folder + '/robot_base_type.txt'
  if not os.path.exists(basetype_info):
    print("basetype_info does not exist...", basetype_info)
    basetype_id = 1004
  else:
    f_tmp_in = open(basetype_info, 'r')
    basetype_info_paras = f_tmp_in.readline()
    f_tmp_in.close()
    basetype_id = int(basetype_info_paras)
    print('read basetype_id from txt, it is: ', basetype_id)
    #input()
  if basetype_id not in range(1004, 2001):
    print('basetype_id not in valid value range [1004, 2000], please check if it is wrong: ', basetype_id)
    input()

  cam_info = cur_folder + '/camrainfo.txt'
  cam_info_exist = False
  if not os.path.exists(cam_info):
    print("cam_info does not exist...", cam_info)
    input()
    cam_info_exist = False
  else:
    cam_info_exist = True
    f_tp_in = open(cam_info, 'r')
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

  data_dict_list = []
  file_idx_ijk = 0
  for file_idx, file_path in enumerate(file_list):
    filename, file_type = os.path.splitext(cur_folder + '/' + file_path)
    if file_type != '.json' or (not os.path.exists(filename + '.jpg')):
      continue
    img_path = filename + '.jpg'
    mask_path = filename + '-mask.jpg'
    mask2_path = filename + '-mask_road2.jpg'
    l_roadside_path = filename + '-L_roadside.txt'
    r_roadside_path = filename + '-R_roadside.txt'
    mask_roadside_l_path = filename + '-mask_l.jpg'
    mask_roadside_r_path = filename + '-mask_r.jpg'
    H_matrix_path = filename + '-H.txt'
    person_box_path = filename + '-person-boxes.txt'
    augment_person_box_path = filename + '-person-boxes-augment.txt'
    augment_custom_person_box_path = filename + '-person-boxes-custom-augment.txt'
    distillation_mask_path = filename + '-no_distillation.txt'
    if file_idx_ijk == 0:
      file_idx_ijk += 1
      img_tmp = cv2.imread(img_path)
      large_fov = False
      if cam_info_exist:
        if fx < 1:
          print(fx, fy, cx, cy, distortion, cur_folder, img_tmp.shape)
          input()
        v_fov = calculate_fov(fx=fx, fy=fy, cx=cx, cy=cy, distortion=distortion, w=img_tmp.shape[1], h=img_tmp.shape[0])
        print('v_fov ', v_fov)
        if abs(v_fov - small_v_fov) > 7 and abs(v_fov - large_v_fov) > 13:
          print('abs(v_fov - small_v_fov) ', abs(v_fov - small_v_fov), abs(v_fov - large_v_fov), v_fov, fx, fy, cx, cy, distortion, img_tmp.shape, cur_folder)
          input()
          
        if abs(v_fov - small_v_fov) > abs(v_fov - large_v_fov):
          large_fov = True
        else:
          #print('small fov')
          large_fov = False
      else:
        large_fov = False
        print("cam_info does not exist...", cam_info)
        input()
    data_dict_list.append(
      {img_path : [mask_path, l_roadside_path, r_roadside_path, 
      mask_roadside_l_path, mask_roadside_r_path, H_matrix_path, mask2_path, person_box_path, augment_person_box_path, distillation_mask_path,
      large_fov, basetype_id, augment_custom_person_box_path]}
    )
  return data_dict_list