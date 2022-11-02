#-*- coding:utf-8 -*-
import cv2
import os
import shutil

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
                
path=r'./datasets/val/'
file_list = []
label_list = []
all_gt_dict = []
dirAll(path)
point_color = (0, 255, 0) # BGR
thickness = 1 
lineType = 4

raw_img = 0
label_img = 0

for img_index in range(len(file_list)):  
    img_path = file_list[img_index]
    img = cv2.imread(img_path)
    save_path = './person_ground_truth/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    source_path_txt = '.' + img_path.split('.')[1] + '-person-boxes.txt'
    save_source_path = save_path + os.path.basename(source_path_txt)
    shutil.copyfile(source_path_txt, save_source_path)
    
    source_path_json = '.' + img_path.split('.')[1] + '.json'
    save_source_path = save_path + os.path.basename(source_path_json)
    shutil.copyfile(source_path_json, save_source_path)
        
    raw_save_path = save_path + os.path.basename(img_path)
    cv2.imwrite(raw_save_path, img)
    raw_img += 1   

    save_path = save_path + os.path.basename(img_path).split('.')[0]  +  '_label.jpg'

    #label res
    box_file = label_list[img_index]
    if(os.path.exists(box_file)):
        for line in open(box_file):
            sep_four = line.strip().split('],')
            sep = sep_four[0].split(',')
            sep[0] = sep[0][2:]
            for i in range(len(sep)):
                sep[i] = sep[i].strip()
            
            x1 = int(sep[0])
            y1 = int(sep[1])
            x2 = int(sep[2])
            y2 = int(sep[3])
            
            cv2.rectangle(img, (x1, y1), (x2, y2), point_color, thickness, lineType) 
    cv2.imwrite(save_path, img)
    label_img += 1
print("raw_img: ", raw_img)
print("label_img:", label_img) 
print("Done!")
