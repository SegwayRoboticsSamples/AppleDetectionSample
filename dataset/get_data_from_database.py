import os 
import requests
import re
import cv2
import time

from dataset import fisheye_img_util
from dataset import email_sender

def unix_time(dt):
    time_array = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(time_array))
    return timestamp

def post_files(url='http://10.10.80.120:5099/annotated/update/t60/', data_family=None):
    request_ok_status = 200
    data = requests.post(url)
    if data.status_code != request_ok_status:
        print('post_files status_code is not ' + str(request_ok_status) + ', it is: ', data.status_code)
        err_str = 'Post files status code is not ' + str(request_ok_status) + ', it is: ' + str(data.status_code) + '. Post url is ' + url
        email_sender('Database operation exception.', err_str)
        input()
        return False
    else:
        return True

def fetch_new_files(url='http://10.10.80.120:5099/annotation/classify/t60/', data_family=None):
    request_ok_status = 200
    data = requests.post(url)
    if data.status_code != request_ok_status:
        print('get new files status code is not ' + str(request_ok_status) + ', it is: ', data.status_code)
        err_str = 'Getting new files status code is not ' + str(request_ok_status) + ', it is: ' + str(data.status_code) + '. Post url is ' + url
        email_sender('Database operation exception.', err_str)
        input()
        return False
    else:
        return True

def request_files(custom_root_dir, data_family, time_filter='2000-02-28 10:23:29'):
    data_list = []
    next_str = ''
    request_idx = 0
    request_ok_status = 200
    unix_t = unix_time(time_filter) #* 1000
    while 1:
        if next_str is None:
            print('next_str is None?')
            input()
        service_err_code = 500

        request_str = 'http://10.10.80.120:5099/annotated/t60/' + data_family
        request_str += '?size=2000'
        request_str += '&date=' + str(unix_t)
        request_str += '&next=' + next_str
        
        data = requests.get(request_str)

        count = 0
        while data.status_code == service_err_code:
            data = requests.get(request_str)
            count += 1
            if count > 20:
                print('status_code is ', service_err_code, ' for 20 times.')
                err_str = 'Request files status code keeps ' + str(service_err_code) + ' for 20 times. Request is: ' + request_str
                email_sender('Database operation exception.', err_str)
                input()
                break

        if data.status_code != request_ok_status:
            print('status_code is not ', request_ok_status, ' it is: ', data.status_code)
            err_str = 'Request files status code is not ' + str(request_ok_status) + ' is is: ' + str(data.status_code) + '. Request is: ' + request_str
            email_sender('Database operation exception.', err_str)
            input()

        cur_data_list = data.json()['list']
        if len(cur_data_list) == 0:
            print('cur_data_list is empty..')
            break
        data_list += cur_data_list
        next_str = data.json()['next']
        if next_str == '' or next_str is None:
            print('next_str is empty ', next_str)
            break
        print('request_idx ', request_idx)
        request_idx += 1

    print('get data size: ', len(data_list))


    custom_data_list = []
    custom_data_dir = {}

    for data_idx, data_path in enumerate(data_list):
        match_data_pat0 = '(\d+)-(\d+)-(\d+)_(\d+)-(\d+)-(\d+)'
        match_data_pat1 = '(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)'
        pattern = re.search(match_data_pat0, data_path, re.M) #|re.I
        if pattern:
            start_idx = pattern.start()
            sub_data_path = data_path[start_idx : ]
            print(sub_data_path, pattern.group())
        else:
            pattern = re.search(match_data_pat1, data_path, re.M) #|re.I
            if pattern is None:
                print('format err, ', data_path)
                input()
                continue
        sub_dir_key = ['manual/', 'discover_demo/', 'discover_alpha/', '/alpha/']
        ck_num = 0
        for sub_key in sub_dir_key:
            if data_path.find(sub_key) < 0:
                ck_num += 1
            else:
                tmp_idx = data_path.find(sub_key)

        if ck_num != len(sub_dir_key)-1:
            print('Maybe sub_dir_key is wrong?', ck_num, len(sub_dir_key), data_path)
            input()

        sub_data_path = data_path[tmp_idx : ]
        custom_data_path = custom_root_dir + '/' + sub_data_path
        print('custom_data_path ', custom_data_path)
        custom_data_list.append(custom_data_path)

        dir_name = custom_data_path[ : custom_data_path.rfind('/fisheye')]
        file_name = custom_data_path[custom_data_path.rfind('/fisheye') + len('/fisheye') + 1: ]
        print('dir_name ', dir_name)
        if custom_data_dir.__contains__(dir_name):
            custom_data_dir[dir_name] += [file_name]
        else:
            custom_data_dir[dir_name] = [file_name]

    data_path_list = []
    for (data_dir, file_list) in custom_data_dir.items():
        if not os.path.isdir(data_dir):
            print('data_dir is not directory: ', data_dir)
            input()
            continue
        print(data_dir)
        cur_folder = data_dir + '/fisheye'
        data_path_list += fisheye_img_util.data_folder_parse(cur_folder, file_list)
    return data_path_list