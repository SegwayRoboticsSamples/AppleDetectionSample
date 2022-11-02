import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib import slim
from metrics import average_precision
from boxes_parser import yolo_parser
from config import config
from data_reader import yolo_v3_reader
from model_builder import builder_llx
from PIL import Image
import cv2
import time
import yaml

from tensorflow.python.client import timeline

def get_params():
    """load config file
    Returns:
	a dictionary contains all parameters.
    """
    print("\nParsing Arguments..\n")
    conf = config.parse_confg()
    config.visualize_config(conf)
    return conf


def eval(conf, train_conf):
    image_num = train_conf['train_img_num']
    batch_size = train_conf['batch_size']

    if isinstance(train_conf["gpus"], int):
        gpu_list = str(train_conf['gpus'])
    elif isinstance(train_conf["gpus"], str):
        gpu_list = train_conf["gpus"]
    gpu_nums = len(gpu_list.split(','))
    
    end_idx = int(conf['test_model'].split('-')[-1])
    start_idx = int(end_idx - image_num / batch_size / gpu_nums * int(conf['epoch_range']))

    model_path = conf['exp_dir']
    models = []
    file_names = os.listdir(model_path)
    if file_names == []:
        print("No model in dir. Check the path.")
        input()
    for file in file_names:
        if file[0:5] == 'model':
            model = os.path.splitext(file)[0]
            if model not in models:
                models.append(model)
    models = list(set(models))
    models = sorted(models)

    mAP_list = []
    mAP_models = []
    for tmd, test_model in enumerate(models):
        iteration = int(test_model[11:])
        if iteration < start_idx or iteration > end_idx:
            continue
        print("Deal the model: ", test_model)
        tf.reset_default_graph()
        reader = yolo_v3_reader.YOLOV3Reader(mode='test',
            anchors=conf['anchors'],
            resize_width=conf['img_width'],
            resize_height=conf['img_height'],
            num_object=conf['num_object'],
            num_class=conf['num_classes'],
            grid_width=conf['grid_width'],
            grid_height=conf['grid_height'],
            random_flip=conf['random_flip'],
            jitter_box=conf['random_jitter_box'],
            random_crop=conf['random_crop'],
            distort_color=conf['random_distort_color'])
     
        if os.path.isdir(conf['data_dir']):
            filenames = [os.path.join(
            conf['data_dir'], x) for x in os.listdir(conf['data_dir'])]
        else:
            filenames= [conf['data_dir']]
        ds = tf.data.TFRecordDataset(filenames)
        ds = ds.map(reader.image_labels).shuffle(
            buffer_size=500).repeat().batch(conf['batch_size']).prefetch(conf['batch_size'])
        iterator = ds.make_one_shot_iterator()
        
        image_batch, _, label_batch2, _, mask1, mask2, mask3= iterator.get_next()
        # image_batch, label_batch1, label_batch2, label_batch3, mask1, mask2, mask3 = iterator.get_next()
        # c1 = tf.reduce_sum(mask1)
        # c2 = tf.reduce_sum(mask2)
        # c3 = tf.reduce_sum(mask3)
        
        data_node = tf.placeholder(tf.float32,
            shape=[None, conf['img_height'], conf['img_width'], 3])
        (model_output1, model_output2, model_output3)= builder_llx.yolo_v3_ouput(
            'mobilenetv1', False, data_node, conf['num_object'], 
            conf['num_classes'], conf['grid_height'], conf['grid_width'],
            conf['depth_multiplier'])
        print(data_node)
        print(model_output1,model_output2,model_output3)
        
        if conf['quantize']:
            tf.contrib.quantize.create_eval_graph()
        variables_to_restore = slim.get_variables_to_restore()
        init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(conf['exp_dir'], test_model), variables_to_restore)

        total_anchors = []
        for i in range(0, 6 * conf['num_object'], 2):
            total_anchors.append(conf['anchors'][i] / conf['img_width'])
            total_anchors.append(conf['anchors'][i + 1] / conf['img_height'])
        
        with tf.Session() as sess:
            #quantize
            
            
            init_fn(sess)
            tf.train.write_graph(
                sess.graph, conf['exp_dir'], 'graph_eval.pbtxt', as_text=True)
            start_index = 0
            all_gt_dict = {}
            all_predict_boxes = {}
            for c in range(conf['num_classes']):
                all_gt_dict[c] = {}
                all_predict_boxes[c] = []

            imglst=[]
            t1=time.time()
            t_m=0
            t_b=0
            t_r=0
            for j in range(conf['max_number_of_steps']):
                #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()
                temp2=time.time()
                image, labels2 = sess.run([image_batch, label_batch2])
                #image, labels1,n1,n2,n3 = sess.run([image_batch, label_batch2,c1,c2,c3])
                # image, labels1, labels2, labels3 = \
                #     sess.run([image_batch, label_batch1, label_batch2, label_batch3])
                #,options=options, run_metadata=run_metadata)
                t_r=t_r+time.time()-temp2
                #print("num:",n1,n2,n3)
                # Create the Timeline object, and write it to a json file
                #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #with open('timeline_data.json', 'w') as f:
                    #f.write(chrome_trace)

                k=0
                for img in image:
                    imglst.append(img*255.0)
                    #Image.fromarray((img*255.0).astype('uint8')).convert('RGB').save("../data/show/"+str(k)+".jpeg")
                    k = k+1
                #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()

                temp=time.time()
                reg1, reg2, reg3 = sess.run(
                    [model_output1, model_output2, model_output3],
                    feed_dict={data_node: image})
                # reg1, reg2 = sess.run(
                #     [model_output1, model_output2],
                #     feed_dict={data_node: image})
                image_show = np.squeeze(image)
                image_show = (image_show*255.0).astype(np.uint8)
                #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #with open('timeline_20.json', 'w') as f:
                    #f.write(chrome_trace)

                temp1=time.time()
                if j >= 25:
                    t_m=t_m+temp1-temp
                start_index = yolo_parser.collect_predict_gt_in_yolo_v3_batch(
                    start_index, [reg1, reg2, reg3], 
                    labels2, all_gt_dict, 
                    all_predict_boxes, conf['num_object'], conf['num_classes'],
                    total_anchors)
                # start_index = yolo_parser.collect_predict_gt_in_yolo_v3_batch(
                #     start_index, [reg1, reg2], 
                #     labels2, all_gt_dict, 
                #     all_predict_boxes, conf['num_object'], conf['num_classes'],
                #     total_anchors)
                t_b=t_b+time.time()-temp1
                if start_index % 500 == 0:
                    print(start_index)
            mAP = 0.0
            t2=time.time()
            # print("t:",t_r,t_m, t_b, t2-t1,(t2-t1)/conf['max_number_of_steps']/conf['batch_size'])
            imglst2=imglst.copy()
            if(os.path.exists("result.txt")):
                print('c++ file exists')
                for c in range(conf['num_classes']):
                    all_predict_boxes[c].clear()
                for line in open("result.txt"):
                    sep = line.strip().split(' ')
                    box=[]
                    box.append(float(sep[0]))
                    box.append(float(sep[1]))
                    box.append(float(sep[2]))
                    box.append(float(sep[3]))
                    box.append(int(sep[4]))
                    box.append(float(sep[5]))
                    box.append(int(sep[6]))
                    all_predict_boxes[int(sep[4])].append(box)
                    
            for c in range(conf['num_classes']):
                ap, predict_box = average_precision.calculate_averge_precision(
                    all_predict_boxes[c], all_gt_dict[c])
                print(ap)
                mAP += ap

                #for key in range(1000):
                    #Image.fromarray(imglst[key].astype('uint8')).convert('RGB').save("../data/show/"+str(key)+".jpg")
            
                # for key in predict_box.keys():
                #     for box in predict_box[key]:
                #         cv2.rectangle(imglst[key],(int(box[0]*conf['img_width']),int(box[1]*conf['img_height'])),(int(box[2]*conf['img_width']),int(box[3]*conf['img_height'])),(0,0,255),3)
                    # Image.fromarray(imglst[key].astype('uint8')).convert('RGB').save("../data/show/"+str(key)+".jpg")
                # for i in range(len(imglst)):
                #     cv2.imwrite("./data/show/"+str(i)+".jpg", imglst[i])
                
                # for key in all_gt_dict[c].keys():
                #     for box in all_gt_dict[c][key]:
                #         cv2.rectangle(imglst[key],(int(box[0]*conf['img_width']),int(box[1]*conf['img_height'])),(int(box[2]*conf['img_width']),int(box[3]*conf['img_height'])),(0,255,0),3)
                #     # Image.fromarray(imglst2[key].astype('uint8')).convert('RGB').save("../data/truth/"+str(key)+".jpg")
                # for i in range(len(imglst)):
                #     cv2.imwrite("./data/show/"+str(i)+".jpg", imglst[i])
                    
            # print("map in test data: {:.3f}".format(mAP / conf['num_classes']))
            mAP_list.append(mAP)
            mAP_models.append(test_model)
            print("map in model: [" + str(test_model) + "] is ", str(mAP) )


    for mid, model in enumerate(mAP_list):
        print(mAP_models[mid])
        print(mAP_list[mid])

if __name__ == '__main__':
    conf = get_params()
    config_file = open('./config/train.yaml')
    train_conf = yaml.load(config_file.read(), Loader=yaml.Loader)
    config_file.close()
    os.environ['CUDA_VISIBLE_DEVICES'] = conf['gpus']
    eval(conf, train_conf)
