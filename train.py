import tensorflow as tf
import os
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib import slim
import re
from deployment import model_deploy
from losses import yolov3_loss
from config import config
from data_reader import yolo_v3_reader
from model_builder import builder_llx

task_index = 0

def get_params():
    print("\nParsing Arguments..\n")
    conf = config.parse_confg()
    config.visualize_config(conf)
    return conf

def train():
    with tf.Graph().as_default():
        num_gpus = len(conf['gpus'].split(','))
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=num_gpus,
            clone_on_cpu=False,
            replica_id=0,
            num_replicas=1)
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()
        with tf.device(deploy_config.inputs_device()):
            reader = yolo_v3_reader.YOLOV3Reader(mode='train',
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
            iters = []
            ds = tf.data.TFRecordDataset(filenames)
            for i in range(num_gpus):
                single_ds = ds.shard(num_gpus, i)
                single_ds = single_ds.repeat().shuffle(buffer_size=5000).map(
                    reader.image_labels, num_parallel_calls=4).batch(
                        conf['batch_size']).prefetch(conf['batch_size'])
                batch_iter = single_ds.make_one_shot_iterator()
                iters.append(batch_iter)

        def clone_fn(iterators, total_anchors, grid_height, grid_width, 
            batch_size, number_object, number_class):
            global task_index
            (image, label1, label2, label3, mask1, mask2, 
                mask3) = iterators[task_index].get_next()
            task_index = task_index + 1

            loss_stage1 = yolov3_loss.YOLOV3Loss(
                grid_height, grid_width, [0, conf['num_object'] - 1], 
                total_anchors, number_class, number_object)
            loss_stage2 = yolov3_loss.YOLOV3Loss(
                2 * grid_height, 2 * grid_width, [conf['num_object'],
                2 * conf['num_object'] - 1], total_anchors, number_class, 
                number_object)
            loss_stage3 = yolov3_loss.YOLOV3Loss(
                4 * grid_height, 4 * grid_width, [2 * conf['num_object'],
                3 * conf['num_object'] - 1], total_anchors, number_class, 
                number_object)
            (model_output1, model_output2, model_output3) = \
                builder_llx.yolo_v3_ouput('mobilenetv1', True, 
                    image, conf['num_object'], conf['num_classes'], 
                    conf['grid_height'], conf['grid_width'], 
                    conf['depth_multiplier'])
                
            with tf.device(deploy_config.optimizer_device()):
                loss1 = loss_stage1.compute_loss(
                    label1, mask1, model_output1, batch_size)
                loss2 = loss_stage2.compute_loss(
                    label2, mask2, model_output2, batch_size)
                loss3 = loss_stage3.compute_loss(
                    label3, mask3, model_output3, batch_size)
                tf.losses.add_loss(loss1 + loss2 + loss3)
            return tf.losses.get_total_loss()

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        total_anchors = []
        for i in range(0, 6 * conf['num_object'], 2):
            total_anchors.append(
                conf['anchors'][i] / conf['img_width'])
            total_anchors.append(
                conf['anchors'][i + 1] / conf['img_height'])
  
        clones = model_deploy.create_clones(
            deploy_config, clone_fn, [iters, total_anchors, 
            conf['grid_height'], conf['grid_width'], 
            conf['batch_size'], conf['num_object'], conf['num_classes']])
        first_clone_scope = deploy_config.clone_scope(0)
        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            0.9, global_step)
        if conf['quantize']:
            tf.contrib.quantize.create_training_graph(
                quant_delay=conf['quant_delay'])
        with tf.device(deploy_config.optimizer_device()):
            boundaries =[10000, 30000, 60000]
            values = [0.000001, conf['learning_rate'], 
                conf['learning_rate'] * 0.1, conf['learning_rate'] * 0.01]
            learning_rate = tf.train.piecewise_constant(
                global_step, boundaries, values)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        variables_to_train = tf.trainable_variables()

        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
        inclusions = ['MobilenetV1']
        variables_to_restore = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            for inlusion in inclusions:
                if var.op.name.startswith(inlusion):
                    variables_to_restore.append(var)
                else:
                    break
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
        init_fn = slim.assign_from_checkpoint_fn(
            conf['pretrained_path'], variables_to_restore)
        # init_fn = None
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        if conf['resume']:
            # In case we're resuming, simply load the full checkpoint to init.
            last_checkpoint = tf.train.latest_checkpoint(conf['exp_dir'])
            print('Restoring from checkpoint: {}'.format(last_checkpoint))
            with tf.Session() as sess:
                saver.restore(sess, last_checkpoint)
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            slim.learning.train(
                train_tensor,
                logdir=conf['exp_dir'],
                saver=saver,
                session_config=session_config,
                summary_op=summary_op,
                number_of_steps=conf['max_number_of_steps'],
                save_summaries_secs=30,
                save_interval_secs=conf['save_interval_secs'])
        else:
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            slim.learning.train(
                train_tensor,
                logdir=conf['exp_dir'],
                init_fn=init_fn,
                saver=saver,
                session_config=session_config,
                summary_op=summary_op,
                number_of_steps=conf['max_number_of_steps'],
                save_summaries_secs=30,
                save_interval_secs=conf['save_interval_secs']
                )

if __name__ == '__main__':
    conf = get_params()
    os.environ['CUDA_VISIBLE_DEVICES']=conf['gpus']
    train()
