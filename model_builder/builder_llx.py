import tensorflow as tf
from tensorflow.contrib import slim
import functools
# from nets.mobilenet import mobilenet_v2
from slim.nets import mobilenet_v1
# from nets import resnet_v2
# from model_builder import darknet53
# from model_builder import resnet_v2_18_200

def get_mobilenetv1_model_v2(is_training,
                        x,
                        num_object,
                        num_class,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        batch_norm_scale=True,
                        weight_decay=0.00004,
                        stddev=0.09,
                        regularize_depthwise=True):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': batch_norm_scale,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, 
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], 
                                weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], 
                                    weights_regularizer=depthwise_regularizer):
                    _, endpoints = mobilenet_v1.mobilenet_v1(
                        x, 
                        depth_multiplier=0.5, num_classes=1001, 
                        is_training=is_training, spatial_squeeze=False)
                    con_activation_fn = functools.partial(tf.nn.leaky_relu, 
                                                          alpha=0.1)
                    with (slim.arg_scope([slim.conv2d], 
                                        activation_fn=con_activation_fn), 
                        tf.name_scope("head")):
                        reg_output = slim.conv2d(
                            endpoints['Conv2d_13_depthwise'], 
                            num_object * (4 + 1 + num_class), [1, 1],
                            stride=1, padding='SAME', activation_fn=None, 
                            normalizer_fn=None, scope='reg13x13_output')
                        return reg_output

def get_mobilenetv1_model(is_training,
                        x,
                        num_object,
                        num_class,
                        grid_height,
                        grid_width,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        weight_decay=0.00004,
                        stddev=0.09,
                        batch_norm_scale=True,
                        regularize_depthwise=True,
                        depth_multiplier=0.25):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': batch_norm_scale,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, 
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], 
                                weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], 
                                    weights_regularizer=depthwise_regularizer):
                    _, endpoints = mobilenet_v1.mobilenet_v1(
                        x, 
                        num_classes=1001, is_training=is_training, 
                        spatial_squeeze=False, depth_multiplier=depth_multiplier)
                    # con_activation_fn = functools.partial(
                    #     tf.nn.leaky_relu, alpha=0.1)
                    con_activation_fn = functools.partial(
                        tf.nn.relu)
                    with slim.arg_scope([slim.conv2d], activation_fn=con_activation_fn):
                        with tf.name_scope("head"):
                            # reg1_1 = slim.conv2d(endpoints['Conv2d_13_depthwise'], 
                            #     32, [1, 1])
                            # reg1_2 = slim.conv2d(reg1_1, 64, [3, 3])
                            # reg1_3 = slim.conv2d(reg1_2, 32, [1, 1])
                            # reg1_4 = slim.conv2d(reg1_3, 64, [3, 3])
                            # reg1_5 = slim.conv2d(reg1_4, 32, [1, 1])
                            # reg1_6 = slim.conv2d(reg1_5, 64, [3, 3])
                            reg1_1 = slim.conv2d(endpoints['Conv2d_13_depthwise'], 
                                64, [1, 1])
                            reg1_2 = slim.conv2d(reg1_1, 128, [3, 3])
                            reg1_3 = slim.conv2d(reg1_2, 64, [1, 1])
                            reg1_4 = slim.conv2d(reg1_3, 128, [3, 3])
                            reg1_5 = slim.conv2d(reg1_4, 64, [1, 1])
                            reg1_6 = slim.conv2d(reg1_5, 128, [3, 3])
                            reg1_output = slim.conv2d(
                                reg1_6, num_object * (4 + 1 + num_class), [1, 1],
                                stride=1, padding='SAME', activation_fn=None, 
                                normalizer_fn=None, scope='reg13x13_output')
                            reg2_1 = slim.conv2d(reg1_5, 32, [1, 1])
                            reg2_2 = tf.image.resize_bilinear(
                                reg2_1, [2 * grid_height, 2 * grid_width])
                            reg2_3 = tf.concat(
                                [reg2_2, endpoints['Conv2d_11_depthwise']], axis=-1)
                            reg2_4 = slim.conv2d(reg2_3, 32, [1, 1])
                            reg2_5 = slim.conv2d(reg2_4, 64, [3, 3])
                            reg2_6 = slim.conv2d(reg2_5, 32, [1, 1])
                            reg2_7 = slim.conv2d(reg2_6, 64, [3, 3])
                            reg2_8 = slim.conv2d(reg2_7, 32, [1, 1])
                            reg2_9 = slim.conv2d(reg2_8, 64, [3, 3])
                            # reg2_1 = slim.conv2d(reg1_5, 16, [1, 1])
                            # reg2_2 = tf.image.resize_bilinear(
                            #     reg2_1, [2 * grid_height, 2 * grid_width])
                            # reg2_3 = tf.concat(
                            #     [reg2_2, endpoints['Conv2d_11_depthwise']], axis=-1)
                            # reg2_4 = slim.conv2d(reg2_3, 16, [1, 1])
                            # reg2_5 = slim.conv2d(reg2_4, 32, [3, 3])
                            # reg2_6 = slim.conv2d(reg2_5, 16, [1, 1])
                            # reg2_7 = slim.conv2d(reg2_6, 32, [3, 3])
                            # reg2_8 = slim.conv2d(reg2_7, 16, [1, 1])
                            # reg2_9 = slim.conv2d(reg2_8, 32, [3, 3])
                            reg2_output = slim.conv2d(
                                reg2_9, num_object * (4 + 1 + num_class), [1, 1],
                                stride=1, padding='SAME', activation_fn=None, 
                                normalizer_fn=None, scope='reg26x26')
                            reg3_1 = slim.conv2d(reg2_8, 16, [1, 1])
                            reg3_2 = tf.image.resize_bilinear(
                                reg3_1, [4 * grid_height, 4 * grid_width])
                            reg3_3 = tf.concat(
                                [reg3_2, endpoints['Conv2d_5_depthwise']], axis=-1)
                            reg3_4 = slim.conv2d(reg3_3, 16, [1, 1])
                            reg3_5 = slim.conv2d(reg3_4, 32, [3, 3])
                            reg3_6 = slim.conv2d(reg3_5, 16, [1, 1])
                            reg3_7 = slim.conv2d(reg3_6, 32, [3, 3])
                            reg3_8 = slim.conv2d(reg3_7, 16, [1, 1])
                            reg3_9 = slim.conv2d(reg3_8, 32, [3, 3])
                            # reg3_1 = slim.conv2d(reg2_8, 8, [1, 1])
                            # reg3_2 = tf.image.resize_bilinear(
                            #     reg3_1, [4 * grid_height, 4 * grid_width])
                            # reg3_3 = tf.concat(
                            #     [reg3_2, endpoints['Conv2d_5_depthwise']], axis=-1)
                            # reg3_4 = slim.conv2d(reg3_3, 8, [1, 1])
                            # reg3_5 = slim.conv2d(reg3_4, 16, [3, 3])
                            # reg3_6 = slim.conv2d(reg3_5, 8, [1, 1])
                            # reg3_7 = slim.conv2d(reg3_6, 16, [3, 3])
                            # reg3_8 = slim.conv2d(reg3_7, 8, [1, 1])
                            # reg3_9 = slim.conv2d(reg3_8, 16, [3, 3])
                            reg3_output = slim.conv2d(
                                reg3_9, num_object * (4 + 1 + num_class), [1, 1],
                                stride=1, padding='SAME', activation_fn=None, 
                                normalizer_fn=None, scope='reg52x52')
                            print("output_tensor:", reg1_output, reg2_output, reg3_output)
                            return reg1_output, reg2_output, reg3_output

def get_mobilenetv2_model(loss_type,
                        is_training,
                        endpoints,
                        num_object,
                        num_class,
                        grid_height,
                        grid_width,
                        batch_norm_decay=0.9997,
                        batch_norm_scale=True,
                        batch_norm_epsilon=0.001,
                        weight_decay=0.00004):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': batch_norm_scale,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    if loss_type== "yolov3":
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu, 
                            normalizer_params=batch_norm_params):
            reg1_1 = slim.conv2d(
                endpoints['layer_16/depthwise_output'], 128, [1, 1])
            reg1_2 = slim.conv2d(reg1_1, 256, [3, 3])
            reg1_3 = slim.conv2d(reg1_2, 128, [1, 1])
            reg1_4 = slim.conv2d(reg1_3, 256, [3, 3])
            reg1_5 = slim.conv2d(reg1_4, 128, [1, 1])
            reg1_6 = slim.conv2d(reg1_5, 256, [3, 3])
            reg1_output = slim.conv2d(reg1_6,num_object * (4 + 1 + num_class), 
                [1, 1], stride=1, padding='SAME', activation_fn=None, 
                normalizer_fn=None)
            reg2_1 = slim.conv2d(reg1_5, 64, [1, 1])
            
            #reg2_2 = tf.image.resize_bilinear(reg2_1,
                #[2 * grid_height, 2 * grid_width])
            reg2_2 = tf.layers.conv2d_transpose(reg2_1, 64, 3 , strides=2, padding='SAME')
            reg2_3 = tf.concat([reg2_2, 
                endpoints['layer_10/depthwise_output']], axis=3)
            
            #reg2_3 = endpoints['layer_10/depthwise_output']
            reg2_4 = slim.conv2d(reg2_3, 64, [1, 1])
            reg2_5 = slim.conv2d(reg2_4, 128, [3, 3])
            reg2_6 = slim.conv2d(reg2_5, 64, [1, 1])
            reg2_7 = slim.conv2d(reg2_6, 128, [3, 3])
            reg2_8 = slim.conv2d(reg2_7, 64, [1, 1])
            reg2_9 = slim.conv2d(reg2_8, 128, [3, 3])
            reg2_output = slim.conv2d(
                reg2_9, num_object*(4 + 1 + num_class), [1, 1], stride=1, 
                padding='SAME', activation_fn=None, normalizer_fn=None)
            reg3_1 = slim.conv2d(reg2_8, 32, [1, 1])
            
            #reg3_2 = tf.image.resize_bilinear(reg3_1, 
                #[4 * grid_height, 4 * grid_width])
            reg3_2 = tf.layers.conv2d_transpose(reg3_1, 32, 3 , strides=2, padding='SAME')
            reg3_3 = tf.concat([reg3_2, 
                endpoints['layer_5/depthwise_output']], axis=3)
            
            #reg3_3 = endpoints['layer_5/depthwise_output']
            reg3_4 = slim.conv2d(reg3_3, 32, [1, 1])
            reg3_5 = slim.conv2d(reg3_4, 64, [3, 3])
            reg3_6 = slim.conv2d(reg3_5, 32, [1, 1])
            reg3_7 = slim.conv2d(reg3_6, 64, [3, 3])
            reg3_8 = slim.conv2d(reg3_7, 32, [1, 1])
            reg3_9 = slim.conv2d(reg3_8, 64, [3, 3])
            reg3_output = slim.conv2d(
                reg3_9, num_object * (4 + 1 + num_class), [1, 1], stride=1, 
                padding='SAME', activation_fn=None, normalizer_fn=None)
            print("output_tensor:",reg1_output,reg2_output,reg3_output)
            return reg1_output, reg2_output, reg3_output
    elif loss_type == "yolov2":
        with slim.arg_scope(mobilenet_v2.training_scope(
                is_training=is_training, bn_decay=batch_norm_decay, 
                weight_decay=weight_decay, stddev=-0.5)):
            reg_output = slim.conv2d(endpoints['layer_16/depthwise_output'], 
                num_object * (4 + 1 + num_class), [1, 1], stride=1, 
                padding='SAME', activation_fn=None, normalizer_fn=None, 
                scope='reg_output')
            return reg_output

def get_resnet50_model(is_training,
                        endpoints,
                        num_object,
                        num_class,
                        grid_height,
                        grid_width,
                        batch_norm_decay=0.9997,
                        batch_norm_scale=True,
                        batch_norm_epsilon=0.001,
                        weight_decay=0.00004):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': batch_norm_scale,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu, 
        normalizer_params=batch_norm_params):
        reg1_1 = slim.conv2d(endpoints['resnet_v2_50/block4'], 512, [1, 1])
        reg1_2 = slim.conv2d(reg1_1, 1024, [3, 3])
        reg1_3 = slim.conv2d(reg1_2, 512, [1, 1])
        reg1_4 = slim.conv2d(reg1_3, 1024, [3, 3])
        reg1_5 = slim.conv2d(reg1_4, 512, [1, 1])
        reg1_6 = slim.conv2d(reg1_5, 1024, [3, 3])
        reg1_output = slim.conv2d(reg1_6, num_object * (4 + 1 + num_class), 
            [1, 1], stride=1, padding='SAME', activation_fn=None, 
            normalizer_fn=None)
        '''
        reg2_1 = slim.conv2d(reg1_5, 256, [1, 1])
        reg2_2 = tf.image.resize_bilinear(
            reg2_1, [2 * grid_height, 2 * grid_width])
        reg2_3 = tf.concat([reg2_2, endpoints['resnet_v2_50/block2']], axis=-1)
        '''
        reg2_3 = endpoints['resnet_v2_50/block2']

        reg2_4 = slim.conv2d(reg2_3, 256, [1, 1])
        reg2_5 = slim.conv2d(reg2_4, 512, [3, 3])
        reg2_6 = slim.conv2d(reg2_5, 256, [1, 1])
        reg2_7 = slim.conv2d(reg2_6, 512, [3, 3])
        reg2_8 = slim.conv2d(reg2_7, 256, [1, 1])
        reg2_9 = slim.conv2d(reg2_8, 512, [3, 3])
        reg2_output = slim.conv2d(
            reg2_9, num_object * (4 + 1 + num_class), [1, 1], stride=1, 
            padding='SAME', activation_fn=None, normalizer_fn=None)
        '''
        reg3_1 = slim.conv2d(reg2_8, 128, [1, 1])
        reg3_2 = tf.image.resize_bilinear(
            reg3_1, [4 * grid_height, 4 * grid_width])
        reg3_3 = tf.concat([reg3_2, endpoints['resnet_v2_50/block1']], axis=-1)
        '''
        reg3_3 = endpoints['resnet_v2_50/block1']

        reg3_4 = slim.conv2d(reg3_3, 128, [1, 1])
        reg3_5 = slim.conv2d(reg3_4, 256, [3, 3])
        reg3_6 = slim.conv2d(reg3_5, 128, [1, 1])
        reg3_7 = slim.conv2d(reg3_6, 256, [3, 3])
        reg3_8 = slim.conv2d(reg3_7, 128, [1, 1])
        reg3_9 = slim.conv2d(reg3_8, 256, [3, 3])
        reg3_output = slim.conv2d(
            reg3_9, num_object * (4 + 1 + num_class), [1, 1], stride=1, 
            padding='SAME', activation_fn=None, normalizer_fn=None)
        return reg1_output, reg2_output, reg3_output

def get_resnet18_model(is_training,
                        endpoints,
                        num_object,
                        num_class,
                        grid_height,
                        grid_width,
                        batch_norm_decay=0.9997,
                        batch_norm_scale=True,
                        batch_norm_epsilon=0.001,
                        weight_decay=0.00004):
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': batch_norm_scale,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu, 
                        normalizer_params=batch_norm_params):
        reg1_1 = slim.conv2d(endpoints['resnet_v2_18/block4'], 128, [1, 1])
        reg1_2 = slim.conv2d(reg1_1, 256, [3, 3])
        reg1_3 = slim.conv2d(reg1_2, 128, [1, 1])
        reg1_4 = slim.conv2d(reg1_3, 256, [3, 3])
        reg1_5 = slim.conv2d(reg1_4, 128, [1, 1])
        reg1_6 = slim.conv2d(reg1_5, 256, [3, 3])
        reg1_output = slim.conv2d(reg1_6, num_object * (4 + 1 + num_class), 
            [1, 1], stride=1, padding='SAME', activation_fn=None, 
            normalizer_fn=None)
        reg2_1 = slim.conv2d(reg1_5, 128, [1, 1])
        reg2_2 = tf.image.resize_bilinear(reg2_1,[2 * grid_height, 2 * grid_width])
        reg2_3 = tf.concat([reg2_2, endpoints['resnet_v2_18/block2']], axis=3)
        reg2_4 = slim.conv2d(reg2_3, 128, [1, 1])
        reg2_5 = slim.conv2d(reg2_4, 256, [3, 3])
        reg2_6 = slim.conv2d(reg2_5, 128, [1, 1])
        reg2_7 = slim.conv2d(reg2_6, 256, [3, 3])
        reg2_8 = slim.conv2d(reg2_7, 128, [1, 1])
        reg2_9 = slim.conv2d(reg2_8, 256, [3, 3])
        reg2_output = slim.conv2d(
            reg2_9, num_object * (4 + 1 + num_class), [1, 1], stride=1, 
            padding='SAME', activation_fn=None, normalizer_fn=None)
        reg3_1 = slim.conv2d(reg2_8, 128, [1, 1])
        reg3_2 = tf.image.resize_bilinear(
            reg3_1, [4 * grid_height, 4 * grid_width])
        reg3_3 = tf.concat([reg3_2, endpoints['resnet_v2_18/block1']], axis=3)
        reg3_4 = slim.conv2d(reg3_3, 128, [1, 1])
        reg3_5 = slim.conv2d(reg3_4, 256, [3, 3])
        reg3_6 = slim.conv2d(reg3_5, 128, [1, 1])
        reg3_7 = slim.conv2d(reg3_6, 256, [3, 3])
        reg3_8 = slim.conv2d(reg3_7, 128, [1, 1])
        reg3_9 = slim.conv2d(reg3_8, 256, [3, 3])
        reg3_output = slim.conv2d(
            reg3_9, num_object * (4 + 1 + num_class), [1, 1], stride=1, 
            padding='SAME', activation_fn=None, normalizer_fn=None)
        return reg1_output, reg2_output, reg3_output

def get_resnet18_model_v2(is_training,
                        endpoints,
                        num_object,
                        num_class):
    reg1_output = slim.conv2d(endpoints['resnet_v2_18/block4'], 
        num_object * (4 + 1 + num_class), [1, 1], stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None, scope='reg_output')
    return reg1_output

def yolo_v2_output(model_name,
                    is_training,
                    image,
                    num_object,
                    num_class,
                    grid_height,
                    grid_width,
                    depth_multiplier,
                    batch_norm_decay=0.9997,
                    weight_decay=0.00004):
    if model_name == "mobilenetv2":
        with slim.arg_scope(mobilenet_v2.training_scope(
            is_training=is_training, bn_decay=0.9997, weight_decay=0.00002)):
            _, endpoints = mobilenet_v2.mobilenet(
                image, depth_multiplier=depth_multiplier, 
                finegrain_classification_mode=True)
            return get_mobilenetv2_model(
                "yolov2", is_training, endpoints, num_object, num_class, 
                grid_height, grid_width, batch_norm_decay=batch_norm_decay, 
                weight_decay=weight_decay)
    elif model_name == "mobilenetv1":
        return get_mobilenetv1_model_v2(
            is_training, image, num_object, num_class)
    elif model_name == "resnet18":
        with slim.arg_scope(
            resnet_v2_18_200.resnet_arg_scope(weight_decay=0.0005)):
            _, endpoints = resnet_v2_18_200.resnet_v2_18(
                image, is_training=is_training, global_pool=False)
        return get_resnet18_model_v2(
            is_training, endpoints, num_object, num_class)
    else:
        print("\n not support model :{}".format(model_name))
        exit()

def trident_output(model_name,
                    is_training,
                    image,
                    num_object,
                    num_class,
                    grid_height,
                    grid_width,
                    weight_decay=0.00004,
                    batch_norm_decay=0.9997):
    if model_name == "trident_resnet18":
        with slim.arg_scope(resnet_v2_18_200.resnet_arg_scope(
                            weight_decay=weight_decay)):
            br1, br2, br3 = resnet_v2_18_200.resnet_v2_18_trident(
                image, is_training=is_training)
            reg1_output = slim.conv2d(br1, num_object * (4 + 1 + num_class), 
                [1, 1], stride=1, padding='SAME', activation_fn=None, 
                normalizer_fn=None)
            reg2_output = slim.conv2d(br2, num_object * (4 + 1 + num_class), 
                [1, 1], stride=1, padding='SAME', activation_fn=None, 
                normalizer_fn=None)
            reg3_output = slim.conv2d(br3, num_object * (4 + 1 + num_class), 
                [1, 1], stride=1, padding='SAME', activation_fn=None, 
                normalizer_fn=None)
        return reg1_output, reg2_output, reg3_output
    else:
        print("\n not support model :{}".format(model_name))
        exit()

def yolo_v3_ouput(model_name,
                    is_training,
                    image,
                    num_object,
                    num_class,
                    grid_height,
                    grid_width,
                    depth_multiplier,
                    batch_norm_decay=0.9997,
                    weight_decay=0.00004):
    if model_name == "mobilenetv2":
        with slim.arg_scope(mobilenet_v2.training_scope(
            is_training=is_training, bn_decay=0.997, weight_decay=0.0002)):
            _, endpoints = mobilenet_v2.mobilenet(
                image, depth_multiplier=depth_multiplier, 
                finegrain_classification_mode=True)
            return get_mobilenetv2_model(
                "yolov3", is_training, endpoints, num_object, num_class, 
                grid_height, grid_width, batch_norm_decay=batch_norm_decay, 
                weight_decay=weight_decay)
    elif model_name == "mobilenetv1":
        return get_mobilenetv1_model(
            is_training, image, num_object, num_class, grid_height, grid_width, 
            batch_norm_decay=batch_norm_decay, weight_decay=weight_decay, 
            depth_multiplier=depth_multiplier)
    elif model_name == "resnet50":
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            _, endpoints = resnet_v2.resnet_v2_50(
                image, is_training=is_training, global_pool=False)
        return get_resnet50_model(
            is_training, endpoints, num_object, num_class, grid_height, 
            grid_width, batch_norm_decay=batch_norm_decay, 
            weight_decay=weight_decay)
    elif model_name == "resnet18":
        with slim.arg_scope(
            resnet_v2_18_200.resnet_arg_scope(weight_decay=0.0001)):
            _, endpoints = resnet_v2_18_200.resnet_v2_18(
                image, is_training=is_training, global_pool=False)
        return get_resnet18_model(
            is_training, endpoints, num_object, num_class, grid_height, 
            grid_width, batch_norm_decay=batch_norm_decay, 
            weight_decay=weight_decay)
    elif model_name == "darknet53":
        with tf.variable_scope('detector'):
            model_output1, model_output2, model_output3 = darknet53.yolo_v3(
                image,
                num_class,
                num_object,
                is_training=is_training)
        return model_output1, model_output2, model_output3
    else:
        print("\n not support model :{}".format(model_name))
        exit()
