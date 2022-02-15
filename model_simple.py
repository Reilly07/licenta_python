# requires keras --pip install keras--
# requires tensorflow --pip install tensorflow--

import struct
import numpy as np
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow import keras
import tensorflow as tf

# For visualization of model
import os


os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'


debug = False


class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh, 
                    grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, 
                    **kwargs):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)), dtype=tf.float32)
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
        
        # initialize the masks
        object_mask     = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)        

        # compute grid factor and net factor
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        net_h       = tf.shape(input_image)[1]
        net_w       = tf.shape(input_image)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
        
        """
        Adjust prediction
        """
        pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
        pred_box_class = y_pred[..., 5:]                                                        # adjust class probabilities      

        """
        Adjust ground truth
        """
        true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
        true_box_wh    = y_true[..., 2:4] # t_wh
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)         

        """
        Compare each predicted box to all true boxes
        """        
        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0 

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious   = tf.reduce_max(iou_scores, axis=4)        
        conf_delta *= tf.expand_dims(tf.cast(best_ious < self.ignore_thresh, dtype=tf.float32), 4)

        """
        Compute some online statistics
        """            
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor 
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half      

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
        
        count       = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.cast((pred_box_conf*object_mask) >= 0.5, dtype=tf.float32)
        class_mask  = tf.expand_dims(tf.cast(tf.equal(tf.argmax(pred_box_class, -1), true_box_class), dtype=tf.float32), 4)
        recall50    = tf.reduce_sum(tf.cast(iou_scores >= 0.5 , dtype=tf.float32) * detect_mask  * class_mask) / (count + 1e-3)
        recall75    = tf.reduce_sum(tf.cast(iou_scores >= 0.75, dtype=tf.float32) * detect_mask  * class_mask) / (count + 1e-3)    
        avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
        avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
        avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3) 

        """
        Warm-up training
        """
        batch_seen = tf.compat.v1.assign_add(batch_seen, 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
                                       tf.ones_like(object_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])

        """
        Compare each true box to all anchor boxes
        """      
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
        conf_delta  = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + (1-object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                      self.class_scale

        loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
        loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
        loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
        loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        if debug:
            loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
            loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)   
            loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)     
            loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy), 
                                        tf.reduce_sum(loss_wh), 
                                        tf.reduce_sum(loss_conf), 
                                        tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t',   summarize=1000)

        return loss*self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def make_model(
	nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, 
    batch_size, 
    warmup_batches,
    ignore_thresh,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale
):

	#input_image = Input(shape=(416, 416, 3))
	input_image = Input(shape=(None, None, 3))
	true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4))
	true_yolo_1 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
	true_yolo_2 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
	true_yolo_3 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
	
	# Layer  0 => 4
	x = Conv2D(32, 3, strides=1, padding='same', name='conv_0', use_bias=False)(input_image)
	x = BatchNormalization(epsilon=0.001, name='bnorm_0')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_0')(x)

	x = ZeroPadding2D(((1, 0), (1, 0)))(x)

	x = Conv2D(64, 3, strides=2, padding='valid', name='conv_1', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_1')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_1')(x)
	skip = x

	x = Conv2D(32, 1, strides=1, padding='same', name='conv_2', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_2')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_2')(x)

	x = Conv2D(64, 3, strides=1, padding='same', name='conv_3', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_3')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_3')(x)

	x = add([skip, x])

	# Layer  5 => 8

	x = ZeroPadding2D(((1, 0), (1, 0)))(x)

	x = Conv2D(128, 3, strides=2, padding='valid', name='conv_5', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_5')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_5')(x)
	skip = x

	x = Conv2D(64, 1, strides=1, padding='same', name='conv_6', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_6')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_6')(x)

	x = Conv2D(128, 3, strides=1, padding='same', name='conv_7', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_7')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_7')(x)

	x = add([skip, x])

	# Layer  9 => 11

	skip = x

	x = Conv2D(64, 1, strides=1, padding='same', name='conv_9', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_9')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_9')(x)

	x = Conv2D(128, 3, strides=1, padding='same', name='conv_10', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_10')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_10')(x)

	x = add([skip, x])

	# Layer  12 => 15

	x = ZeroPadding2D(((1, 0), (1, 0)))(x)

	x = Conv2D(256, 3, strides=2, padding='valid', name='conv_12', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_12')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_12')(x)

	skip = x

	x = Conv2D(128, 1, strides=1, padding='same', name='conv_13', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_13')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_13')(x)

	x = Conv2D(256, 3, strides=1, padding='same', name='conv_14', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_14')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_14')(x)

	x = add([skip, x])

	# Layer  16 => 36

	for i in range(7):
		skip = x

		x = Conv2D(128, 1, strides=1, padding='same', name='conv_' + str(16 + i * 3), use_bias=False)(x)
		x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(16 + i * 3))(x)
		x = LeakyReLU(alpha=0.1, name='leaky_' + str(16 + i * 3))(x)

		x = Conv2D(256, 3, strides=1, padding='same', name='conv_' + str(17 + i * 3), use_bias=False)(x)
		x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(17 + i * 3))(x)
		x = LeakyReLU(alpha=0.1, name='leaky_' + str(17 + i * 3))(x)

		x = add([skip, x])

	skip_36 = x

	x = ZeroPadding2D(((1, 0), (1, 0)))(x)

	# Layer 37 => 40

	x = Conv2D(512, 3, strides=2, padding='valid', name='conv_37', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_37')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_37')(x)

	skip = x

	x = Conv2D(256, 1, strides=1, padding='same', name='conv_38', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_38')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_38')(x)

	x = Conv2D(512, 3, strides=1, padding='same', name='conv_39', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_39')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_39')(x)

	x = add([skip, x])

	# Layer 41 => 61

	for i in range(7):
		skip = x

		x = Conv2D(256, 1, strides=1, padding='same', name='conv_' + str(41 + i * 3), use_bias=False)(x)
		x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(41 + i * 3))(x)
		x = LeakyReLU(alpha=0.1, name='leaky_' + str(41 + i * 3))(x)

		x = Conv2D(512, 3, strides=1, padding='same', name='conv_' + str(42 + i * 3), use_bias=False)(x)
		x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(42 + i * 3))(x)
		x = LeakyReLU(alpha=0.1, name='leaky_' + str(42 + i * 3))(x)

		x = add([skip, x])

	skip_61 = x

	x = ZeroPadding2D(((1, 0), (1, 0)))(x)

	# Layer 62 => 65

	x = Conv2D(1024, 3, strides=2, padding='valid', name='conv_62', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_62')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_62')(x)

	skip = x

	x = Conv2D(512, 1, strides=1, padding='same', name='conv_63', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_63')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_63')(x)

	x = Conv2D(1024, 3, strides=1, padding='same', name='conv_64', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_64')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_64')(x)

	x = add([skip, x])

	# Layer 66 => 74

	for i in range(3):
		skip = x

		x = Conv2D(512, 1, strides=1, padding='same', name='conv_' + str(66 + i * 3), use_bias=False)(x)
		x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(66 + i * 3))(x)
		x = LeakyReLU(alpha=0.1, name='leaky_' + str(66 + i * 3))(x)

		x = Conv2D(1024, 3, strides=1, padding='same', name='conv_' + str(67 + i * 3), use_bias=False)(x)
		x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(67 + i * 3))(x)
		x = LeakyReLU(alpha=0.1, name='leaky_' + str(67 + i * 3))(x)

		x = add([skip, x])

	# Layer 75 => 79

	x = Conv2D(512, 1, strides=1, padding='same', name='conv_75', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_75')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_75')(x)

	x = Conv2D(1024, 3, strides=1, padding='same', name='conv_76', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_76')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_76')(x)

	x = Conv2D(512, 1, strides=1, padding='same', name='conv_77', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_77')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_77')(x)

	x = Conv2D(1024, 3, strides=1, padding='same', name='conv_78', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_78')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_78')(x)

	x = Conv2D(512, 1, strides=1, padding='same', name='conv_79', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_79')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_79')(x)

	# Layer 80 => 82

	yolo_82 = Conv2D(1024, 3, strides=1, padding='same', name='conv_80', use_bias=False)(x)
	yolo_82 = BatchNormalization(epsilon=0.001, name='bnorm_80')(yolo_82)
	yolo_82 = LeakyReLU(alpha=0.1, name='leaky_80')(yolo_82)

	#yolo_82 = Conv2D(255, 1, strides=1, padding='same', name='conv_81', use_bias=True)(yolo_82)
	yolo_82 = Conv2D(3*(5+nb_class), 1, strides=1, padding='same', name='conv_81', use_bias=True)(yolo_82)
	pred_yolo_1 = yolo_82
	
	loss_yolo_1 = YoloLayer(anchors[12:], 
                            [1*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])

	# Layer 83 => 86

	x = Conv2D(256, 1, strides=1, padding='same', name='conv_84', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_84')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_84')(x)

	x = UpSampling2D(2)(x)

	x = concatenate([x, skip_61])

	# Layer 87 => 91

	x = Conv2D(256, 1, strides=1, padding='same', name='conv_87', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_87')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_87')(x)

	x = Conv2D(512, 3, strides=1, padding='same', name='conv_88', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_88')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_88')(x)

	x = Conv2D(256, 1, strides=1, padding='same', name='conv_89', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_89')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_89')(x)

	x = Conv2D(512, 3, strides=1, padding='same', name='conv_90', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_90')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_90')(x)

	x = Conv2D(256, 1, strides=1, padding='same', name='conv_91', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_91')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_91')(x)

	# Layer 92 => 94

	yolo_94 = Conv2D(512, 3, strides=1, padding='same', name='conv_92', use_bias=False)(x)
	yolo_94 = BatchNormalization(epsilon=0.001, name='bnorm_92')(yolo_94)
	yolo_94 = LeakyReLU(alpha=0.1, name='leaky_92')(yolo_94)

	#yolo_94 = Conv2D(255, 1, strides=1, padding='same', name='conv_93', use_bias=True)(yolo_94)
	yolo_94 = Conv2D(3*(5+nb_class), 1, strides=1, padding='same', name='conv_93', use_bias=True)(yolo_94)
	pred_yolo_2 = yolo_94
	
	loss_yolo_2 = YoloLayer(anchors[6:12], 
                            [2*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])


	# Layer 95 => 98

	x = Conv2D(128, 1, strides=1, padding='same', name='conv_96', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_96')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_96')(x)

	x = UpSampling2D(2)(x)

	x = concatenate([x, skip_36])

	# Layer 99 => 106

	x = Conv2D(128, 1, strides=1, padding='same', name='conv_99', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_99')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_99')(x)

	x = Conv2D(256, 3, strides=1, padding='same', name='conv_100', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_100')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_100')(x)

	x = Conv2D(128, 1, strides=1, padding='same', name='conv_101', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_101')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_101')(x)

	x = Conv2D(256, 3, strides=1, padding='same', name='conv_102', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_102')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_102')(x)

	x = Conv2D(128, 1, strides=1, padding='same', name='conv_103', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_103')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_103')(x)

	x = Conv2D(256, 3, strides=1, padding='same', name='conv_104', use_bias=False)(x)
	x = BatchNormalization(epsilon=0.001, name='bnorm_104')(x)
	x = LeakyReLU(alpha=0.1, name='leaky_104')(x)

	#x = Conv2D(255, 1, strides=1, padding='same', name='conv_105', use_bias=True)(x)
	x = Conv2D(3*(5+nb_class), 1, strides=1, padding='same', name='conv_105', use_bias=True)(x)
	pred_yolo_3 = x
	
	loss_yolo_3 = YoloLayer(anchors[:6], 
                            [4*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_3, true_yolo_3, true_boxes]) 

	train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3], [loss_yolo_1, loss_yolo_2, loss_yolo_3])
	infer_model = Model(input_image, [yolo_82, yolo_94, x])
	#model = Model(input_image, [yolo_82, yolo_94, x])
	return [train_model, infer_model]
	#return model


def dummy_loss(y_true, y_pred):
	return tf.sqrt(tf.reduce_sum(y_pred))


'''
def main():
	# define the model
	model_train, model_infer = make_model()
	model_infer.summary()
	keras.utils.plot_model(model_infer, "model_simple.png", show_shapes=True)
	# load the model weights
	weight_reader = WeightReader('yolov3.weights')
	# set the model weights into the model
	weight_reader.load_weights(model)
	#model.load_weights("model_simple.h5")
	#save model to file
	model.save('model_simple.h5')


if __name__ == "__main__":
	main()
'''