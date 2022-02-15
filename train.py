#! /usr/bin/env python
import tensorflow as tf

#tf.config.set_visible_devices([], 'GPU')
#gpus = tf.config.experimental.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(gpus[0], True)

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from model_simple import make_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from callbacks import CustomModelCheckpoint
#import model_yolo5_2 as model_yolo5_2
#from bdd import parse_bdd_annotation
#from bdd import parse_json
#from yolo import create_yolov3_model, dummy_loss
#from model_yolo5 import make_model_yolo5
#from model_yolo5_2 import build_model

#import keras
from tensorflow.keras.models import load_model
#tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)




config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def create_training_instances(
    train_annot_folder,
    train_image_folder,
    valid_annot_folder,
    valid_image_folder,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, labels)
    #train_ints, train_labels = parse_bdd_annotation(train_annot_folder, train_image_folder, labels)
    #train_ints, train_labels = parse_json(train_annot_folder, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, labels)
        #valid_ints, valid_labels = parse_bdd_annotation(valid_annot_folder, valid_image_folder, labels)
        #valid_ints, valid_labels = parse_json(valid_annot_folder, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        #np.random.seed(0)
        #np.random.shuffle(train_ints)
        #np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t'  + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, batch_size, 
    warmup_batches, 
    ignore_thresh,  
    saved_weights_name, 
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale  
):

    template_model, infer_model = make_model(
        nb_class            = nb_class, 
        anchors             = anchors, 
        max_box_per_image   = max_box_per_image, 
        max_grid            = max_grid, 
        batch_size          = batch_size, 
        warmup_batches      = warmup_batches,
        ignore_thresh       = ignore_thresh,
        grid_scales         = grid_scales,
        obj_scale           = obj_scale,
        noobj_scale         = noobj_scale,
        xywh_scale          = xywh_scale,
        class_scale         = class_scale
    )  

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name): 
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights("backend.h5", by_name=True)       

    train_model = template_model      

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model


def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    # Parse annotations
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['model']['labels']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    # Create generators for train and infer
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    # Create models
    if os.path.exists(config['train']['saved_weights_name']): 
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))   

    train_model, infer_model = create_model(
        nb_class            = len(labels), 
        anchors             = config['model']['anchors'], 
        max_box_per_image   = max_box_per_image, 
        max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
        batch_size          = config['train']['batch_size'], 
        warmup_batches      = warmup_batches,
        ignore_thresh       = config['train']['ignore_thresh'],
        saved_weights_name  = config['train']['saved_weights_name'],
        lr                  = config['train']['learning_rate'],
        grid_scales         = config['train']['grid_scales'],
        obj_scale           = config['train']['obj_scale'],
        noobj_scale         = config['train']['noobj_scale'],
        xywh_scale          = config['train']['xywh_scale'],
        class_scale         = config['train']['class_scale'],
    )

    # Create callbacks
    early_stop = EarlyStopping(
        monitor     = 'loss', 
        min_delta   = 0.01, 
        patience    = 10, 
        mode        = 'min', 
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = infer_model,
        filepath        = config['train']['saved_weights_name'],# + '{epoch:02d}.h5', 
        monitor         = 'loss', 
        verbose         = 1, 
        save_best_only  = True, 
        mode            = 'min', 
        period          = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )

    callbacks = [early_stop, checkpoint, reduce_on_plateau]
    #callbacks = create_callbacks(config['train']['saved_weights_name'], infer_model)

    #optimizer = tf.keras.optimizers.Adam(model_yolo5_2.CosineLrSchedule(len(train_generator) * config['train']['train_times']), 0.937)
    #train_model.compile(loss=dummy_loss, optimizer=optimizer)  

    # Start training
    print("Attention")
    print(len(train_generator))
    train_model.fit(
        x                = train_generator, 
        steps_per_epoch  = len(train_generator) * config['train']['train_times'], #231/483/931 or len(train_generator) * config['train']['train_times']
        epochs           = config['train']['nb_epochs'] + config['train']['warmup_epochs'], 
        verbose          = 2 if config['train']['debug'] else 1,
        callbacks        = callbacks, 
        workers          = 4,
        max_queue_size   = 8
    )

    infer_model = load_model(config['train']['saved_weights_name'])

    # Run evaluation
    # compute mAP for all the classes
    average_precisions, average_ious = evaluate(infer_model, valid_generator)
    # print the score
    print('mAP')
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))    

    print('iou')
    for label, average_iou in average_ious.items():
        print(labels[label] + ': {0}'.format(average_iou))
    print('iou: {0}'.format(sum(average_ious.values()) / len(average_ious))) 	


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
