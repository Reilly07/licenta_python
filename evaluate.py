#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from bdd import parse_bdd_annotation
from bdd import parse_json
from yolo import create_yolov3_model
from model_yolo5_2 import build_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model

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

    template_model, infer_model = build_model(
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
    #if os.path.exists(saved_weights_name): 
        #print("\nLoading pretrained weights.\n")
        #template_model.load_weights(saved_weights_name)
    #else:
        #template_model.load_weights("backend.h5", by_name=True)       

    train_model = template_model      

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    # Create the validate generator
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_ints, labels = parse_voc_annotation(
			config['valid']['valid_annot_folder'], 
			config['valid']['valid_image_folder'], 
			config['model']['labels']
		)
    else:
        train_ints, labels = parse_voc_annotation(
			config['train']['train_annot_folder'], 
			config['train']['train_image_folder'], 
			config['model']['labels']
		)
        split = int(0.8*len(train_ints))
        train_valid_split = int(0.8*len(train_ints))
        valid_ints = train_ints[train_valid_split:]
	

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)
   
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    # Load model and evaluate
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    '''
    train_model, infer_model = create_model(
        nb_class            = len(labels), 
        anchors             = config['model']['anchors'], 
        max_box_per_image   = 10, 
        max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
        batch_size          = config['train']['batch_size'], 
        warmup_batches      = 0,
        ignore_thresh       = config['train']['ignore_thresh'],
        saved_weights_name  = config['train']['saved_weights_name'],
        lr                  = config['train']['learning_rate'],
        grid_scales         = config['train']['grid_scales'],
        obj_scale           = config['train']['obj_scale'],
        noobj_scale         = config['train']['noobj_scale'],
        xywh_scale          = config['train']['xywh_scale'],
        class_scale         = config['train']['class_scale'],
    )
    '''
    #infer_model.load_weights("raccoon.h5")
    infer_model = load_model(config['train']['saved_weights_name'])

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
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')    
    
    args = argparser.parse_args()
    _main_(args)
