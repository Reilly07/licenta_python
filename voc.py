import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import json
import argparse


def parse_voc_annotation(ann_dir, img_dir, labels=[]):

    all_insts = []
    seen_labels = {}
        
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        try:
            tree = ET.parse(ann_dir + ann)
            #print(ann_dir + ann)
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + ann_dir + ann)
            continue
            
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                    
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        #print(attr.text)

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                            
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            print(img['filename'])    
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_insts += [img]

    
                        
    return all_insts, seen_labels


def _main_(args):

	config_path = args.conf
	with open(config_path) as config_buffer:    
		config = json.loads(config_buffer.read())
		
	train_ints, train_labels = parse_voc_annotation(config['train']['train_annot_folder'], config['train']['train_image_folder'], config['model']['labels'])
	valid_ints, valid_labels = parse_voc_annotation(config['valid']['valid_annot_folder'], config['valid']['valid_image_folder'], config['model']['labels'])
	
	#print(train_ints)
	print("\n")
	#print(valid_ints)
	print("\n")
	print(len(train_ints))
	print("\n")
	print(len(valid_ints))
	print("\n")
	print(train_labels)
	print(valid_labels)


if __name__ == "__main__":
	argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
	argparser.add_argument('-c', '--conf', help='path to configuration file') 
	
	args = argparser.parse_args()
	_main_(args)