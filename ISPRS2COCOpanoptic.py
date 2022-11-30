import numpy as np
import tifffile
from pathlib import Path
from skimage import io, measure
import json
import shutil
converter = __import__('2channels2panoptic_coco_format').converter


# the list of categories in Vaihingen/Potsdam
categories_list = [
    {"name": "impervious_surfaces", "color": (255,255,255), 'id':1},
    {"name": "building", "color": (0, 0, 255), 'id':2},
    {"name": "low_vegetation", "color": (0, 255, 255), 'id':3},
    {"name": "tree", "color": (0, 255, 0), 'id':4},
    {"name": "car", "color": (255, 255, 0), 'id':5},
    {"name": "clutter/background", "color": (255, 0, 0), 'id':6}
]

# the list of not instances
notthing_ids = [1,3,6]

def create_panoptic_coco_categories_list(categories_list=categories_list, notthing_ids=notthing_ids):
    '''
    create categories list in panoptic coco format.
    :param categories_list: the list of categories in Vaihingen/Potsdam, e.g. line12
    :param notthing_ids: the list of not instances, e.g. line22
    :return panoptic_coco_categories_list: categories list in panoptic coco format 
    '''
    panoptic_coco_categories_list = []

    for category in categories_list:
        isthing = 1
        if category['id'] in notthing_ids:
            isthing = 0
        
        panoptic_coco_category = {
            "supercategory": category["name"],
            "color": category["color"],
            "isthing": isthing,
            "id": category['id'],
            "name": category["name"]
        }
        panoptic_coco_categories_list.append(panoptic_coco_category)
    
    return panoptic_coco_categories_list

def concat_num(num1, num2):
    if int(num2)<10:
        num2 = "0" + num2
    return num1 + num2

def computer_image_id(image_file):
    '''
    computer image id.
    :param image_file: pathlib.Path
    :return : id of the images

    example:
    image name : top_mosaic_09cm_area1.tif    id : 1 = int('0'+'1')    
    # '0' is the number of data set Vaihingen, '1' is the number part of the name that distinguishes the image.

    image name : top_potsdam_2_10_RGBIR.tif   id : 1210 = int('1'+'2'+'10') 
    # '1' is the number of data set Potsdam, '2','10' is the number part of the name that distinguishes the image.
    '''
    name_part_list = image_file.stem.split('_')
    if name_part_list[1] == 'mosaic':
        return int(concat_num('0', name_part_list[3][4:]))
    elif name_part_list[1] == 'potsdam':
        return int(concat_num('1', concat_num(image_file.stem.split('_')[2],image_file.stem.split('_')[3])))
    else:
        pass

def get_image_file_name(image_file):
    '''
    get new imagefile name, because the coco format requires the image and label file names to be same.
    :param image_file: pathlib.Path
    :return : new imagefile name

    example:
    image file name : top_mosaic_09cm_area1.tif    new imagefile name : top_mosaic_09cm_area1.tif
    image name : top_potsdam_2_10_RGBIR.tif    new imagefile name : top_potsdam_2_10.tif
    '''
    name_parts_list = image_file.stem.split('_')
    if name_parts_list[1] == 'mosaic':
        return image_file.stem
    elif name_parts_list[1] == 'potsdam':
        return '_'.join(image_file.stem.split('_')[:-1])
    else:
        pass


def create_panoptic_coco_images_list(image_files_list):
    '''
    read image information and save it in coco format.
    :param image_files_list: list
    :return : list
    '''
    panoptic_coco_images_list = []

    for image_file in image_files_list:
        
        image = tifffile.imread(str(image_file))
        height, width, _ = image.shape
        
        image_name_str = get_image_file_name(image_file)

        panoptic_coco_image = {
            "file_name": image_name_str+'.tif',
            "height": height,
            "width": width,
            "id": computer_image_id(image_file)
        }
        panoptic_coco_images_list.append(panoptic_coco_image)
    return panoptic_coco_images_list

def com_encode_label(label):
    label = label.astype(np.int64)
    return 255 * 255 * label[:,:,0] + 255 * label[:,:,1] + label[:,:,2]

def com_encode_color(color):
    return 255 * 255 * color[0] + 255 * color[1] + color[2]

def create_2channels_format_label(label_file, categories_list = categories_list, notthing_ids = notthing_ids):
    '''
    read label information and save it in 2 channels format. Each segment of 2 channels label is defined by two labels: (1) semantic category label and (2) instance ID label.
    :param image_files_list: list
    :return : list, num
    '''
    label = tifffile.imread(label_file)
    encode_label = com_encode_label(label)
    label_1_band = np.zeros(label.shape[0:2])
    for category in categories_list:
        label_1_band = np.where(encode_label==com_encode_color(category["color"]), category['id'], label_1_band)
    inss_label, num_features = measure.label(label_1_band,return_num=True)
    
    label_2channels = np.zeros(label.shape)
    label_2channels[:,:,0], label_2channels[:,:,1] = label_1_band, inss_label

    for category in categories_list:
        if category['id'] in notthing_ids:
            label_2channels[:,:,1] = np.where(encode_label==com_encode_color(category["color"]), 0, label_2channels[:,:,1])

    return label_2channels.astype(np.uint8), num_features

def cut_images_per(images_list, per_train_images):
    '''
    divide training set and verification set by proportion
    :param images_list: list, images list
    :param per_train_images: float, Percentage of training set
    :return train_images_list, val_images_list: list,list
    '''
    train_images_num = int(per_train_images * len(images_list))
    train_images_list = images_list[0:train_images_num]
    val_images_list = panoptic_coco_images_list[train_images_num:]
    return train_images_list, val_images_list

def cut_images_id(images_list, id_list):
    '''
    divide training set and verification set by id
    :param images_list: list, images list
    :param per_train_images: list, e.g.[{1,1210},{1211}]
    :return train_images_list, val_images_list: list,list
    '''
    train_images_list = [x for x in images_list if x['id'] in id_list[0]]
    val_images_list = [x for x in images_list if x['id'] in id_list[1]]
    return train_images_list, val_images_list



def create_images_info(images_folder, cut_parameter):
    '''
    create images list in panoptic coco format.
    :param images_folder: str, images list
    :param cut_parameter: the param per_train_images of def cut_images_per() or id_list of def cut_images_id()
    :return train_json, val_json: list,list
    '''
    images_folder = Path(images_folder)
    image_files_list = [x for x in images_folder.iterdir() if x.suffix == '.tif']

    panoptic_coco_images_list = create_panoptic_coco_images_list(image_files_list)
    category_list = create_panoptic_coco_categories_list()

    if type(cut_parameter) == list:
        train_images_list, val_images_list = cut_images_id(panoptic_coco_images_list, cut_parameter)
    elif type(cut_parameter) == float:
        train_images_list, val_images_list = cut_images_per(panoptic_coco_images_list, cut_parameter)
    
    train_json = {
        "info":{},
        "images":train_images_list,
        "categories":category_list
    }

    val_json = {
        "info":{},
        "images":val_images_list,
        "categories":category_list
    }

    return train_json, val_json

def gen_labels_2channels_folder(labels_folder, temp_folder):
    '''
    save 2 channels labels in temp folder.
    :param labels_folder: str
    :param temp_folder: str
    '''
    labels_folder = Path(labels_folder)
    label_files_list = [x for x in labels_folder.iterdir() if x.suffix == '.tif']
    for label_file in label_files_list:
        label_2channels, num_features = create_2channels_format_label(label_file)
        label_file_str = get_image_file_name(label_file)
        filename = str(Path(temp_folder) / label_file_str) + '.png'
        print('Label {name} have {num} instance(s).'.format(name=label_file.name,num=num_features))
        io.imsave(filename, label_2channels)

def gen_panoptic_coco_format(labels_2channels_folder, images_json_file, categories_json_file, predictions_json_file):
    '''
    save panoptic coco format annotations in folder.
    :param labels_2channels_folder: str
    :param images_json_file: str
    :param categories_json_file: str
    :param predictions_json_file: str
    '''
    source_folder = labels_2channels_folder
    converter(source_folder, images_json_file, categories_json_file,
              segmentations_folder = None, predictions_json_file=predictions_json_file,
              VOID=0)

def gen_images_info_json(images_folder, temp_folder, cut_parameter):
    '''
    save images list in panoptic coco format.
    :param images_folder: str
    :param temp_folder: str
    :param cut_parameter: the param per_train_images of def cut_images_per() or id_list of def cut_images_id()
    '''
    train_json, val_json = create_images_info(images_folder, cut_parameter)
    temp_folder = Path(temp_folder)
    with open(temp_folder / 'train_images.json', 'w') as f:
        json.dump(train_json, f)
    print('Train images json file has been generated! There have {num} train image(s).'.format(num=len(train_json["images"])))
    with open(temp_folder / 'val_images.json', 'w') as f:
        json.dump(val_json, f)
    print('Val images json file has been generated! There have {num} val image(s).'.format(num=len(val_json["images"])))

def copy_images(original_folder, new_folder, images_json_file):
    '''
    copy images from original folder to new folder by images json.
    :param original_folder: str
    :param new_folder: str
    :param images_json_file: str
    '''
    original_folder = Path(original_folder)
    original_files_list = [x for x in original_folder.iterdir() if x.suffix == '.tif']
    new_folder = Path(new_folder)
    for original_file in original_files_list:
        with open(images_json_file, 'r') as f:
            images_json = json.load(f)
        images_file_list = [x['file_name'] for x in images_json["images"]]
        original_file_str = get_image_file_name(original_file)
        if original_file_str+'.tif' in images_file_list:
            new_file = str(new_folder / original_file_str) + '.tif'
            shutil.copyfile(original_file, new_file)
            print(original_file_str+'.tif havs been copied!')

def gen_dataset(
    images_folder, labels_folder, coco_dataset_folder,
    cut_parameter,
    images_json_file = None,
    categories_json_file = None,
    labels_2channels_folder = None,
    ):
    '''
    save the coco format Vaihingen/Potsdam dataset.
    :param images_folder: str
    :param labels_folder: str
    :param coco_dataset_folder: str, save generated dataset.
    :images_json_file str, if have this file, program will not generate
    :categories_json_file str, if have this file, program will not generate
    :labels_2channels_folder str, if have this file, program will not generate
    '''

    coco_dataset_folder = Path(coco_dataset_folder)
    if not images_json_file:
        print('\n'+'Generate json file!'.center(60,'=')+'\n')
        images_json_file = coco_dataset_folder / 'temp_images_json'
        images_json_file = Path(images_json_file)
        images_json_file.mkdir(parents=True, exist_ok=True)
        gen_images_info_json(images_folder, str(images_json_file), cut_parameter)

    if not categories_json_file:
        categories_json = create_panoptic_coco_categories_list()
        categories_json_file = str(images_json_file / 'categories.json')
        with open(categories_json_file, 'w') as f:
            json.dump(categories_json, f)
        
        print('Categories json file has been generated!')

    if not labels_2channels_folder:
        print('\n'+'Generate two channels labels!'.center(60,'=')+'\n')
        labels_2channels_folder = coco_dataset_folder / 'temp_labels_2channels'
        labels_2channels_folder.mkdir(parents=True, exist_ok=True)
        gen_labels_2channels_folder(labels_folder, str(labels_2channels_folder))

    print('\n'+'Generate train set!'.center(60,'=')+'\n')

    (coco_dataset_folder / 'annotations').mkdir(parents=True, exist_ok=True)

    train_images_json_file = str(Path(images_json_file) / 'train_images.json')
    predictions_train_json_file = str(coco_dataset_folder / 'annotations' / 'panoptic_train')+'.json'
    gen_panoptic_coco_format(str(labels_2channels_folder), train_images_json_file, categories_json_file, predictions_train_json_file)

    print('\n'+'Generate val set!'.center(60,'=')+'\n')
    val_images_json_file = str(Path(images_json_file) / 'val_images.json')
    predictions_val_json_file = str(coco_dataset_folder / 'annotations' / 'panoptic_val')+'.json'
    gen_panoptic_coco_format(labels_2channels_folder, val_images_json_file, categories_json_file, predictions_val_json_file)

    print('\n'+'Copy images from original folder!'.center(60,'=')+'\n')
    train_images_folder = coco_dataset_folder / 'train'
    train_images_folder.mkdir(parents=True,exist_ok=True)
    copy_images(images_folder, str(train_images_folder), train_images_json_file)
    val_images_folder = coco_dataset_folder / 'val'
    val_images_folder.mkdir(parents=True,exist_ok=True)
    copy_images(images_folder, str(val_images_folder), val_images_json_file)

    print('\n'+'Finish!'.center(60,'=')+'\n')



if __name__ == '__main__':

    images_folder = r'Potsdam\4_Ortho_RGBIR'
    labels_folder = r'Potsdam\5_Labels_all'
    coco_dataset_folder = r'Potsdam\Potsdam_coco_format'
    id_list = [
        {1210,1211,1212,1213,1214,
        1310,1311,1312,1313,1314,
        1410,1411,1412,1413,1414,1415,
        1510,1511,1512,1513,1514,1515,
        1607,1608,1609,1610,1611,1612,1613,1614,1615,
        1707,1708,1709,1710,1711,1712,1713},
        {1210}] # all id about Potsdam images
    gen_dataset(
    images_folder, labels_folder, coco_dataset_folder,
    cut_parameter = id_list,
    )

    # images_folder = r'Vaihingen\top'
    # labels_folder = r'Vaihingen\ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE'
    # coco_dataset_folder = r'Vaihingen\Vaihingen_coco_format'
    # id_list = [
    #     {1,2,3,4,5,6,7,8,
    #     10,11,12,13,14,15,16,17,
    #     20,21,22,23,24,26,27,28,29,
    #     30,31,32,33,34,35,37,38},
    #     {1}] # all id about Vaihingen images
    # gen_dataset(
    # images_folder, labels_folder, coco_dataset_folder,
    # cut_parameter = id_list,
    # )
