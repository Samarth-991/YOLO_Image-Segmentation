import logging
import os
import re
import cv2
import numpy as np
# from imgaug.augmentables.bbs import BoundingBox
from tensorflow.keras.preprocessing import image


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as err:
            logging.info(err)
    return 0


def write_image(image, filename, dir_path):
    if not os.path.isdir(dir_path):
        make_dir(dir_path)
    cv2.imwrite(os.path.join(dir_path, filename), image)


def write_data(files_list, out_file):
    with open(out_file, 'w+') as outfd:
        for fname in files_list:
            if '.txt' not in fname:
                outfd.write("{}\n".format(fname))
    return 0


def write_objdata(obj_file, n_classes, train_file, val_file, names_file, backup='backup', result_path='result'):
    with open(obj_file, 'w+') as objfd:
        objfd.write("classes = {}\n".format(n_classes))
        objfd.write("train = {}\n".format(train_file))
        objfd.write("val = {}\n".format(val_file))
        objfd.write("names = {}\n".format(names_file))
        objfd.write("backup = {}\n".format(backup))
        objfd.write("results = {}\n".format(result_path))
    return


def parse_config_file(yolo_cfg, nb_classes, network_size=(416, 416), epochs=170000, batch_size=32, learning_rate=1e-7):
    with open(yolo_cfg, 'r+') as yfd:
        configs = list(map(lambda x: x.strip(), yfd.readlines()))
    for i, val in enumerate(configs):
        if re.search('width', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(network_size[0]))
        if re.search('height', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(network_size[1]))
        if re.search('max_batches', val):
            configs[i] = configs[i].replace(configs[i].split('=')[1], str(epochs))
        if re.search('batch', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(batch_size))
        if re.search('learning_rate', val):
            configs[i] = configs[i].replace(val.split('=')[1], str(learning_rate))
        #     configs[i - 4] = configs[i - 4].replace(configs[i - 4].split('=')[1], str(3 * (nb_classes + 5)))

    new_cfg = os.path.join(os.path.dirname(yolo_cfg), os.path.basename(yolo_cfg).replace('.cfg', '_custom.cfg'))
    with open(new_cfg, 'w+') as nfd:
        for cfg_val in configs:
            nfd.writelines("{}\n".format(cfg_val))
    nfd.close()
    return new_cfg


def decode_segmentation_masks(mask, colormap, n_classes=2):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def visulize_colormap(imagefile, maskfile, color_map, n_cls):
    # read image
    image = cv2.imread(imagefile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # read mask
    mask = cv2.imread(maskfile)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
    color_mask = decode_segmentation_masks(mask, color_map, n_classes=n_cls)
    overlay = cv2.addWeighted(image, 0.35, color_mask, 0.65, 0)
    return mask, overlay
