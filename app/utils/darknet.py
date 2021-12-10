#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"
- Set environment variable "DARKNET_PATH" to path darknet lib .so (for Linux)

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
from ctypes import *
import numpy as np
import random
import os

seed = np.random.seed(47)


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value {} not forcing CPU mode".format(tmp))
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(
                winNoGPUdll))
else:
    lib = CDLL(os.path.join(os.getenv('DARKNET_PATH', './'), "libdarknet.so"), RTLD_GLOBAL)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

"""Define MAKE IMAGE"""
make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

"""Define Network Symbols"""
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int
""" Define Network Predict"""
predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

"""Define Network Preiction Symbols"""
predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

"""Define Network Loading Symbols"""
load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

"""Define IMAGE ATTRIBUTES"""
free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

get_network_image = lib.get_network_image
get_network_image.argtypes = [c_void_p]
get_network_image.restype = IMAGE

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

mask_to_rgb = lib.mask_to_rgb
mask_to_rgb.argtypes = [IMAGE]
mask_to_rgb.restype = IMAGE

show_image = lib.show_image
show_image.argtypes = [IMAGE, c_char_p, c_int]

"""Define META ATTRIBUTES"""
load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA
"""Define COPY FROM IMAGE BYTES"""
copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in range(names)}


def label_to_colorid(segmentation_classes):
    color_id = {}
    for i, label in enumerate(segmentation_classes):
        if label == 'background':
            color_id[i] = (0, 0, 0)
        else:
            color = tuple(np.random.choice(range(256), size=3))
            color_id[i] = color
    return color_id


def get_net_attributes(net):
    width = lib.network_width(net)
    height = lib.network_height(net)
    return width, height


def predict_image_mask(img_file, net):
    rgb_image = load_image(img_file.encode('utf-8'), 0, 0)
    width, height = get_net_attributes(net)
    image_sized = letterbox_image(rgb_image, width, height)
    predictions = predict(net, image_sized.data)
    im_pred = get_network_image(net)
    pr_mask = mask_to_rgb(im_pred)
    show_image(im_pred, "test".encode("utf-8"), 0)
    return


def load_network(cfg_file, weights_file, meta_data_file):
    net = load_net(cfg_file.encode("utf-8"), weights_file.encode("utf-8"), 0)
    metadata = load_meta(meta_data_file.encode("utf-8"))
    class_names = [metadata.names[i].decode("utf-8") for i in range(metadata.classes)]
    color_map = label_to_colorid(class_names)
    return net, class_names, color_map
