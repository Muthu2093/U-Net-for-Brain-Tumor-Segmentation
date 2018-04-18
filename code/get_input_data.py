import os
from medpy.io import load
import numpy as np
import cv2 as cv
import itk
from medpy.filter import otsu
from medpy.io import save


PATH = "/Users/muthuvel/Desktop/Independent Study - M Gao/Dataset/BRATS-train/Image_Data"

def pad_image(img, desired_shape=(256, 256)):
    pad_top = 0
    pad_bot = 0
    pad_left = 0
    pad_right = 0
    if desired_shape[0] > img.shape[0]:
        pad_top = int((desired_shape[0] - img.shape[0]) / 2)
        pad_bot = desired_shape[0] - img.shape[0] - pad_top
    if desired_shape[1] > img.shape[1]:
        pad_left = int((desired_shape[1] - img.shape[1]) / 2)
        pad_right = desired_shape[1] - img.shape[1] - pad_left
    img = np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), 'constant')
    assert (img.shape == desired_shape)
    return img


def normalize(img):
    nimg = None
    nimg = cv.normalize(img, nimg, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    nimg = pad_image(nimg, desired_shape=(256, 256))
    return nimg


def load_single_image(path):
    for dir, subdir, files in os.walk(path):
        for file in files:
            if file.endswith(".mha"):
                img, header = load(os.path.join(path, file))
                return img


def create_4_chan_data(t2, t1c, t1, flair, ot):
    ot_layers = []
    flair_layers = []
    t2_layers = []
    t1_layers = []
    t1c_layers = []
    for layer in range(ot.shape[2]):
        ot_layers.append(pad_image(ot[:, :, layer], desired_shape=(256, 256)))
        flair_layers.append(normalize(flair[:, :, layer]))
        t1_layers.append(normalize(t1[:, :, layer]))
        t2_layers.append(normalize(t2[:, :, layer]))
        t1c_layers.append(normalize(t1c[:, :, layer]))

    return np.stack(ot_layers, axis=0), np.stack(flair_layers, axis=0), np.stack(t1_layers, axis=0), \
           np.stack(t1c_layers, axis=0), np.stack(t2_layers, axis=0)


def load_dataset(path):
    train_t2 = []
    train_t1c = []
    train_t1 = []
    train_flair = []
    train_ot = []

    for dir in os.listdir(path):
        if dir == 'HG':
            HG_path = os.path.join(path, 'HG')
            for dir2 in os.listdir(HG_path):
                if dir2 != '.DS_Store':
                    HG_t2 = load_single_image(os.path.join(HG_path, dir2, 'VSD.Brain.XX.O.MR_T2'))
                    HG_t1c = load_single_image(os.path.join(HG_path, dir2, 'VSD.Brain.XX.O.MR_T1c'))
                    HG_flair = load_single_image(os.path.join(HG_path, dir2, 'VSD.Brain.XX.O.MR_Flair'))
                    HG_t1 = load_single_image(os.path.join(HG_path, dir2, 'VSD.Brain.XX.O.MR_T1'))
                    HG_ot = load_single_image(os.path.join(HG_path, dir2, 'VSD.Brain_3more.XX.XX.OT'))
                    assert (HG_ot.shape == HG_flair.shape == HG_t1.shape == HG_t1c.shape == HG_t2.shape)
                    HG_samples = create_4_chan_data(HG_t2, HG_t1c, HG_t1, HG_flair, HG_ot)
                    train_ot.append(HG_samples[0])
                    train_flair.append(HG_samples[1])
                    train_t1.append(HG_samples[2])
                    train_t1c.append(HG_samples[3])
                    train_t2.append(HG_samples[4])
        if dir == 'LG':
            brain_1 = brain_2 = brain_3 = False
            LG_path = os.path.join(path, 'LG')
            for dir3 in os.listdir(LG_path):
                if dir3 != '.DS_Store':
                    LG_t2 = load_single_image(os.path.join(LG_path, dir3, 'VSD.Brain.XX.O.MR_T2'))
                    LG_t1c = load_single_image(os.path.join(LG_path, dir3, 'VSD.Brain.XX.O.MR_T1c'))
                    LG_flair = load_single_image(os.path.join(LG_path, dir3, 'VSD.Brain.XX.O.MR_Flair'))
                    LG_t1 = load_single_image(os.path.join(LG_path, dir3, 'VSD.Brain.XX.O.MR_T1'))

                    brain_1 = os.path.exists(os.path.join(LG_path, dir3, 'VSD.Brain_1more.XX.XX.OT'))
                    brain_2 = os.path.exists(os.path.join(LG_path, dir3, 'VSD.Brain_2more.XX.XX.OT'))
                    brain_3 = os.path.exists(os.path.join(LG_path, dir3, 'VSD.Brain_3more.XX.XX.OT'))
                    if brain_1:
                        LG_ot = load_single_image(os.path.join(LG_path, dir3, 'VSD.Brain_1more.XX.XX.OT'))
                    if brain_2:
                        LG_ot = load_single_image(os.path.join(LG_path, dir3, 'VSD.Brain_2more.XX.XX.OT'))
                    if brain_3:
                        LG_ot = load_single_image(os.path.join(LG_path, dir3, 'VSD.Brain_3more.XX.XX.OT'))

                    assert (LG_ot.shape == LG_flair.shape == LG_t1.shape == LG_t1c.shape == LG_t2.shape)
                    LG_samples = create_4_chan_data(LG_t2, LG_t1c, LG_t1, LG_flair, LG_ot)
                    train_ot.append(LG_samples[0])
                    train_flair.append(LG_samples[1])
                    train_t1.append(LG_samples[2])
                    train_t1c.append(LG_samples[3])
                    train_t2.append(LG_samples[4])
    # Stacking all individual layers
    train_ot = np.vstack(train_ot)
    train_flair = np.vstack(train_flair)
    train_t1 = np.vstack(train_t1)
    train_t1c = np.vstack(train_t1c)
    train_t2 = np.vstack(train_t2)
    assert (train_ot.shape == train_flair.shape == train_t1.shape == train_t1c.shape == train_t2.shape)
    return [train_t2, train_t1c,train_t1,train_flair,train_ot]


##img=mpimg.imread('your_image.png')
[train_t2, train_t1c,train_t1,train_flair,train_ot] = load_dataset(PATH)

