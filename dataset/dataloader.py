from mxnet.gluon.data import Dataset
from math import ceil
from os import listdir
from os.path import join, isdir

from common.utils import isjpeg


def get_minicub_data(data_path, train_split=0.5, **kwargs):
    classes = [x for x in listdir(data_path) if isdir(join(data_path, x))]

    classes.sort()
    num_training = int(len(classes) * train_split)

    image_paths = []

    for c in classes:
        images = [join(data_path, str(c), x) for x in listdir(join(data_path, str(c))) if isjpeg(x)]
        image_paths.append(images)

    train_image_files = image_paths[:num_training]
    test_image_files = image_paths[num_training:]

    return train_image_files, test_image_files, None


def get_cub_data(data_path, num_train_classes=100, **kwargs):
    boxes = {}
    num_test_classes = 200 - num_train_classes
    train_image_files = [[] for _ in range(num_train_classes)]
    test_image_files = [[] for _ in range(num_test_classes)]

    with open(join(data_path, 'images.txt'), 'r') as f_img, \
            open(join(data_path, 'image_class_labels.txt'), 'r') as f_label, \
            open(join(data_path, 'bounding_boxes.txt'), 'r') as f_box:
        for line_img, line_label, line_box in zip(f_img, f_label, f_box):
            fname = join(data_path, 'images', line_img.strip().split()[-1])
            label = int(line_label.strip().split()[-1]) - 1
            box = [int(float(v)) for v in line_box.split()[-4:]]
            boxes[fname] = box

            if label < num_train_classes:
                train_image_files[label].append(fname)
            else:
                test_image_files[label - num_train_classes].append(fname)

    return train_image_files, test_image_files, boxes


def get_cars196_data(data_path, train_split=0.5, **kwargs):
    boxes = None  # CARS196 has no bbox information
    classes = [int(x) for x in listdir(data_path) if isdir(join(data_path, x)) and x.isdigit()]

    classes.sort()
    num_training = int(len(classes) * train_split)

    image_paths = []
    for c in classes:
        images = [join(data_path, str(c), x) for x in listdir(join(data_path, str(c))) if isjpeg(x)]
        image_paths.append(images)

    train_image_files = image_paths[:num_training]
    test_image_files = image_paths[num_training:]

    return train_image_files, test_image_files, boxes


def get_sop_data(data_path, train_split=0.5, **kwargs):
    categories = [join(data_path, x) for x in listdir(data_path) if isdir(join(data_path, x))]

    all_data = {}
    for category in categories:
        img_files = [x for x in listdir(category) if isjpeg(x)]
        for img in img_files:
            classname, imgid = tuple(img.split('_'))
            if classname not in all_data:
                all_data[classname] = []
            all_data[classname].append(join(category, img))

    all_data = list(all_data.values())

    num_training = int(ceil(len(all_data) * train_split))

    train_image_files = all_data[:num_training]
    test_image_files = all_data[num_training:]

    return train_image_files, test_image_files, None


def get_imagenet_data(data_path, train_split=0.5, **kwargs):
    categories = [join(data_path, x) for x in listdir(data_path) if isdir(join(data_path, x))]
    all_data = {}
    for category in categories:
        img_files = [x for x in listdir(category) if isjpeg(x)]
        for img in img_files:
            classname, imgid = tuple(img.split('_'))
            if classname not in all_data:
                all_data[classname] = []
            all_data[classname].append(join(category, img))
    all_data = list(all_data.values())
    num_training = int(ceil(len(all_data) * train_split))
    train_image_files = all_data[:num_training]
    test_image_files = all_data[num_training:]
    print ('IMAGENET: train %d | test: %d' % (sum(len(i) for i in train_image_files), sum(len(j) for j in test_image_files)))
    return train_image_files, test_image_files, None


class DatasetIterator(Dataset):
    """
    This class allows mxnet dataloaders to load parallel batches on iterators
    """
    def __init__(self, data_iterator, length, next_call='next', call_params=None):
        self.data_iterator = data_iterator
        self._length = length
        self._next_call = next_call
        self._call_params = call_params if (call_params is not None) else {}

    def __getitem__(self, item):
        return getattr(self.data_iterator, self._next_call)(**self._call_params)

    def __len__(self):
        return self._length

    def num_classes(self):
        return self.data_iterator.num_classes()
