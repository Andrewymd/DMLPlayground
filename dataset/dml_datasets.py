import mxnet as mx
import numpy as np
from itertools import cycle

from mxnet.image import ForceResizeAug
from mxnet.io import DataDesc

from models.loss import pairwise_distance


def get_datasets(train_image_files, test_image_files, boxes, data_shape, use_crops, scale_image_data, use_aug=True,
                 with_proxy=False, **kwargs):
    """Return training and testing datasets"""

    return (ImageDataset(train_image_files, data_shape, boxes if use_crops else None, use_aug=use_aug,
                         with_proxy=with_proxy, scale_image_data=scale_image_data, **kwargs),
            ImageDataset(test_image_files, data_shape, boxes if use_crops else None, use_aug=False,
                         with_proxy=with_proxy, scale_image_data=scale_image_data, **kwargs))


def get_iterators(train_image_files, test_image_files, boxes, batch_k, batch_size, data_shape, use_crops,
                  scale_image_data, **kwargs):
    """Return training and testing iterators."""

    return (
        ImageDatasetIterator(train_image_files, test_image_files, boxes, batch_k, batch_size, data_shape, use_crops,
                             is_train=True, scale_image_data=scale_image_data, **kwargs),
        ImageDatasetIterator(train_image_files, test_image_files, boxes, batch_k, batch_size, data_shape, use_crops,
                             is_train=False, scale_image_data=scale_image_data, **kwargs))


def get_npairs_iterators(train_image_files, test_image_files, boxes, batch_size, data_shape, test_batch_size, use_crops,
                         scale_image_data, same_image_sampling):
    """Return training and testing npairs iterators."""

    return (NPairsIterator(train_image_files, test_image_files, boxes, batch_size, data_shape, use_crops=use_crops,
                           is_train=True, scale_image_data=scale_image_data,
                           same_image_sampling=same_image_sampling),
            NPairsIterator(train_image_files, test_image_files, boxes, batch_size, data_shape, use_crops=use_crops,
                           test_batch_size=test_batch_size, is_train=False, scale_image_data=scale_image_data,
                           same_image_sampling=0))


def get_prototype_iterators(train_image_files, test_image_files, boxes, Nc, Ns, Nq, data_shape, test_batch_size,
                            use_crops, scale_image_data):
    """Return training and testing prototype iterators."""

    return (PrototypeIterator(train_image_files, test_image_files, boxes, Nc, Ns, Nq, data_shape, use_crops=use_crops,
                              is_train=True, scale_image_data=scale_image_data),
            PrototypeIterator(train_image_files, test_image_files, boxes, Nc, Ns, Nq, data_shape, use_crops=use_crops,
                              test_batch_size=test_batch_size, is_train=False, scale_image_data=scale_image_data))


class CircularIterator:
    """
    Circular iterator over a dataset
    """

    def __init__(self, dataset):
        self.pool = cycle(dataset)

    def next_batch(self, b):
        return [self.pool.next() for _ in range(b)]

    def next(self):
        return self.pool.next()


color_mean = [123.68, 116.779, 103.939]
color_std = [58.393, 57.12, 57.375]


def transform(data, resize_to, output_shape, use_aug, box, scale):
    """Crop and normalize an image nd array."""
    if box is not None:
        x, y, w, h = box
        data = data[y:min(y + h, data.shape[0]), x:min(x + w, data.shape[1])]

    data = data.astype('float32')

    if use_aug:
        augmeters = mx.image.CreateAugmenter(data_shape=(3, output_shape, output_shape),
                                             #resize=resize_to,
                                             rand_resize=True,
                                             rand_crop=True,
                                             rand_mirror=True,
                                             mean=np.array(color_mean),
                                             std=np.array(color_std) if scale else None,
                                             inter_method=10)
        if resize_to is not None:
            augmeters = [ForceResizeAug((resize_to, resize_to))] + augmeters
        for aug in augmeters:
            data = aug(data)
    else:
        augmeters = mx.image.CreateAugmenter(data_shape=(3, output_shape, output_shape),
                                             resize=256 if resize_to is None else resize_to,
                                             mean=np.array(color_mean),
                                             std=np.array(color_std) if scale else None)
        if resize_to is not None:
            augmeters[0] = ForceResizeAug((resize_to, resize_to))
        for aug in augmeters:
            data = aug(data)

    data = mx.nd.transpose(data, (2, 0, 1))

    # If image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = mx.nd.tile(data, (3, 1, 1))
    return data


class ImageDataset(mx.gluon.data.Dataset):
    def __init__(self, class_data, data_shape, boxes, use_aug, scale_image_data, max_images_per_class=-1,
                 with_proxy=False, resize_img=None):
        super(ImageDataset, self).__init__()
        self._num_classes = len(class_data)
        self._data_shape = data_shape
        self._use_aug = use_aug
        self._boxes = boxes
        self._scale_image_data = scale_image_data
        self._with_proxy = with_proxy
        self._class_mapping = None
        self._num_remapped_classes = None
        self._resize_img = resize_img

        self._data = []
        for i, image_list in enumerate(class_data):
            images = [(path, i) for path in image_list]
            self._data += images if max_images_per_class < 0 else images[:max_images_per_class]

    def set_class_mapping(self, mapping, num_classes):
        self._class_mapping = mapping
        self._num_remapped_classes = num_classes

    def __getitem__(self, idx):
        path, c = self._data[idx]

        if self._class_mapping is not None:
            c = self._class_mapping[c]

        img_arr = mx.image.imread(path, flag=1)
        if self._boxes:
            img_arr = transform(img_arr, self._resize_img, self._data_shape, self._use_aug, self._boxes[path],
                                self._scale_image_data)
        else:
            img_arr = transform(img_arr, self._resize_img, self._data_shape, self._use_aug, None, self._scale_image_data)

        if self._with_proxy:
            if self._num_remapped_classes is not None:
                ar = np.arange(0, self._num_remapped_classes)
            else:
                ar = np.arange(0, self._num_classes)
            negatives = ar[ar != c]
            return img_arr, c, negatives

        return img_arr, c

    def __len__(self):
        return len(self._data)

    def num_classes(self):
        return self._num_classes


class ImageDatasetIterator(mx.io.DataIter):
    """Iterator for an image dataset. Supports custom batch samples.
    """

    def __init__(self, train_images, test_images, boxes, batch_k, batch_size, data_shape, use_crops, is_train,
                 scale_image_data, resize_image=256, batchify=True):
        super(ImageDatasetIterator, self).__init__(batch_size)
        self._data_shape = data_shape
        self._batch_size = batch_size
        self._batch_k = batch_k
        self.is_train = is_train
        self.num_train_classes = len(train_images)
        self.num_test_classes = len(test_images)
        self._resize_image = resize_image
        self._batchify = batchify

        self._train_images = train_images
        self._boxes = boxes
        self._test_count = 0
        self._use_crops = use_crops
        self._scale_image_data = scale_image_data

        self._test_data = [(f, l) for l, files in enumerate(test_images) for f in files]

        self.n_test = len(self._test_data)

        self._train_images_it = [CircularIterator(x) for x in self._train_images]

        if batch_k is None:
            self._flattened_training_data = [(item, label) for label, class_images in enumerate(self._train_images) for
                                             item in class_images]

    def num_training_images(self):
        return sum([len(x) for x in self._train_images])

    def num_classes(self):
        return self.num_train_classes if self.is_train else self.num_test_classes

    def get_image(self, img, is_train):
        """Load and transform an image."""
        img_arr = mx.image.imread(img)
        img_arr = transform(img_arr, self._resize_image, self._data_shape, is_train,
                            self._boxes[img] if self._use_crops else None, self._scale_image_data)
        return img_arr.expand_dims(0)

    def sample_train_batch(self):
        """Sample a training batch (data and label)."""
        batch = []
        labels = []
        num_classes = self._batch_size // self._batch_k  # initial number of classes to be selected
        expected_batch_size = num_classes * self._batch_k

        # we choose the first set of classes
        sampled_classes = np.random.choice(self.num_train_classes, num_classes, replace=False)

        # verify we have enough samples fill up the batch
        num_images_per_samples_class = [min(len(self._train_images[x]), self._batch_k) for x in sampled_classes]

        # add more classes until batch is full
        while sum(num_images_per_samples_class) < expected_batch_size:
            # sample a new class and add it to the existing list
            new_sample_class = np.random.choice(np.delete(np.arange(self.num_train_classes), sampled_classes), 1,
                                                replace=False)
            sampled_classes = np.concatenate((sampled_classes, new_sample_class))
            # recompute number of images
            num_images_per_samples_class = [min(len(self._train_images[x]), self._batch_k) for x in sampled_classes]

        # collect images
        for c in sampled_classes:
            img_fnames = np.random.choice(self._train_images[c],
                                          min(self._batch_k, len(self._train_images[c])), replace=False)
            batch += [self.get_image(img_fname, is_train=True) for img_fname in img_fnames]
            labels += [c for _ in range(self._batch_k)]

        # remove overflow
        batch = batch[:expected_batch_size]
        labels = labels[:expected_batch_size]

        return mx.nd.concatenate(batch, axis=0), labels

    def sample_proxy_train_batch(self, sampled_classes, chose_classes_randomly=False, distances=None):
        """Sample a training batch for proxy training (data, label and negative labels) from the given classes."""
        batch = []
        labels = []
        negative_labels = []
        if sampled_classes is not None:
            if isinstance(sampled_classes, list):
                num_groups = len(sampled_classes)
                expected_batch_size = num_groups * self._batch_k
            else:
                num_groups = sampled_classes
                expected_batch_size = num_groups * self._batch_k
                if distances is None:
                    # Sample classes randomly
                    sampled_classes = np.random.choice(self.num_train_classes, num_groups, replace=False)
                else:
                    # Sample classes based on class distances
                    distances = distances.asnumpy()
                    mask = np.tril(distances, 0)
                    distances[mask == 0] = 1e10  # CxC

                    num_classes = distances.shape[0]
                    distances_flat = np.reshape(distances, -1)
                    asorted_dist = np.argsort(distances_flat)
                    first_null = (distances_flat.size - num_classes) // 2
                    asorted_dist = asorted_dist[:first_null]  # (C*(C -1)) // 2
                    probs = 1 / distances_flat[asorted_dist]
                    probs = (probs / np.sum(probs))

                    pairs = np.random.choice(np.arange(0, probs.size), probs.size, p=probs, replace=False)
                    pairs_indices_flat = asorted_dist[pairs]
                    pairs_indices = np.unravel_index(pairs_indices_flat, distances.shape)

                    sampled_classes = set()
                    counter = 0
                    while len(sampled_classes) < num_groups:
                        pair_idx = pairs_indices[0][counter], pairs_indices[1][counter]
                        sampled_classes.add(pair_idx[0])
                        if len(sampled_classes) < num_groups:
                            sampled_classes.add(pair_idx[1])

                        counter += 1

                # make sure we have enough data
                num_images_per_samples_class = [min(len(self._train_images[x]), self._batch_k) for x in sampled_classes]

                # add more classes until batch is full
                while sum(num_images_per_samples_class) < expected_batch_size:
                    # sample a new class and add it to the existing list
                    new_sample_class = np.random.choice(np.delete(np.arange(self.num_train_classes), sampled_classes),
                                                        1, replace=False)
                    sampled_classes = np.concatenate((sampled_classes, new_sample_class))
                    # recompute number of images
                    num_images_per_samples_class = [min(len(self._train_images[x]), self._batch_k) for x in
                                                    sampled_classes]

            for c in sampled_classes:
                if chose_classes_randomly:
                    img_fnames = np.random.choice(self._train_images[c],
                                                  min(self._batch_k, len(self._train_images[c])), replace=False)
                else:
                    img_fnames = self._train_images_it[c].next_batch(self._batch_k)
                batch += [self.get_image(img_fname, is_train=True) for img_fname in img_fnames]
                labels += [c for _ in range(self._batch_k)]

                ar = np.arange(0, self.num_train_classes)
                negatives = ar[ar != c]
                negative_labels += [mx.nd.array(negatives).expand_dims(0) for _ in range(self._batch_k)]

            # remove overflow
            batch = batch[:expected_batch_size]
            labels = labels[:expected_batch_size]
            negative_labels = negative_labels[:expected_batch_size]

        else:
            chosen_data_idx = np.random.choice(range(len(self._flattened_training_data)), self.batch_size,
                                               replace=False)
            batch += [self.get_image(self._flattened_training_data[idx][0], is_train=True) for idx in chosen_data_idx]
            labels += [self._flattened_training_data[idx][1] for idx in chosen_data_idx]
            for l in labels:
                ar = np.arange(0, self.num_train_classes)
                negatives = ar[ar != l]
                negative_labels.append(mx.nd.array(negatives).expand_dims(0))

        return mx.nd.concatenate(batch, axis=0), labels, mx.nd.concatenate(negative_labels, axis=0)

    def get_test_batch(self, batch_size=None):
        """Sample a testing batch (data and label)."""
        if batch_size is None:
            batch_size = self._batch_size

        data, labels = zip(
            *[self._test_data[(self._test_count * batch_size + i) % len(self._test_data)] for i in range(batch_size)])
        data = [self.get_image(x, is_train=False) for x in data]
        return mx.nd.concatenate(data, axis=0), labels

    def reset(self):
        """Reset an iterator."""
        self._test_count = 0

    def next(self):
        """Return a batch."""
        if self.is_train:
            data, labels = self.sample_train_batch()
        else:
            if self._test_count * self._batch_size < len(self._test_data):
                data, labels = self.get_test_batch()
                self._test_count += 1
            else:
                self._test_count = 0
                raise StopIteration
        if self._batchify:
            return mx.io.DataBatch(data=[data], label=[labels])
        return data, labels

    def next_proxy_sample(self, sampled_classes, chose_classes_randomly=False, proxies=None):
        if self.is_train:
            if proxies is not None:
                distances = pairwise_distance(mx.nd, proxies)  # CxC
            else:
                distances = None
            data, labels, negative_labels = self.sample_proxy_train_batch(sampled_classes, chose_classes_randomly, distances)
            if self._batchify:
                return mx.io.DataBatch(data=[data, labels, negative_labels], label=[])
            else:
                return data, labels, negative_labels
        else:
            return self.next()

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator."""
        real_batch_size = (self._batch_size // self._batch_k) * self._batch_k
        return [
            DataDesc('data', (real_batch_size, 3, self._data_shape, self._data_shape), np.float32)
        ]

    @property
    def provide_label(self):
        real_batch_size = (self._batch_size // self._batch_k) * self._batch_k
        return [
            DataDesc('label', (real_batch_size,), np.int64)
        ]


class NPairsIterator(ImageDatasetIterator):
    """NPairs data iterator for the CUB200-2011 dataset.
    """

    def __init__(self, train_images, test_images, boxes, batch_size, data_shape, use_crops, is_train, scale_image_data,
                 test_batch_size=128, resize_image=256, same_image_sampling=0):
        if is_train:
            self.N = batch_size // 2
            batch_size = self.N * 2
        else:
            batch_size = test_batch_size
        super(NPairsIterator, self).__init__(train_images, test_images, boxes, 0, batch_size, data_shape, use_crops,
                                             is_train, scale_image_data, resize_image=resize_image)

        self._test_batch_size = test_batch_size
        self._same_image_sampling = same_image_sampling

    def sample_train_batch(self):
        """Sample a training batch (data and label)."""
        anchors = []
        positives = []
        labels = []

        sampled_classes = np.random.choice(self.num_train_classes, self.N, replace=False)
        for i in range(self.N):
            label = sampled_classes[i]
            if np.random.random_sample() < self._same_image_sampling:
                img_fnames = np.random.choice(self._train_images[label], 1, replace=False)
                img_fnames = np.repeat(img_fnames, 2)
            else:
                img_fnames = np.random.choice(self._train_images[label], 2, replace=False)
            anchors.append(self.get_image(img_fnames[0], is_train=True))
            positives.append(self.get_image(img_fnames[1], is_train=True))
            labels.append(label)

        return mx.nd.concatenate(anchors, axis=0), mx.nd.concatenate(positives, axis=0), labels

    def next(self):
        """Return a batch."""
        if self.is_train:
            anchors, positives, labels = self.sample_train_batch()
            return anchors, positives, labels
        else:
            if self._test_count * self._test_batch_size < len(self._test_data):
                data, labels = self.get_test_batch(self._test_batch_size)
                self._test_count += 1
            else:
                self._test_count = 0
                raise StopIteration
        return mx.io.DataBatch(data=[data], label=[labels])


class PrototypeIterator(ImageDatasetIterator):
    """Prototype networks data iterator.
    """

    def __init__(self, train_images, test_images, boxes, nc, ns, nq, data_shape, use_crops, is_train, scale_image_data,
                 test_batch_size=128, resize_image=256):
        batch_size = nc * (ns + nq)
        super(PrototypeIterator, self).__init__(train_images, test_images, boxes, 0, batch_size, data_shape, use_crops,
                                                is_train, scale_image_data, resize_image=resize_image)
        self.nc = nc
        self.ns = ns
        self.nq = nq
        self._test_batch_size = test_batch_size

    def next(self):
        """Return a batch."""
        if self.is_train:
            supports, queries, labels = self.sample_train_batch()
            return supports, queries, labels
        else:
            if (self._test_count * self._test_batch_size) < len(self._test_data):
                data, labels = self.get_test_batch(self._test_batch_size)
                self._test_count += 1
            else:
                self._test_count = 0
                raise StopIteration
        return mx.io.DataBatch(data=[data], label=[labels])

    def sample_train_batch(self):
        """Sample a training batch (data and label)."""
        sampled_classes = np.random.choice(self.num_train_classes, self.nc, replace=False)
        sampled_classes.sort()

        supports = []  # <Nc x Ns x I>
        queries = []  # <Nc x Nq x I>
        labels = []  # <Nc x 1>

        for i in range(sampled_classes.shape[0]):
            label = sampled_classes[i]
            img_fnames = np.random.choice(self._train_images[label], self.nq + self.ns, replace=False)
            # img_fnames = self.train_image_files[label].next_batch(self.nq + self.ns)
            images = [self.get_image(img_fname, is_train=True) for img_fname in img_fnames]
            support_set = images[:self.ns]
            query_set = images[self.ns:]

            supports.append(mx.nd.concatenate(support_set, axis=0).expand_dims(0))
            queries.append(mx.nd.concatenate(query_set, axis=0).expand_dims(0))
            labels.append(label)

        return mx.nd.concatenate(supports, axis=0), mx.nd.concatenate(queries, axis=0), labels
