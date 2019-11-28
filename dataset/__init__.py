import dataset.dataloader
import dataset.dml_datasets

dataloaders = {
    'miniCUB': dataloader.get_minicub_data,
    'CUB': dataloader.get_cub_data,
    'CARS196': dataloader.get_cars196_data,
    'SOP': dataloader.get_sop_data,
    'IMAGENET': dataloader.get_imagenet_data,
}


def get_dataset(dataset_name, dataset_path, **kwargs):
    if dataset_name not in dataloaders:
        raise RuntimeError('Unknown dataset name: %s' % dataset_name)

    train_image_files, test_image_files, boxes = dataloaders[dataset_name](dataset_path, **kwargs)
    return dml_datasets.get_datasets(train_image_files, test_image_files, boxes, **kwargs)


def get_dataset_iterator(dataset_name, dataset_path, **kwargs):
    if dataset_name not in dataloaders:
        raise RuntimeError('Unknown dataset name: %s' % dataset_name)

    train_image_files, test_image_files, boxes = dataloaders[dataset_name](dataset_path, **kwargs)
    return dml_datasets.get_iterators(train_image_files, test_image_files, boxes, **kwargs)


def get_npairs_iterators(dataset_name, dataset_path, **kwargs):
    if dataset_name not in dataloaders:
        raise RuntimeError('Unknown dataset name: %s' % dataset_name)

    train_image_files, test_image_files, boxes = dataloaders[dataset_name](dataset_path, **kwargs)
    return dml_datasets.get_npairs_iterators(train_image_files, test_image_files, boxes, **kwargs)


def get_prototype_iterators(dataset_name, dataset_path, **kwargs):
    if dataset_name not in dataloaders:
        raise RuntimeError('Unknown dataset name: %s' % dataset_name)

    train_image_files, test_image_files, boxes = dataloaders[dataset_name](dataset_path, **kwargs)
    return dml_datasets.get_prototype_iterators(train_image_files, test_image_files, boxes, **kwargs)
