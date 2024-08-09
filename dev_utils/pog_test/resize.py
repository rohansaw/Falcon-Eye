"modified from source: https://github.com/cogsys-tuebingen/adaptive_resizer/blob/main/dataset.py"
import os
import torch
import numpy as np

from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import pandas


class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, img_dir, transform=None, delete_exclusives=None, opt=None):

        self.root_dir = root_dir
        self.img_dir = img_dir
        self.transform = transform
        self.opt = opt

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', ann_file))
        self.image_ids = self.coco.getImgIds()

        self.metamax_train = -1e6
        self.metamin_train = 1e6

        # if self.opt.adaptive and (not self.opt.dataset):
        self.metadata = self.read_meta_csv(os.path.join(self.root_dir, 'annotations', 'meta' + '.csv'))

        for k in self.coco.anns:
            if 'iscrowd' not in self.coco.anns[k].keys():
                self.coco.anns[k]['iscrowd'] = 0

        self.load_classes()

    def give_id_for_filename(self, filename: str) -> list:
        idxs = []
        for img in self.coco.dataset['images']:
            # if img['id'] >= 5270:
            #     print('Jetzt simmer gleich da; im Moment {}.'.format(str(img['id'])))
            if os.path.splitext(filename)[0] == os.path.splitext(img['file_name'])[0]:
                idxs.append(img['id'])

        return idxs

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}

        meta = torch.Tensor([self.load_meta(idx)]).float()

        sample['img'] = [sample['img'], meta]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.img_dir, image_info['file_name'])
        print(path)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def read_meta_csv(self, meta_annotation_file):
        if os.path.isfile(meta_annotation_file):
            read_dict = pandas.read_csv(meta_annotation_file).to_dict()
        else:
            return None
        return_dict = {}
        maximum = -1e6
        minimum = 1e6
        for number in read_dict['image_name']:
            try:
                height = float(read_dict['altitude_normalized'][number])
                return_dict[read_dict['image_name'][number]] = height
                maximum = np.max([maximum, height])
                minimum = np.min([minimum, height])
            except:
                height = float(read_dict['altitude(feet)'][number])
                return_dict[read_dict['image_name'][number]] = height
                maximum = np.max([maximum, height])
                minimum = np.min([minimum, height])
        self.metamax_train = maximum
        self.metamin_train = minimum
        return return_dict

    def load_meta(self, image_index):
        # normalizer = lambda x: (x - self.metamin_train) / (self.metamax_train - self.metamin_train)
        normalizer = lambda x: x
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        try:
            return normalizer(self.metadata[os.path.splitext(image_info['file_name'])[0]])
        except KeyError:
            return normalizer(self.metadata[image_info['file_name']])

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def num_classes(self):
        # return 80
        return len(self.coco_labels)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    actual_imgs, metadata = [], []
    for img in imgs:
        actual_imgs.append(img[0])
        metadata.append(img[1])

    actual_imgs = torch.from_numpy(np.stack(actual_imgs, axis=0))
    metadata = torch.from_numpy(np.stack(metadata, axis=0))

    # make metadata a tensor of valid dimensions. len(imgs[0][1]) here is an attempt at the number of meta-parameters
    if type(imgs[0][1]) == list:
        metadata = metadata.view(-1, len(imgs[0][1]))
    else:
        metadata = metadata.view(-1, 1)

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    actual_imgs = actual_imgs.permute(0, 3, 1, 2)

    return {'img': [actual_imgs, metadata], 'annot': annot_padded, 'scale': scales}


class AdaptiveResizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, use_adaptive, opt):
        self.use_adaptive = use_adaptive
        self.opt = opt
        super(AdaptiveResizer, self)

    @staticmethod
    def _calc_size(meta, opt):
        # standards are opt.resize_mode = linear and opt.resize_parameters = None
        # linear makes sense geometrically. opt.resize_parameters is subject to optimization
        # depending on the data set and restricted by the available GPU memory.
        mode, parameters = opt.resize_mode, opt.resize_parameters
        freely = [2176]
        if (mode == 'free') and (parameters is not None):
            freely = parameters

        mode_to_sizes = {'linear': [512 + 128 * m for m in range(7)],
                         'interpolation': [384, 768, 768, 1536, 1536, 1536],
                         'linear_huge': [768 + 128 * m for m in range(7)],
                         'interpolation_huge': [384, 896, 1152, 1792, 1792, 2176],
                         'free': freely,
                         'interpolation_fine': [200, 248, 320, 560, 880, 984, 1296, 1792, 1856, 1856, 2000, 2160, 2160,
                                                2160, 2160, 2160, 2160, 2160],
                         'interpolation_non_monoton': [200, 248, 320, 560, 880, 984, 1296, 1792, 1128, 1856, 2000, 2160,
                                                       1488, 1528, 2176, 2176, 2176, 2160]}
        sizes = mode_to_sizes[mode]

        # TODO: make easier:
        decision_grid = np.linspace(0.0, 1.0, len(sizes) + 1)
        return_index = len(sizes) - 1
        for k in range(1, len(decision_grid) - 1):
            if decision_grid[k] > meta:
                return_index = k - 1
                break
        
        return sizes[return_index]

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image, metadata = image[0], image[1]

        if self.use_adaptive:
            common_size = AdaptiveResizer._calc_size(metadata, self.opt)
        else:
            # ist diese inplace-Operation problematisch?
            sample['img'] = [torch.from_numpy(image), metadata]
            sample['annot'] = torch.from_numpy(annots)
            sample['scale'] = sample['scale']

            return sample

        height, width, _ = image.shape

        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        # this assumes that the convolutional net needs the input tensor's dimensions to be a multiple of 128.
        new_image = np.zeros((int((common_size + 128) / 128) * 128, int((common_size + 128) / 128) * 128, 3))
        new_image[0:resized_height, 0:resized_width] = image

        if annots is not None:
            annots[:, :4] *= scale

        sample['img'] = [torch.from_numpy(new_image), metadata]
        sample['annot'] = torch.from_numpy(annots)
        sample['scale'] = scale * sample['scale']

        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, common_size):  # Konstruktor ist von mir!
        self.common_size = common_size

    def __call__(self, sample, common_size=1280):
        common_size = self.common_size
        image, annots = sample['img'], sample['annot']
        image, metadata = image[0], image[1]

        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        # ist diese inplace-Operation problematisch?
        sample['img'] = [new_image, metadata]
        sample['annot'] = annots
        sample['scale'] = scale

        return sample

        # return {'img': [new_image, metadata], 'annot': annots, 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image, metadata = image[0], image[1]

            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': [image, metadata], 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.43717304, 0.44310874, 0.33362516]]])
        self.std = np.array([[[0.23407398, 0.21981522, 0.2018422]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image, metadata = image[0], image[1]

        sample['img'] = [((image.astype(np.float32) - self.mean) / self.std), metadata]
        sample['annot'] = annots
        return sample