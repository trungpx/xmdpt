import json
import cv2
import numpy as np

from torch.utils.data import DataLoader, Dataset
import os
import math
import lmdb
from io import BytesIO
from PIL import Image

import torch
import torchvision.transforms.functional as F

import random

import torchvision.transforms as transforms

class FashionDataset(Dataset):
    def __init__(self, data_dir=None, is_inference=False, labels_required = False):
        self.root = data_dir # './datasets/deepfashion/'
        self.semantic_path = self.root
        self.sub_path = '256-256'
        self.scale_param = 0.05
        path = os.path.join(self.root, str(self.sub_path))
        self.path = path
        self.labels_required = labels_required

        self.file_path = 'train_pairs.txt' if not is_inference else 'test_pairs.txt'
        self.data = self.get_paths(self.root, self.file_path)
        self.is_inference = is_inference
        self.scale_param = self.scale_param if not is_inference else 0
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def __len__(self):
        return len(self.data)

    def get_paths(self, root, path):
        fd = open(os.path.join(root, path))
        lines = fd.readlines()
        fd.close()

        image_paths = []
        for item in lines:
            dict_item={}
            item = item.strip().split(',')
            dict_item['source_image'] = [path.replace('.jpg', '.png') for path in item[1:]]
            dict_item['source_label'] = [os.path.join(self.semantic_path, self.img_to_label(path)) for path in dict_item['source_image']]
            dict_item['target_image'] = item[0].replace('.jpg', '.png')
            dict_item['target_label'] = os.path.join(self.semantic_path, self.img_to_label(dict_item['target_image']))
            image_paths.append(dict_item)
        return image_paths

    def open_lmdb(self):
        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(buffers=True)

    def __getitem__(self, index):

        if not hasattr(self, 'txn'):
            self.open_lmdb()

        path_item = self.data[index]
        
        i = np.random.choice(list(range(0, len(path_item['source_image']))))
        source_image_path = path_item['source_image'][i]
        source_label_path = path_item['source_label'][i]

        target_image_tensor, param = self.get_image_tensor(path_item['target_image'])
        
        if self.labels_required:
            target_label_tensor, target_face_center = self.get_label_tensor(path_item['target_label'], target_image_tensor, param)
        ref_tensor, param = self.get_image_tensor(source_image_path)
        if self.labels_required:
            label_ref_tensor, ref_face_center = self.get_label_tensor(source_label_path, ref_tensor, param)
        
        image_path = self.get_image_path(source_image_path, path_item['target_image'])
        if not self.is_inference:
            if torch.rand(1) < 0.5:
                target_image_tensor = F.hflip(target_image_tensor)
                ref_tensor = F.hflip(ref_tensor)

                if self.labels_required:
                    target_label_tensor = F.hflip(target_label_tensor)
                    label_ref_tensor = F.hflip(label_ref_tensor)
        #### DINO input 224 // 14 for ViT
        transform = transforms.Compose([transforms.Resize(224, interpolation=Image.BICUBIC)])
        dino_src = transform(ref_tensor)
        dino_tgt = transform(target_image_tensor)
        dino_pose_src = transform(label_ref_tensor)
        dino_pose_tgt = transform(target_label_tensor)
        ####
        if self.labels_required:
            input_dict = {'target_skeleton': target_label_tensor,
                        'target_image': target_image_tensor, # NEW .permute(2,0,1)
                        'target_face_center': target_face_center,

                        'source_image': ref_tensor, # NEW
                        'source_skeleton': label_ref_tensor,
                        'source_face_center': ref_face_center,

                        'path': image_path,
                        
                        'dino_src': dino_src,
                        'dino_tgt': dino_tgt,
                        
                        'dino_pose_src': dino_pose_src,
                        'dino_pose_tgt': dino_pose_tgt,
                        
                        }
        else:
            input_dict = {'target_image': target_image_tensor,
                          'source_image': ref_tensor,
                         }
        classes = {}
        classes["y"] = np.array(0, dtype=np.int64)
        return input_dict, classes


    def get_image_path(self, source_name, target_name):
        source_name = self.path_to_fashion_name(source_name)
        target_name = self.path_to_fashion_name(target_name)
        image_path = os.path.splitext(source_name)[0] + '_2_' + os.path.splitext(target_name)[0]+'_vis.png'
        return image_path

    def path_to_fashion_name(self, path_in):
        path_in = path_in.split('img/')[-1]
        path_in = os.path.join('fashion', path_in)
        path_names = path_in.split('/') 
        path_names[3] = path_names[3].replace('_', '')
        path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
        path_names = "".join(path_names)
        return path_names

    def img_to_label(self, path):
        return path.replace('img/', 'pose/').replace('.png', '.txt')

    def get_image_tensor(self, path):
        with self.env.begin(write=False) as txn:
            key = f'{path}'.encode('utf-8')
            img_bytes = txn.get(key) 
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        param = get_random_params(img.size, self.scale_param)
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param

    def get_label_tensor(self, path, img, param):
        canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)
        keypoint = np.loadtxt(path)
        keypoint = self.trans_keypoins(keypoint, param, img.shape[1:])
        stickwidth = 4
        for i in range(18):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
        joints = []
        for i in range(17):
            Y = keypoint[np.array(self.limbSeq[i])-1, 0]
            X = keypoint[np.array(self.limbSeq[i])-1, 1]            
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)
        pose = F.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        pose /= pose.max()

        if int(keypoint[14, 0]) != -1 and int(keypoint[15, 0]) != -1:
            y0, x0 = keypoint[14, 0:2]
            y1, x1 = keypoint[15, 0:2]
            face_center = torch.tensor([y0, x0, y1, x1]).float()
        else:
            face_center = torch.tensor([-1, -1, -1, -1]).float()               
        return pose, face_center
        # return label_tensor, face_center

    def trans_keypoins(self, keypoints, param, img_size):
        missing_keypoint_index = keypoints == -1
        
        # crop the white line in the original dataset
        keypoints[:,0] = (keypoints[:,0]-40)

        # resize the dataset
        img_h, img_w = img_size
        scale_w = 1.0/176.0 * img_w
        scale_h = 1.0/256.0 * img_h

        if 'scale_size' in param and param['scale_size'] is not None:
            new_h, new_w = param['scale_size']
            scale_w = scale_w / img_w * new_w
            scale_h = scale_h / img_h * new_h

        if 'crop_param' in param and param['crop_param'] is not None:
            w, h, _, _ = param['crop_param']
        else:
            w, h = 0, 0

        keypoints[:,0] = keypoints[:,0]*scale_w - w
        keypoints[:,1] = keypoints[:,1]*scale_h - h
        keypoints[missing_keypoint_index] = -1
        
        return keypoints


def get_random_params(size, scale_param):
    w, h = size
    scale = random.random() * scale_param

    new_w = int( w * (1.0+scale) )
    new_h = int( h * (1.0+scale) )
    x = random.randint(0, np.maximum(0, new_w - w))
    y = random.randint(0, np.maximum(0, new_h - h))
    return {'crop_param': (x, y, w, h), 'scale_size':(new_h, new_w)}        


def get_transform(param, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if 'scale_size' in param and param['scale_size'] is not None:
        osize = param['scale_size']
        transform_list.append(transforms.Resize(osize, interpolation=method))

    if 'crop_param' in param and param['crop_param'] is not None:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, param['crop_param'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __crop(img, pos):
    x1, y1, tw, th = pos
    return img.crop((x1, y1, x1 + tw, y1 + th))


### load data for MDT
def load_my_data(
    *,
    data_dir,
    batch_size,
    is_inference=False,
    labels_required=True,
    number_of_workers=8,
    # image_size,
    # class_cond=False,
    # deterministic=False,
    # random_crop=False,
    # random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    # classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    # dataset = FashionDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,
    # )
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    # while True:
    #     yield from loader

    dataset = FashionDataset(data_dir=data_dir, 
                             is_inference=is_inference, 
                             labels_required=labels_required
                             )
    loader = DataLoader(dataset, num_workers=number_of_workers, batch_size=batch_size, shuffle=not is_inference, drop_last=not is_inference)
    while True:
        yield from loader