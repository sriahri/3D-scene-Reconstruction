import torch
import os
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


def create_image_transforms(mode='train', augment_params=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                            do_augmentation=True, size=(256, 512)):
    if mode == 'train':
        transform = transforms.Compose([
            ResizeImage(train=True, size=size),
            RandomFlip(do_augmentation),
            ToTensor(train=True),
            AugmentImagePair(augment_params, do_augmentation)
        ])
    elif mode == 'test':
        transform = transforms.Compose([
            ResizeImage(train=False, size=size),
            ToTensor(train=False),
            DoTest(),
        ])
    return transform

class ResizeImage(object):
    def __init__(self, train=True, size=(256, 512)):
        self.is_train = train
        self.resize_transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.is_train:
            left_img, right_img = sample['left_image'], sample['right_image']
            left_img_resized = self.resize_transform(left_img)
            right_img_resized = self.resize_transform(right_img)
            sample = {'left_image': left_img_resized, 'right_image': right_img_resized}
        else:
            left_img = sample
            left_img_resized = self.resize_transform(left_img)
            sample = left_img_resized
        return sample

class DoTest(object):
    def __call__(self, sample):
        return torch.stack((sample, torch.flip(sample, [2])))

class ToTensor(object):
    def __init__(self, train):
        self.is_train = train
        self.tensor_transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.is_train:
            left_img, right_img = sample['left_image'], sample['right_image']
            left_tensor = self.tensor_transform(left_img)
            right_tensor = self.tensor_transform(right_img)
            sample = {'left_image': left_tensor, 'right_image': right_tensor}
        else:
            left_img = sample
            sample = self.tensor_transform(left_img)
        return sample

class RandomFlip(object):
    def __init__(self, do_augmentation):
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_img, right_img = sample['left_image'], sample['right_image']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                flipped_left = self.flip_transform(right_img)
                flipped_right = self.flip_transform(left_img)
                sample = {'left_image': flipped_left, 'right_image': flipped_right}
        else:
            sample = {'left_image': left_img, 'right_image': right_img}
        return sample

class AugmentImagePair(object):
    def __init__(self, augment_params, do_augmentation):
        self.do_augmentation = do_augmentation
        self.gamma_low, self.gamma_high, self.brightness_low, self.brightness_high, self.color_low, self.color_high = augment_params

    def __call__(self, sample):
        left_img, right_img = sample['left_image'], sample['right_image']
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > 0.5:
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_img_aug = left_img ** random_gamma
                right_img_aug = right_img ** random_gamma
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                left_img_aug *= random_brightness
                right_img_aug *= random_brightness
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    left_img_aug[i, :, :] *= random_colors[i]
                    right_img_aug[i, :, :] *= random_colors[i]
                left_img_aug = torch.clamp(left_img_aug, 0, 1)
                right_img_aug = torch.clamp(right_img_aug, 0, 1)
                sample = {'left_image': left_img_aug, 'right_image': right_img_aug}
        else:
            sample = {'left_image': left_img, 'right_image': right_img}
        return sample

class DepthDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, 'left/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname in os.listdir(left_dir)])
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'right/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname in os.listdir(right_dir)])
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_img = Image.open(self.right_paths[idx])
            sample = {'left_image': left_img, 'right_image': right_img}
            if self.transform:
                sample = self.transform(sample)
        else:
            if self.transform:
                left_img = self.transform(left_img)
            sample = left_img
        return sample


def prepare_dataloader(data_directory, mode, augment_params, do_augmentation, batch_size, size, num_workers):
    data_dirs = os.listdir(data_directory)
    transform = create_image_transforms(
        mode=mode,
        augment_params=augment_params,
        do_augmentation=do_augmentation,
        size=size)
    datasets = [DepthDataset(os.path.join(data_directory, data_dir), mode, transform=transform)
                for data_dir in data_dirs]
    dataset = ConcatDataset(datasets)
    n_img = len(dataset)
    print('Use a dataset with', n_img, 'images')
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=(mode == 'train'), num_workers=num_workers,
                        pin_memory=True)
    return n_img, loader

class DepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(DepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1]
        return gx

    def gradient_y(self, img):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :]
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()
        x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)
        x_shifts = disp[:, 0, :, :]
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y
        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]
        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]
        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]
        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]
        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(self.n)]

    def forward(self, input, target):
        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]
        left_est = [self.generate_image_left(right_pyramid[i], disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i], disp_right_est[i]) for i in range(self.n)]
        right_left_disp = [self.generate_image_left(disp_right_est[i], disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i], disp_right_est[i]) for i in range(self.n)]
        disp_left_smoothness = self.disp_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est, right_pyramid)
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i])) for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i] - right_pyramid[i])) for i in range(self.n)]
        ssim_left = [torch.mean(self.SSIM(left_est[i], left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i], right_pyramid[i])) for i in range(self.n)]
        image_loss_left = [self.SSIM_w * ssim_left[i] + (1 - self.SSIM_w) * l1_left[i] for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i] + (1 - self.SSIM_w) * l1_right[i] for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i] - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i] - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)
        disp_left_loss = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)
        loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss
        return loss