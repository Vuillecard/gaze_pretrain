import hydra
import numpy as np
import torch
import torchvision.transforms.v2 as tf
from omegaconf import DictConfig
from torchvision.ops import box_convert
from torchvision.transforms import Compose


class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
        """
        augmentations = []
        if not transforms_cfg.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for augmentation_name in transforms_cfg.get("order"):
            augmentation = hydra.utils.instantiate(transforms_cfg.get(augmentation_name))
            if augmentation is not None:
                augmentations.append(augmentation)
        self.augmentations = Compose(augmentations)

    def get_transforms(self):
        """Get TransformsWrapper module.
        Returns:
            Any: Transformation results.
        """
        return self.augmentations


class BboxReshape(object):
    def __init__(self, square: bool = True, ratio: float = 0.0):
        self.square = square
        self.ratio = ratio
        print("BboxReshape is square ", square, ratio)
    def __call__(self, sample):
        bbox = sample["head_bbox"]
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.tensor(np.array(bbox))

        bbox_cxcywh = box_convert(bbox, in_fmt="xyxy", out_fmt="cxcywh")

        if self.square:
            sizes = torch.max(bbox_cxcywh[:, 2:], 1)[0]
            sizes = torch.stack([sizes, sizes], 1)
        else:
            sizes = bbox_cxcywh[:, 2:]

        bbox_cxcywh[:, 2:] = sizes + (sizes * self.ratio)
        sample["head_bbox"] = box_convert(bbox_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")

        return sample


class BboxRandomJitter(object):  # TODO
    """Jitter the bounding box of the face in a sample.
    If jitter_size is 1 then the bbox can have no overlap with the original bbox
    The same jittering is applied to all the bbox of the sequence

    Args:
        jitter_size (int): Desired jitter size.
    """

    def __init__(
        self, 
        jitter_ratio: float = 0.1, 
        p :float = 0.5):
        self.jitter_ratio = jitter_ratio
        assert 0 <= jitter_ratio <= 1
        self.p = p

    def __call__(self, sample):

        if torch.rand(1) < self.p:
            bbox = sample["head_bbox"]
            if not isinstance(bbox, torch.Tensor):
                bbox = torch.tensor(np.array(bbox)).type(torch.int16)
            # assert bbox.dim() == 2
            # assert bbox.size(1) == 4
            nb_box = bbox.size(0)

            bbox_cxcywh = box_convert(bbox, in_fmt="xyxy", out_fmt="cxcywh")
            sizes = torch.max(bbox_cxcywh[:, 2:], 1)[0]

            if sample["bbox_strategy"] == "fixed_center":
                jitter = sizes[nb_box // 2] * self.jitter_ratio
                jitter = jitter * (torch.rand(2) * 2 - 1)

            elif sample["bbox_strategy"] == "followed":
                jitter = sizes * self.jitter_ratio
                jitter = torch.stack([jitter, jitter], 1)
                jitter = jitter * (torch.rand((2)) * 2 - 1)

            else:
                raise ValueError(f"Unkown bbox strategy {sample['bbox_strategy']}")

            bbox_cxcywh[:, :2] = bbox_cxcywh[:, :2] + jitter
            sample["head_bbox"] = box_convert(bbox_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")

        return sample


class Crop(object):
    """Crop according to the bbox in the sample

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        head_bbox = sample["head_bbox"]
        head_bbox = box_convert(head_bbox, in_fmt="xyxy", out_fmt="xywh")
        head_bbox = head_bbox.type(torch.int16)

        # crop and resize functions
        sample["images"] = [
            tf.functional.resized_crop(
                img,
                head_bbox[i, 1],
                head_bbox[i, 0],
                head_bbox[i, 3],
                head_bbox[i, 2],
                self.output_size,
                antialias=True,
            )
            for i, img in enumerate(sample["images"])
        ]
        # sample["images"] = [
        #     tf.functional.resize(
        #         tf.functional.crop(
        #             img,
        #             head_bbox[i, 1],
        #             head_bbox[i, 0],
        #             head_bbox[i, 3],
        #             head_bbox[i, 2],
        #         ),
        #         self.output_size,
        #         antialias=True,
        #     )
        #     for i, img in enumerate(sample["images"])
        # ]

        return sample


class CropRandomResize(object):
    """Crop according to the bbox in the sample

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=224, scale=(0.8, 1.0)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.tf_random_resize = tf.RandomResizedCrop(output_size, scale=scale, antialias=True)

    def __call__(self, sample):
        head_bbox = sample["head_bbox"]
        head_bbox = box_convert(head_bbox, in_fmt="xyxy", out_fmt="xywh")
        head_bbox = head_bbox.type(torch.int16)
        i, j, h, w = self.tf_random_resize.get_params(
            torch.zeros(3, self.output_size[0], self.output_size[1]),
            scale=self.tf_random_resize.scale,
            ratio=self.tf_random_resize.ratio,
        )
        # crop and random resize functions
        sample["images"] = [
            tf.functional.resized_crop(
                tf.functional.resized_crop(
                    img,
                    head_bbox[p, 1],
                    head_bbox[p, 0],
                    head_bbox[p, 3],
                    head_bbox[p, 2],
                    self.output_size,
                    antialias=True,
                ),
                i,
                j,
                h,
                w,
                self.output_size,
                antialias=True,
            )
            for p, img in enumerate(sample["images"])
        ]

        return sample


class HorizontalFlip(object):
    """Flip the images in the sample horizontally"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            sample["images"] = [tf.functional.hflip(img) for img in sample["images"]]
            sample["task_gaze_yawpitch"] = sample["task_gaze_yawpitch"] * torch.tensor([-1, 1]) # yaw is flipped
            sample["task_gaze_vector"] = sample["task_gaze_vector"] * torch.tensor([-1, 1, 1]) # x is flipped
        return sample


class ColorJitter(object):
    """
    Applies random colors transformations to the input (ie. brightness,
    contrast, saturation and hue).
    """

    def __init__(self, brightness, contrast, saturation, hue, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, sample):

        if torch.rand(1) <= self.p:

            # Sample color transformation factors and order
            brightness_factor = None if self.brightness is None else torch.rand(1).uniform_(*self.brightness)
            contrast_factor = None if self.contrast is None else torch.rand(1).uniform_(*self.contrast)
            saturation_factor = None if self.saturation is None else torch.rand(1).uniform_(*self.saturation)
            hue_factor = None if self.hue is None else torch.rand(1).uniform_(*self.hue)
            
            if np.array(sample['image']).sum()>0:
                fn_indices = torch.randperm(4)
                for fn_id in fn_indices:

                    if fn_id == 0 and brightness_factor is not None:
                        sample["image"] = [ tf.functional.adjust_brightness(img, brightness_factor) for img in sample["image"] ]            
                        
                    elif fn_id == 1 and contrast_factor is not None:
                        sample["image"] = [ tf.functional.adjust_contrast(img, contrast_factor) for img in sample["image"] ]            
                        
                    elif fn_id == 2 and saturation_factor is not None:
                        sample["image"] = [ tf.functional.adjust_saturation(img, saturation_factor) for img in sample["image"] ]
                        
                    elif fn_id == 3 and hue_factor is not None:
                        sample["image"] = [ tf.functional.adjust_hue(img, hue_factor) for img in sample["image"] ]
                      
        return sample

class RandomGaussianBlur(object):
    """Apply random gaussian blur to the input"""

    def __init__(
            self, 
            radius=4,
            sigma = (0.2,3.0), 
            p=0.5):
        self.p = p
        self.radius = radius
        self.sigma = sigma

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            kernel_size = torch.randint(1, self.radius, (1,)).item() * 2 + 1
            sample["images"] = [tf.functional.gaussian_blur(img, kernel_size, self.sigma) for img in sample["images"]]
        
        return sample

class Random(object):
    """Crop according to the bbox in the sample

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=224, scale=(0.8, 1.0)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.tf_random_resize = tf.RandomResizedCrop(output_size, scale=scale, antialias=True)

    def __call__(self, sample):
        head_bbox = sample["head_bbox"]
        head_bbox = box_convert(head_bbox, in_fmt="xyxy", out_fmt="xywh")
        head_bbox = head_bbox.type(torch.int16)

        # crop and random resize functions
        sample["images"] = [
            self.tf_random_resize(
                tf.functional.crop(
                    img,
                    head_bbox[i, 1],
                    head_bbox[i, 0],
                    head_bbox[i, 3],
                    head_bbox[i, 2],
                )
            )
            for i, img in enumerate(sample["images"])
        ]

        return sample


class ToImage(object):
    """Convert PIL image to Tensor.
    make sure all the dimensions are correct
    ref: https://pytorch.org/vision/main/transforms.html#range-and-dtype"""

    def __call__(self, sample):
        sample["images"] = [
            tf.functional.to_dtype(tf.functional.to_image(img), dtype=torch.uint8, scale=True)
            for img in sample["images"]
        ]

        return sample


class ToTensor(object):
    """Convert tensor image to float"""

    def __call__(self, sample):
        sample["images"] = torch.stack(
            [
                tf.functional.to_dtype(img, dtype=torch.float32, scale=True)
                for img in sample["images"]
            ],
            0,
        )
        return sample


class Normalize(object):
    """Normalize the images in the sample"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # B, C, H, W
        # assert images.dim() == 4
        # assert images.size(1) == 3
        sample["images"] = tf.functional.normalize(sample["images"], self.mean, self.std)

        return sample
