import random
import math

import cv2
import numpy as np
from PIL import Image

class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].
    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.
    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, im, label=None):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.
        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
        if isinstance(label, str):
            label = np.asarray(Image.open(label))
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        im = np.transpose(im, (2, 0, 1))
        return (im, label)

class RandomHorizontalFlip:
    """
    Flip an image horizontally with a certain probability.
    Args:
        prob (float, optional): A probability of horizontally flipping. Default: 0.5.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, label=None):
        if random.random() < self.prob:
            im = functional.horizontal_flip(im)
            if label is not None:
                label = functional.horizontal_flip(label)
        if label is None:
            return (im, )
        else:
            return (im, label)

class RandomDistort:
    """
    Distort an image with random configurations.
    Args:
        brightness_range (float, optional): A range of brightness. Default: 0.5.
        brightness_prob (float, optional): A probability of adjusting brightness. Default: 0.5.
        contrast_range (float, optional): A range of contrast. Default: 0.5.
        contrast_prob (float, optional): A probability of adjusting contrast. Default: 0.5.
        saturation_range (float, optional): A range of saturation. Default: 0.5.
        saturation_prob (float, optional): A probability of adjusting saturation. Default: 0.5.
        hue_range (int, optional): A range of hue. Default: 18.
        hue_prob (float, optional): A probability of adjusting hue. Default: 0.5.
        sharpness_range (float, optional): A range of sharpness. Default: 0.5.
        sharpness_prob (float, optional): A probability of adjusting saturation. Default: 0.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.
        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        sharpness_lower = 1 - self.sharpness_range
        sharpness_upper = 1 + self.sharpness_range
        ops = [
            functional.brightness, functional.contrast, functional.saturation,
            functional.hue, functional.sharpness
        ]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            },
            'sharpness': {
                'sharpness_lower': sharpness_lower,
                'sharpness_upper': sharpness_upper,
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        im = im.astype('uint8')
        im = Image.fromarray(im)
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = im
            if np.random.uniform(0, 1) < prob:
                im = ops[id](**params)
        im = np.asarray(im).astype('float32')
        if label is None:
            return (im, )
        else:
            return (im, label)

class RandomRotation:
    """
    Rotate an image randomly with padding.
    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    """

    def __init__(self,
                 max_rotation=15,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.
        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.max_rotation > 0:
            (h, w) = im.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation,
                                            self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            cos = np.abs(r[0, 0])
            sin = np.abs(r[0, 1])

            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))

            (cx, cy) = pc
            r[0, 2] += (nw / 2) - cx
            r[1, 2] += (nh / 2) - cy
            dsize = (nw, nh)
            im = cv2.warpAffine(
                im,
                r,
                dsize=dsize,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
            if label is not None:
                label = cv2.warpAffine(
                    label,
                    r,
                    dsize=dsize,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.label_padding_value)

        if label is None:
            return (im, )
        else:
            return (im, label)


class RandomBlur:
    """
    Blurring an image by a Gaussian function with a certain probability.
    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.1.
        blur_type(str, optional): A type of blurring an image,
            gaussian stands for cv2.GaussianBlur,
            median stands for cv2.medianBlur,
            blur stands for cv2.blur,
            random represents randomly selected from above.
            Default: gaussian.
    """

    def __init__(self, prob=0.1, blur_type="gaussian"):
        self.prob = prob
        self.blur_type = blur_type

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.
        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im = np.array(im, dtype='uint8')
                if self.blur_type == "gaussian":
                    im = cv2.GaussianBlur(im, (radius, radius), 0, 0)
                elif self.blur_type == "median":
                    im = cv2.medianBlur(im, radius)
                elif self.blur_type == "blur":
                    im = cv2.blur(im, (radius, radius))
                elif self.blur_type == "random":
                    select = random.random()
                    if select < 0.3:
                        im = cv2.GaussianBlur(im, (radius, radius), 0)
                    elif select < 0.6:
                        im = cv2.medianBlur(im, radius)
                    else:
                        im = cv2.blur(im, (radius, radius))
                else:
                    im = cv2.GaussianBlur(im, (radius, radius), 0, 0)
        im = np.array(im, dtype='float32')
        if label is None:
            return (im, )
        else:
            return (im, label)

class RandomScaleAspect:
    """
    Crop a sub-image from an original image with a range of area ratio and aspect and
    then scale the sub-image back to the size of the original image.
    Args:
        min_scale (float, optional): The minimum area ratio of cropped image to the original image. Default: 0.5.
        aspect_ratio (float, optional): The minimum aspect ratio. Default: 0.33.
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.
        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im.shape[0]
            img_width = im.shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    im = im[h1:(h1 + dh), w1:(w1 + dw), :]
                    im = cv2.resize(
                        im, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    if label is not None:
                        label = label[h1:(h1 + dh), w1:(w1 + dw)]
                        label = cv2.resize(
                            label, (img_width, img_height),
                            interpolation=cv2.INTER_NEAREST)
                    break
        if label is None:
            return (im, )
        else:
            return (im, label)