import cv2
import numpy as np
import torch


def _is_numpy_image(image):
    '''
    Description: Return whether image is np.ndarray and the number of dimensions of image
    Return: True or False.
    '''
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})


class RandomHorizontalFlip:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, mask=None):
        if not _is_numpy_image(image):
            raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

        if np.random.random() < self.prob:
            if image.shape[2] == 1:
                image = cv2.flip(image, 1)[:, :, np.newaxis]  # keep image.shape = H x W x 1
            else:
                image = cv2.flip(image, 1)
            if mask is not None:
                if not (_is_numpy_image(mask)):
                    raise TypeError("sample['mask'] should be numpy.ndarray. Got {}".format(type(mask)))

                mask = cv2.flip(mask, 1)

        return image, mask


class RandomVerticalFlip:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, mask=None):
        if not _is_numpy_image(image):
            raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

        if np.random.random() < self.prob:
            if image.shape[2] == 1:
                image = cv2.flip(image, 0)[:, :, np.newaxis]  # keep image.shape = H x W x 1
            else:
                image = cv2.flip(image, 0)
            if mask is not None:
                if not (_is_numpy_image(mask)):
                    raise TypeError("sample['mask'] should be numpy.ndarray. Got {}".format(type(mask)))

                mask = cv2.flip(mask, 0)

        return image, mask


class RandomCrop:
    def __init__(self, output_size=0, prob=1.0):
        self.output_size = output_size
        self.prob = prob

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, image, mask=None):
        if not _is_numpy_image(image):
            raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

        crop_image, crop_mask = None, None
        if np.random.random() < self.prob:
            image_h, image_w = image.shape[0], image.shape[1]
            crop_w, crop_h = self.output_size
            assert (image_w >= crop_w) and (image_h >= crop_h)

            if image_w == crop_w:
                topleft_h = 0
            else:
                topleft_h = np.random.randint(low=0, high=image_h - crop_h)

            if image_h == crop_h:
                topleft_w = 0
            else:
                topleft_w = np.random.randint(low=0, high=image_w - crop_w)
            crop_image = image[topleft_h:topleft_h + crop_h, topleft_w:topleft_w + crop_w, :]

            if mask is not None:
                if not (_is_numpy_image(mask)):
                    raise TypeError("sample['mask'] should be numpy.ndarray. Got {}".format(type(mask)))

                crop_mask = mask[topleft_h:topleft_h + crop_h, topleft_w:topleft_w + crop_w, :]

        return crop_image, crop_mask


class ToTensor:
    def __call__(self, image, mask=None):
        if not _is_numpy_image(image):
            raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

        # handle numpy.array
        if image.ndim == 2:
            image = image[:, :, None]

        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if isinstance(image, torch.ByteTensor) or image.dtype == torch.uint8:
            image = image.float().div(255)

        if mask is not None:
            if not (_is_numpy_image(mask)):
                raise TypeError("sample['mask'] should be numpy.ndarray. Got {}".format(type(mask)))

            if mask.ndim == 2:
                mask = mask[:, :, None]
            mask = torch.from_numpy(mask.transpose((2, 0, 1)))
            mask = mask.float()

        return image, mask
