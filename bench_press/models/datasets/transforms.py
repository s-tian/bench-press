
class ImageTransform:

    def __init__(self, per_image_transform):
        self.transform = per_image_transform

    def __call__(self, sample):
        images = sample['images']
        transformed_images = [self.transform(image) for image in images]
        sample['images'] = transformed_images
        return sample


