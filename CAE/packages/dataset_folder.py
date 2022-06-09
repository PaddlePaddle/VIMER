import random
import paddle
from paddle.vision import datasets
from paddle.vision.datasets.folder import make_dataset, default_loader


IMG_EXTENSIONS = ('.JPEG', '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


class DatasetFolder(datasets.DatasetFolder):
        
    def __init__(self,
                 root,
                 loader=None,
                 extensions=None,
                 transform=None,
                 is_valid_file=None):
        self.root = root
        self.transform = transform
        if extensions is None:
            extensions = IMG_EXTENSIONS
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions,
                               is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError(
                "Found 0 directories in subfolders of: " + self.root + "\n"
                "Supported extensions are: " + ",".join(extensions)))

        self.loader = default_loader if loader is None else loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.dtype = paddle.get_default_dtype()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        self.extensions = IMG_EXTENSIONS
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
                                                                                                                                                                                                                                                                                                        
        return sample, target
