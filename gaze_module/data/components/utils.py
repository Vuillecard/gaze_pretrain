from typing import Iterator, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Sampler

from gaze_module.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


"""
This module contains utility functions and classes for the datamodules.
Base classes and functions for the datamodules.
* BaseDataset: Base class for the dataset
* BaseDataModule: Base class for the datamodule
* DataSampler: Select a sample strategy for the dataset
* BatchSamplerCombined: Class based on BatchSampler in pytorch
* pil_loader: Load image using PIL
* get_bbox_in_body: Get the bounding box in the body
* create_window: Create a window of frames around the current frame
* default_loader: Load image using PIL
"""


def default_loader(path):
    try:
        im = Image.open(path).convert("RGB")
        return im
    except OSError:
        raise OSError(f"Cannot load image {path}")
        print(path)
        return Image.new("RGB", (224, 224), "white")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_bbox_in_body(bbox, body_bbox):
    bbox_in_body = np.zeros_like(bbox)
    bbox_in_body[0] = bbox[0] - body_bbox[0]
    bbox_in_body[1] = bbox[1] - body_bbox[1]
    bbox_in_body[2] = bbox[2] - body_bbox[0]
    bbox_in_body[3] = bbox[3] - body_bbox[1]
    bbox_in_body = bbox_in_body.astype(int)
    return bbox_in_body


def create_window(frame: int, window_size: int, window_stride: int) -> List[int]:
    """Create a window of frames around the current frame

    Args:
        frame (int): The current frame
        window_size (int): The size of the window
        window_stride (int): The stride of the window

    Returns:
        List[int]: A list of frame indices
    """
    assert window_size % 2 == 1, "Window size must be odd"

    window_min = frame - (window_size // 2) * window_stride
    window_max = frame + ((window_size // 2) + 1) * window_stride
    return np.arange(window_min, window_max, window_stride)


# class based on BatchSampler in pytorch
class BatchSamplerCombined(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.
    each batch sample one batch from one dataset according to a sample probability.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        samplers: List[Sampler[int]],
        data_size: List[int],
        batch_size: int,
        generator=None,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )

        self.samplers = samplers
        self.data_size = data_size
        self.batch_size = batch_size
        self.drop_last = True
        self.num_sample_per_dataset = [
            len(sampler) // self.batch_size for sampler in self.samplers
        ]
        self.num_samples = sum([len(sampler) // self.batch_size for sampler in self.samplers])
        self.generator = generator

    def get_data_index(self, i):
        if i == 0:
            return 0
        return sum(self.data_size[:i])

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951

        # Randomly shuffle the batch selection
        data_selection = torch.concatenate(
            [
                torch.full((self.num_sample_per_dataset[i],), i, dtype=torch.int64)
                for i in range(len(self.samplers))
            ]
        )
        data_selection = data_selection[
            torch.randperm(len(data_selection), generator=self.generator)
        ]
        samplers_iter = [iter(sampler) for sampler in self.samplers]

        # Add the data index to account for the offset index after concat datasets
        for i in data_selection:
            batch = [
                next(samplers_iter[i]) + self.get_data_index(i) for _ in range(self.batch_size)
            ]
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        return self.num_samples


class BatchSamplerSequential(Sampler[List[int]]):
    r"""Wraps sequential samplers in batch for mulitple datasets.
    It is mainly used for validataion and test dataloader inorder to have sequential batch without drop last
    Warning: batch at the end of the dataset can be smaller than batch_size
    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(SequentialBatchSampler(SequentialSampler(range(10)), batch_size=3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """

    def __init__(self, samplers: List[Sampler[int]], batch_size) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.

        self.batch_sampler = [
            torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
            for sampler in samplers
        ]
        self.data_size = [len(sampler) for sampler in samplers]

    def get_data_index(self, i):
        if i == 0:
            return 0
        return sum(self.data_size[:i])

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951

        for idx_data, batch_sampler in enumerate(self.batch_sampler):
            for batch in batch_sampler:
                yield [batch[i] + self.get_data_index(idx_data) for i in range(len(batch))]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        return sum([len(batch_sampler) for batch_sampler in self.batch_sampler])


# _H5PY_FILES_CACHE = weakref.WeakValueDictionary()


# def save(file, array):
#     magic_string = b"\x93NUMPY\x01\x00v\x00"
#     header = bytes(
#         (
#             "{'descr': '"
#             + array.dtype.descr[0][1]
#             + "', 'fortran_order': False, 'shape': "
#             + str(array.shape)
#             + ", }"
#         ).ljust(127 - len(magic_string))
#         + "\n",
#         "utf-8",
#     )
#     if type(file) == str:
#         file = open(file, "wb")
#     file.write(magic_string)
#     file.write(header)
#     file.write(array.tobytes())


# def load(file):
#     if type(file) == str:
#         file = open(file, "rb")
#     header = file.read(128)
#     if not header:
#         return None
#     descr = str(header[19:25], "utf-8").replace("'", "").replace(" ", "")
#     shape = tuple(
#         int(num)
#         for num in str(header[60:120], "utf-8")
#         .replace(", }", "")
#         .replace("(", "")
#         .replace(")", "")
#         .split(",")
#     )
#     datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
#     for dimension in shape:
#         datasize *= dimension
#     return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))


# def generate_sample_clip(image_database_sample, window_size: int, window_stride: int):
#     video_path = image_database_sample["clip_path"]
#     frames = create_window(image_database_sample["frame"], window_size, window_stride)
#     frames = [f - 1 for f in frames]
#     video = VideoReader(video_path)
#     images = video.get_batch(frames).asnumpy()
#     # buffer = io.BytesIO()
#     # save(buffer, images)
#     # fastnumpyio_save_data = buffer.getvalue()
#     return images


# def generate_sample_image(
#     image_database_sample, dir_tmp="/idiap/temp/pvuillecard/datasets/CCDbHG/samples/tmp"
# ):
#     video_path = image_database_sample["clip_path"]
#     frame = image_database_sample["frame"]
#     video = VideoReader(video_path)
#     image = video[frame - 1].asnumpy()
#     image = Image.fromarray(image)
#     image.save(os.path.join(dir_tmp, f"{image_database_sample['clip_id']}_{frame:08d}.jpg"))
#     out = np.fromfile(
#         os.path.join(dir_tmp, f"{image_database_sample['clip_id']}_{frame:08d}.jpg"),
#         dtype=np.uint8,
#     )
#     os.remove(os.path.join(dir_tmp, f"{image_database_sample['clip_id']}_{frame:08d}.jpg"))
#     return out


# def parallel_generator(
#     func: Callable,
#     array: Iterable,
#     n_jobs: Optional[int] = None,
#     buffer: int = 1024,
# ) -> Any:
#     """Generator in parallel threads."""
#     array = iter(array)
#     thread_queue = queue.Queue(buffer)
#     n_jobs = os.cpu_count() if n_jobs is None else n_jobs
#     n_jobs = 1
#     with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
#         # Prefill thread queue with buffer elements
#         for item in itertools.islice(array, buffer):
#             thread_queue.put(executor.submit(func, item))
#         # Start giving out results, while refilling tasks
#         for item in array:
#             yield thread_queue.get().result()
#             thread_queue.put(executor.submit(func, item))
#         # Give out remaining results
#         while not thread_queue.empty():
#             yield thread_queue.get().result()


# class H5PyFile:
#     """This is a wrapper around a h5py file/dataset, which discards the open
#     dataset, when pickling and reopens it, when unpickling, instead of trying
#     to pickle the h5py.File object itself.

#     Please note, that this wrapper doesn't provide any extra safeguards
#     for parallel interactions with the dataset. Reading in parallel is safe,
#     writing in parallel may be not. Check h5py docs, when in doubt.

#     Thanks to Andrei Stoskii for this module which I slightly reworked.
#     """

#     def __init__(self, filename: Optional[str] = None, mode: str = "r", **kwargs: Any) -> None:
#         """H5PyFile module.

#         Args:
#             filename (:obj:`str`, optional): h5py filename. Default to None.
#             mode (str): h5py file operation mode (r, r+, w, w-, x, a). Default to 'r'.
#             **kwargs: Additional arguments for h5py.File class initialization.
#         """

#         self.filename = filename
#         self.mode = mode
#         self.dataset = None
#         self._kwargs = kwargs

#     def _lazy_load_(self) -> None:
#         if self.dataset is not None:
#             return

#         if not self.filename:
#             raise FileNotFoundError(f"File '{self.filename}' is not found!")

#         can_use_cache = True
#         if self.mode != "r":
#             # Non-read mode
#             can_use_cache = False

#         # Load dataset (from cache or from disk)
#         dataset = None
#         if can_use_cache:
#             dataset = _H5PY_FILES_CACHE.get(self.filename, None)
#         if dataset is None:
#             dataset = h5py.File(self.filename, swmr=True, **self._kwargs)

#         # Save dataset to cache and to self
#         if can_use_cache:
#             _H5PY_FILES_CACHE[self.filename] = dataset
#         self.dataset = dataset

#     def __getitem__(self, *args: Any, **kwargs: Any) -> Any:
#         self._lazy_load_()
#         return self.dataset.__getitem__(*args, **kwargs)[...]

#     def __setitem__(self, *args: Any, **kwargs: Any) -> Any:
#         self._lazy_load_()
#         return self.dataset.__setitem__(*args, **kwargs)

#     def __getstate__(self) -> Tuple[str, str, Dict[str, Any]]:
#         return self.filename, self.mode, self._kwargs

#     def __setstate__(self, state: Tuple[str, str, Dict[str, Any]]) -> Any:
#         return self.__init__(state[0], state[1], **state[2])

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}({self.filename}, {self.mode})"

#     @classmethod
#     def create(
#         cls,
#         filename: str,
#         content: List[str],
#         dirname: Optional[str] = None,
#         verbose: bool = True,
#     ) -> None:
#         """Create h5py file for dataset from scratch.

#         Args:
#             filename (str): h5py filename.
#             content (List[str]): Dataset content. Requires List[data filepath].
#             dirname (:obj:`str`, optional): Additional dirname for data filepaths.
#                 Default to None.
#             verbose (bool): Verbose option. If True, it would show tqdm progress bar.
#                 Default to True.
#         """
#         filename = Path(filename)
#         ext = filename.suffix
#         if ext != ".h5":
#             raise RuntimeError(f"Expected extension to be '.h5', instead got '{ext}'.")
#         dirname = Path("" if dirname is None else dirname)
#         progress_bar = tqdm if verbose else (lambda it, *_, **__: it)

#         # Check that all files exist
#         generator = parallel_generator(
#             lambda fp: (dirname / fp, (dirname / fp).is_file()),
#             content,
#             n_jobs=128,
#         )
#         for filepath, found in progress_bar(
#             generator, desc="Indexing content", total=len(content)
#         ):
#             if not found:
#                 raise FileNotFoundError(filepath)

#         # Read files from disk and save them to the dataset
#         generator = parallel_generator(
#             lambda fp: (fp, np.fromfile(dirname / fp, dtype=np.uint8)),
#             content,
#             n_jobs=128,
#         )
#         with h5py.File(filename, mode="x") as dataset:
#             for filepath, data in progress_bar(
#                 generator, desc="Creating dataset", total=len(content)
#             ):
#                 dataset[str(filepath)] = data

#     @classmethod
#     def create_video(
#         cls,
#         filename: str,
#         image_database_path: str,
#         dirname: Optional[str] = None,
#         verbose: bool = True,
#     ) -> None:
#         """Create h5py file for dataset from scratch.

#         Args:
#             filename (str): h5py filename.
#             content (List[str]): Dataset content. Requires List[data filepath].
#             dirname (:obj:`str`, optional): Additional dirname for data filepaths.
#                 Default to None.
#             verbose (bool): Verbose option. If True, it would show tqdm progress bar.
#                 Default to True.
#         """
#         images_dir = "/idiap/project/epartners4all/data/CCDb_images"
#         filename = Path(filename)
#         ext = filename.suffix
#         if ext != ".h5":
#             raise RuntimeError(f"Expected extension to be '.h5', instead got '{ext}'.")
#         dirname = Path("" if dirname is None else dirname)
#         progress_bar = tqdm if verbose else (lambda it, *_, **__: it)

#         # load the image database
#         with open(image_database_path, "rb") as f:
#             image_database = pickle.load(f)
#         keys_image = list(image_database.keys())

#         # Read files from disk and save them to the dataset
#         # generator = parallel_generator(
#         #     lambda fp: (fp, generate_sample_image(image_database[fp])),
#         #     keys_image )
#         # generator = parallel_generator(
#         #     lambda fp: (fp, generate_sample_image(image_database[fp])),
#         #     keys_image )
#         # with h5py.File(filename, mode="x") as dataset:
#         #     for filepath, data in progress_bar(
#         #         generator, desc="Creating dataset", total=len(keys_image)
#         #     ):
#         #         dataset[filepath] = data

#         with h5py.File(filename, mode="x") as dataset:
#             for key in tqdm(keys_image):
#                 dataset[key] = generate_sample_image(image_database[key])

#     @classmethod
#     def create_video_v2(
#         cls,
#         filename: str,
#         clip_database_path: str,
#         dirname: Optional[str] = None,
#         verbose: bool = True,
#     ) -> None:
#         """Create h5py file for dataset from scratch.

#         Args:
#             filename (str): h5py filename.
#             content (List[str]): Dataset content. Requires List[data filepath].
#             dirname (:obj:`str`, optional): Additional dirname for data filepaths.
#                 Default to None.
#             verbose (bool): Verbose option. If True, it would show tqdm progress bar.
#                 Default to True.
#         """
#         dir_tmp = "/idiap/temp/pvuillecard/datasets/CCDbHG/samples/tmp"

#         filename = Path(filename)
#         ext = filename.suffix
#         if ext != ".h5":
#             raise RuntimeError(f"Expected extension to be '.h5', instead got '{ext}'.")
#         dirname = Path("" if dirname is None else dirname)

#         # load the image database
#         with open(clip_database_path, "rb") as f:
#             clip_database = pickle.load(f)
#         key_clips = list(clip_database_path.keys())

#         # Read files from disk and save them to the dataset
#         # generator = parallel_generator(
#         #     lambda fp: (fp, generate_sample_image(image_database[fp])),
#         #     keys_image )
#         # generator = parallel_generator(
#         #     lambda fp: (fp, generate_sample_image(image_database[fp])),
#         #     keys_image )
#         # with h5py.File(filename, mode="x") as dataset:
#         #     for filepath, data in progress_bar(
#         #         generator, desc="Creating dataset", total=len(keys_image)
#         #     ):
#         #         dataset[filepath] = data

#         with h5py.File(filename, mode="x") as dataset:
#             for key_clip in key_clips:
#                 video_decoder = VideoReader(clip_database[key_clip]["clip_path"])
#                 frames = clip_database[key_clip]["frames"]
#                 image_ids = clip_database[key_clip]["image_ids"]
#                 for frame, image_id in tqdm(zip(frames, image_ids), total=len(frames)):
#                     image = video_decoder[frame - 1].asnumpy()
#                     image = Image.fromarray(image)
#                     image.save(os.path.join(dir_tmp, f"{key_clip}_{frame:08d}.jpg"))
#                     dataset[image_id] = np.fromfile(
#                         os.path.join(dir_tmp, f"{key_clip}_{frame:08d}.jpg"), dtype=np.uint8
#                     )
#                     os.remove(os.path.join(dir_tmp, f"{key_clip}_{frame:08d}.jpg"))
