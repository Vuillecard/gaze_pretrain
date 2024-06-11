
import copy
import os 
import pickle
import shutil
from time import time
from multiprocessing.pool import ThreadPool

from typing import Any, Dict, List, Optional, Tuple

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from gaze_module.data.components.utils import (
    BatchSamplerCombined,
    BatchSamplerSequential,
)
from gaze_module.utils import RankedLogger

log = RankedLogger(__name__)

class ConcatenateDataModule(LightningDataModule):
    """`LightningDataModule` to combined dataset for training with multiple datasets

    This method rely on ConcatDataset to combine multiple datasets into one.
    We develop a custom batch sampler such that one batch is batch from one dataset

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        datasets_train: List,
        datasets_test: List,
        train_transform,
        test_transform ,
        sampling_dataset: str = "max",
        num_workers: int = 0,
        batch_size: int = 8,
        pin_memory: bool = False,
        data_to_cluster: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.sampling_dataset = sampling_dataset

        self.batch_size = batch_size
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # In case we run on slurm, we copy the data to the local node for fast reading
        pass
        # if self.hparams.data_to_cluster:
        #     try:
        #         tmp_dir = os.environ["TMPDIR"]
        #         self.data_location = os.path.join(tmp_dir, f"data_{self.data_name}")
        #     except:
        #         self.data_location = None

        #     if self.data_location:
        #         # load the clip database

        #         for dataset in self.dataset_name: 
        #         os.makedirs(self.data_location, exist_ok=True)
        #         with open(
        #             os.path.join(self.data_dir, f"{self.data_name}_image_database.pkl"), "rb"
        #         ) as f:
        #             image_database = pickle.load(f)

        #         def _copy_image_to_tmp(image_dict):
        #             src = image_dict["image_path_crop"]
        #             image_name = image_dict["image_path_crop"].split("/")[-1]
        #             dst = os.path.join(self.data_location, image_name)
        #             if not os.path.exists(dst):
        #                 shutil.copyfile(src, dst)

        #         # Copy images to tmp dir using multiprocessing
        #         # Faster IO than reading from NFS on the cluster
        #         log.info(f"Copying images to tmp dir: {self.data_location}")
        #         start = time()
        #         image_dicts = list(image_database.values())
        #         total_imgs = len(image_dicts)
        #         with ThreadPool(4) as p:
        #             p.map(_copy_image_to_tmp, image_dicts)
        #         end = time()
        #         log.info(f"Done copying {total_imgs} images to tmp dir took: {end-start} s")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # train
        train_data = [dataset(split = 'train',transform=self.train_transform) for dataset in self.hparams.datasets_train]
        self.data_train = ConcatDataset(
            [copy.deepcopy(data) for data in train_data]
        )
        self.train_datasets_size = [
            len(data) for data in train_data
        ]

        if self.sampling_dataset == "max":
            sampling_size = max(self.train_datasets_size)
        elif self.sampling_dataset == "min":
            sampling_size = min(self.train_datasets_size)
        elif self.sampling_dataset == "mean":
            sampling_size = int(sum(self.train_datasets_size) / len(self.train_datasets_size))
        else:
            raise ValueError(f"Unknown sampling dataset method: {self.sampling_dataset}")
        print(f"Sampling size: {sampling_size}")

        self.samplers_train = [
            RandomSampler(
                range(len(data)), replacement=True, num_samples=sampling_size
            )
            for data in train_data
        ]
        del train_data

        # val
        val_data = [dataset(split = 'validation',transform=self.test_transform) for dataset in self.hparams.datasets_train]
        self.data_val = ConcatDataset(
            [copy.deepcopy(data) for data in val_data]
        )
        self.samplers_sequential_val = [
            SequentialSampler(range(len(data)))
            for data in val_data
        ]
        del val_data

        # test
        test_data = [dataset(split = 'test',transform=self.test_transform) for dataset in self.hparams.datasets_test]
        self.data_test = ConcatDataset(
            [copy.deepcopy(data) for data in test_data]
        )
        self.samplers_sequential_test = [
            SequentialSampler(range(len(data)))
            for data in test_data
        ]
        del test_data

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The combined loader max_size_cycle oversample the smallest dataset.
        """
        combined_sampler = BatchSamplerCombined(
            self.samplers_train, self.train_datasets_size, batch_size=self.batch_size_per_device
        )
        return DataLoader(
            self.data_train,
            batch_sampler=combined_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        sequential_sampler = BatchSamplerSequential(
            self.samplers_sequential_val, batch_size=self.batch_size_per_device
        )
        return DataLoader(
            self.data_val,
            batch_sampler=sequential_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        sequential_sampler = BatchSamplerSequential(
            self.samplers_sequential_test, batch_size=self.batch_size_per_device
        )
        return DataLoader(
            self.data_test,
            batch_sampler=sequential_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )