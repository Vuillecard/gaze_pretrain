import os
import io
import numpy as np
from PIL import Image
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from gaze_module.utils.metrics_utils import cartesial2spherical, spherical2cartesial


def get_bbox_in_body(bbox, body_bbox):
    bbox_in_body = np.zeros_like(bbox)
    bbox_in_body[0] = bbox[0] - body_bbox[0]
    bbox_in_body[1] = bbox[1] - body_bbox[1]
    bbox_in_body[2] = bbox[2] - body_bbox[0]
    bbox_in_body[3] = bbox[3] - body_bbox[1]
    bbox_in_body = bbox_in_body.astype(int)
    return bbox_in_body


class BaseImage(Dataset):

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                data_location = None ,
                transform=None):
        self.data_location = data_location
        self.image_db_path = image_db_path
        self.sample_path = sample_path
        self.split = split
        self.head_bbox_name = head_bbox_name
        self.read_mode = "pillow"
        self.transform = transform

        self.setup()

    def setup(self):
        # load image dataset 
        with open(self.image_db_path, 'rb') as f:
            self.data = pickle.load(f)
        # load the sample
        self.sample = pd.read_csv(self.sample_path)
        self.sample = self.sample[self.sample['split'] == self.split]
        self.sample.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.sample)

    def get_path_data(self, path):
        if self.data_location:
            name_data = path.split("/")[-1]
            return os.path.join(self.data_location, name_data)
        return path
    
    def read_image(self, image, frame=None, video_decoder=None):
        if self.read_mode == "pillow":
            if not isinstance(image, (str)):
                image = io.BytesIO(image)
            image = np.array(Image.open(image).convert("RGB"), copy=True)
        elif self.read_mode == "video":
            if frame is None and video_decoder is None:
                raise ValueError("frame must be set when using video decoder")
            image = video_decoder[frame - 1].asnumpy()
        else:
            raise NotImplementedError("use pillow or cv2")

        return image
    
    def __getitem__(self, idx):

        frames_info = self.data[self.sample.iloc[idx]["image_id"]]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"])) 
        ]
        
        sample["frame_id"] = frames_info["frame"]
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["head_bbox"] = [get_bbox_in_body(frames_info["other"][self.head_bbox_name],
                                               frames_info["other"]['head_bbox_crop'])]
        sample["bbox_strategy"] = "fixed_center"

        
        
        # task gaze
        gazes = [
            frames_info["other"]["gaze_dir"]
        ]  # need to process the gaze cf gaze360 git
        gaze_float = torch.Tensor(np.array(gazes))
        normalized_gazes = nn.functional.normalize(gaze_float)
        # print(torch.norm(normalized_gazes, dim=1))
        
        spherical_vector = torch.stack(
            (
                torch.atan2(normalized_gazes[:, 0], -normalized_gazes[:, 2]),
                torch.asin(normalized_gazes[:, 1]),
            ),
            1,
        )
        # only the gaze of the middle frame
        sample["task_gaze"] = spherical_vector[0] # yaw pitch

        if self.transform:
            sample = self.transform(sample)

        return sample

    
class Gaze360Image(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                data_location = None ,
                transform=None):
        super().__init__(image_db_path, sample_path, split, head_bbox_name, data_location, transform)

    
    def __getitem__(self, idx):

        frames_info = self.data[self.sample.iloc[idx]["image_id"]]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"])) 
        ]
        
        sample["frame_id"] = frames_info["frame"]
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["person_id"] = 1
        sample["data_id"] = 1
        sample["head_bbox"] = [get_bbox_in_body(frames_info["other"][self.head_bbox_name],
                                               frames_info["other"]['head_bbox_crop'])]
        sample["bbox_strategy"] = "fixed_center"

        # task gaze
        gazes = [
            frames_info["other"]["gaze_dir"]
        ]  # need to process the gaze cf gaze360 git
        gaze_float = torch.Tensor(np.array(gazes))
        normalized_gazes = nn.functional.normalize(gaze_float, p=2, dim=1)
        spherical_vector = cartesial2spherical(normalized_gazes)
        
        # only the gaze of the middle frame
        sample["task_gaze_yawpitch"] = spherical_vector[0]
        sample["task_gaze_vector"] = normalized_gazes[0]

        if self.transform:
            sample = self.transform(sample)

        return sample


class GFIEImage(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                data_location = None ,
                transform=None):
        super().__init__(image_db_path, sample_path, split, head_bbox_name, data_location, transform)


    def __getitem__(self, idx):

        frames_info = self.data[self.sample.iloc[idx]["image_id"]]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"])) 
        ]
        
        sample["frame_id"] = frames_info["frame"]
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["person_id"] = 1
        sample["data_id"] = 2
        sample["head_bbox"] = [ frames_info["other"]['head_bbox_yolo_in_head_crop'] ]
        sample["bbox_strategy"] = "fixed_center"

        # task gaze
        gazes = [
            frames_info["other"]["gaze_direction"]
        ]  # need to process the gaze cf gaze360 git
        gaze_float = torch.Tensor(np.array(gazes))
        normalized_gazes = nn.functional.normalize(gaze_float)
        spherical_vector = cartesial2spherical(normalized_gazes)
        
        # only the gaze of the middle frame
        sample["task_gaze_yawpitch"] = spherical_vector[0]
        sample["task_gaze_vector"] = normalized_gazes[0]

        if self.transform:
            sample = self.transform(sample)

        return sample


class MPSGazeImage(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                data_location = None ,
                transform=None):
        if split == "test":
            split = "validation"
        super().__init__(image_db_path, sample_path, split, head_bbox_name, data_location, transform)


    def __getitem__(self, idx):

        frames_info = self.data[self.sample.iloc[idx]["image_id"]]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"])) 
        ]
        
        sample["frame_id"] = int(frames_info["image_i"])
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["person_id"] = int(frames_info["person_id"].replace("face_", ""))
        sample["data_id"] = 3
        sample["head_bbox"] = [ frames_info["other"]['head_bbox_yolo_in_head_crop'] ]
        sample["bbox_strategy"] = "fixed_center"

        # task gaze
        gazes = [
            frames_info["other"]["gaze_pitch_yaw"][::-1]
        ]  # need to process the gaze cf gaze360 git
        gaze_yawpitch = torch.Tensor(np.array(gazes))

        gaze_direction = spherical2cartesial(gaze_yawpitch)
        normalized_gazes = nn.functional.normalize(gaze_direction)
        
        # only the gaze of the middle frame
        sample["task_gaze_yawpitch"] = gaze_yawpitch[0]
        sample["task_gaze_vector"] = normalized_gazes[0]

        if self.transform:
            sample = self.transform(sample)

        return sample
