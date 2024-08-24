import os
import io
from venv import create
import numpy as np
from PIL import Image
import pickle
import pandas as pd
from sympy import sequence
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from gaze_module.utils.metrics_utils import cartesial2spherical, spherical2cartesial

DATASET_ID = {
    1: "gaze360",
    2: "gfie",
    3: "mpsgaze",
    4: "gazefollow",
    5: "childplay",
    6: "eyediap" ,
    7: "gaze360video",
    8: "gfievideo",
    9: "vat",
    10: "vatvideo",
    11: "eyediapvideo",
    12: "mpiiface"
}

def get_bbox_in_body(bbox, body_bbox):
    bbox_in_body = np.zeros_like(bbox)
    bbox_in_body[0] = bbox[0] - body_bbox[0]
    bbox_in_body[1] = bbox[1] - body_bbox[1]
    bbox_in_body[2] = bbox[2] - body_bbox[0]
    bbox_in_body[3] = bbox[3] - body_bbox[1]
    bbox_in_body = bbox_in_body.astype(int)
    return bbox_in_body

def create_window(frame: int, window_size: int, window_stride: int):
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

class BaseAnnotation(): 
    """
    used to store additional dataset information
    """
    def __init__(self,
               name,
               location,
               ):
        self.name = name
        self.location = location
        self.data = None

        self.load_data()

    def load_data(self):
        with open(self.location, 'rb') as f:
            self.data = pickle.load(f)
    
    def get_data(self, key):
        return self.data[key]

class GazeAnnotation(BaseAnnotation):

    def __init__(self, location: str):
        super().__init__("gaze", location)

    def get_data(self, key):
        return self.data[key]['gaze_vector_pred']

class BaseImage(Dataset):

    def __init__(self,
                data_name,
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None, 
                additonal_data = []):
        self.data_name = data_name
        self.data_location = None
        self.image_db_path = image_db_path
        self.sample_path = sample_path
        self.split = split
        self.head_bbox_name = head_bbox_name
        self.read_mode = "pillow"
        self.transform = transform
        self.additonal_data = { add_data.name : add_data for add_data in additonal_data }
        self.base_data_dir = '/idiap/project/epartners4all/data/uniface_database'
        self.base_slurm_dir = '/tmp'
        self.setup()

    def setup(self):
        # load image dataset 
        with open(self.image_db_path, 'rb') as f:
            self.data = pickle.load(f)
        # load the sample
        self.sample = pd.read_csv(self.sample_path)
        if self.split != "all":
            self.sample = self.sample[self.sample['split'] == self.split]
        self.sample.reset_index(drop=True, inplace=True)

        # slurm data location
        if os.path.exists( os.path.join(self.base_slurm_dir, self.data_name)):
            self.data_location = True
            #print(f"using slurm data location for {self.data_name} dataset")
        else:
            self.data_location = False

    def check_data_is_defined(self): 
        key_image = set(self.sample['image_id'].to_list())
        for k, v in self.additonal_data.items():
            key_data = set(v.data.keys())
            if len(key_image - key_data) != 0:
                print(f"missing data in {k} additional data")

    def __len__(self):
        return len(self.sample)

    def get_path_data(self, path):
        if self.data_location:
            new_path = path.replace(self.base_data_dir, self.base_slurm_dir)
            
            return new_path
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
                transform=None):
        super().__init__('Gaze360',image_db_path, sample_path, split, head_bbox_name, transform)

    
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

class Gaze360Video(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None,
                strategy = "fixed_center"):
        super().__init__('Gaze360',image_db_path, sample_path, split, head_bbox_name, transform)
        self.strategy = strategy
        self.window_size = 7
        self.window_stride = 1

        if split in ["train", "validation"]:
            #self.filter_existing_sequence()
            self.downsample_data()
        if split == "test":
            self.check_validity_sequence()

    def get_clip_frame(self, key):
        key_split = key.split("_frame_")
        frame = int(key_split[-1])
        clip = int(key_split[0].replace("clip_", ""))
        return clip, frame
    
    def check_validity_sequence(self):
        all_image_key = set(list(self.data.keys()))
        keys_sample = set(self.sample['image_id'].to_list())
        for key in keys_sample:
            clip, frame = self.get_clip_frame(key)
            sequence_keys = set([f"clip_{clip:08d}_frame_{i:08d}" for i in 
                             create_window(frame, self.window_size, self.window_stride)])
            if not sequence_keys.issubset(all_image_key):
                print(f"missing sequence {key}")
    
    def downsample_data(self): 

        keys = self.sample['image_id'].to_list()
        # sort the keys
        keys.sort()
        # sample with a stride of 3
        keys = keys[::3]
        self.sample = self.sample[self.sample['image_id'].isin(keys)]
        self.sample.reset_index(drop=True, inplace=True)

    def __getitem__(self, idx):

        image_key = self.sample.iloc[idx]["image_id"]
        clip, frame = self.get_clip_frame(image_key)

        # jitter randomly the frame if training
        if self.split == "train":
            frame += np.random.randint(-1, 1)
        
        middle_frame_infos = self.data[image_key]
        sequence_keys = [f"clip_{clip:08d}_frame_{i:08d}" for i in 
                             create_window(frame, self.window_size, self.window_stride)]
        valid_keys = [ key in self.data for key in sequence_keys ]
        frames_infos = [ self.data[key] if valid else middle_frame_infos.copy() 
                        for key, valid in zip(sequence_keys, valid_keys)]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"]))
            for frames_info in frames_infos
        ]
        sample["images"] = [ image if valid else np.zeros_like(image) 
                            for image, valid in zip(sample["images"], valid_keys)]
        
        sample["frame_id"] = middle_frame_infos["frame"]
        sample["clip_id"] = int(middle_frame_infos["clip_id"].replace("clip_", ""))
        sample["person_id"] = 1
        sample["data_id"] = 7
        head_bbox_center = middle_frame_infos["other"]["head_bbox_yolo"]

        if self.strategy == "fixed_center":
            sample["head_bbox"] = [
                get_bbox_in_body(head_bbox_center,
                    frames_info["other"]['head_bbox_crop'])
                        for frames_info in frames_infos]
            
        elif self.strategy == "followed":
            sample["head_bbox"] = [
                get_bbox_in_body(f["other"]["head_bbox_yolo"],
                    f["other"]["head_bbox_crop"])
                        for f in frames_infos
                ]
        else: 
            raise ValueError(f"strategy {self.strategy} not implemented")
        
        sample["bbox_strategy"] = self.strategy

        # task gaze
        gazes = [
            middle_frame_infos["other"]["gaze_dir"]
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
                transform=None, 
                additonal_data = []):
        super().__init__('GFIE',image_db_path, sample_path, split, 
                         head_bbox_name, transform, 
                         additonal_data)

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

class GFIEVideo(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None, 
                additonal_data = [],
                strategy = "fixed_center"):
        super().__init__('GFIE',image_db_path, sample_path, split, 
                         head_bbox_name, transform, 
                         additonal_data)
        self.strategy = strategy
        self.window_size = 7
        self.window_stride = 3 # 30 fps 

        if split in ["train", "validation"]:
            self.downsample_data()
    
    def downsample_data(self): 
        keys = self.sample['image_id'].to_list()
        # sort the keys
        keys.sort()
        # sample with a stride of 3
        keys = keys[::3]
        self.sample = self.sample[self.sample['image_id'].isin(keys)]
        self.sample.reset_index(drop=True, inplace=True)

    def get_clip_frame(self, key):
        key_split = key.split("_frame_")
        frame = int(key_split[-1])
        clip = int(key_split[0].replace("clip_", ""))
        return clip, frame
    
    def __getitem__(self, idx):
        
        image_key = self.sample.iloc[idx]["image_id"]
        clip, frame = self.get_clip_frame(image_key)

        # jitter randomly the frame if training
        if self.split == "train":
            frame += np.random.randint(-1, 1)
        
        middle_frame_infos = self.data[image_key]
        sequence_keys = [f"clip_{clip:08d}_frame_{i:08d}" for i in 
                             create_window(frame, self.window_size, self.window_stride)]
        valid_keys = [ key in self.data for key in sequence_keys ]
        frames_infos = [ self.data[key] if valid else middle_frame_infos.copy() 
                        for key, valid in zip(sequence_keys, valid_keys)]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"])) 
            for frames_info in frames_infos
        ]
        sample["images"] = [ image if valid else np.zeros_like(image) 
                            for image, valid in zip(sample["images"], valid_keys)]
        
        sample["frame_id"] = middle_frame_infos["frame"]
        sample["clip_id"] = int(middle_frame_infos["clip_id"].replace("clip_", ""))
        sample["person_id"] = 1
        sample["data_id"] = 8

        if self.strategy == "followed":
            sample["head_bbox"] = [ frames_info["other"]['head_bbox_yolo_in_head_crop']
                                for frames_info in frames_infos]
        else : 
            raise ValueError(f"strategy {self.strategy} not implemented")

        # task gaze
        gazes = [
            middle_frame_infos["other"]["gaze_direction"]
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
                transform=None):
        if split == "test":
            split = "validation"
        super().__init__('MPSGaze',image_db_path, sample_path, split, head_bbox_name, transform)


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

class GazeFollowImage(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None,
                additonal_data = []):
        
        super().__init__('Gazefollow',image_db_path, sample_path, split, head_bbox_name, transform,
                         additonal_data )

        self.filter_invalide_gaze()
        self.check_data_is_defined()

    def filter_invalide_gaze(self): 
        # filter out invalide gaze
        self.sample['gaze_valid'] = 1
        def _check_for_valide(row):
            gaze_vector = self.data[row["image_id"]]["other"]["gaze_vector"]
            if np.isnan(gaze_vector).any():
                return 0
            return 1
        self.sample['gaze_valid'] = self.sample.apply(_check_for_valide, axis=1)
        self.sample = self.sample[self.sample['gaze_valid'] == 1]
        self.sample.reset_index(drop=True, inplace=True)
    
    def __getitem__(self, idx):
        
        key = self.sample.iloc[idx]["image_id"]
        frames_info = self.data[key]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"])) 
        ]
        
        sample["frame_id"] = int(key.split("_")[1])
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["person_id"] = int(frames_info["person_id"].replace("face_", ""))
        sample["data_id"] = 4
        sample["head_bbox"] = [ frames_info["other"]['head_bbox_yolo_in_head_crop'] ]
        sample["bbox_strategy"] = "fixed_center"

        if "gaze" in self.additonal_data:
            
            # task gaze
            gazes = [
                self.additonal_data["gaze"].get_data(key)
            ]
            # check if nan in target_vector
            gaze_float = torch.Tensor(np.array(gazes))
            normalized_gazes = nn.functional.normalize(gaze_float)
            spherical_vector = cartesial2spherical(normalized_gazes)
            
            # only the gaze of the middle frame
            sample["task_gaze_yawpitch"] = spherical_vector[0]
            sample["task_gaze_vector"] = normalized_gazes[0]

        else:
            # task gaze
            gazes = [
                frames_info["other"]["gaze_vector"]
            ]  # need to process the gaze cf gaze360 git
            gaze_xy = torch.Tensor(np.array(gazes))
            gaze_xy = nn.functional.normalize(gaze_xy, p=2, dim=1)
            # check if nan in target_vector
            if torch.isnan(gaze_xy).any():
                print('dataset target_vector has nan')

            # only the gaze of the middle frame
            sample["task_gaze_yawpitch"] = -1
            sample["task_gaze_vector"] = gaze_xy[0]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ChildPlayImage(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None):
        
        super().__init__('ChildPlay',image_db_path, sample_path, split, head_bbox_name, transform)

        #self.filter_invalide_gaze()

    def filter_invalide_gaze(self): 
        # filter out invalide gaze
        self.sample['gaze_valid'] = 1
        def _check_for_valide(row):
            gaze_vector = self.data[row["image_id"]]["other"]["gaze_vector"]
            if np.isnan(gaze_vector).any():
                return 0
            return 1
        self.sample['gaze_valid'] = self.sample.apply(_check_for_valide, axis=1)
        self.sample = self.sample[self.sample['gaze_valid'] == 1]
        self.sample.reset_index(drop=True, inplace=True)
    
    def __getitem__(self, idx):
        
        key = self.sample.iloc[idx]["image_id"]
        frames_info = self.data[key]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path"])) 
        ]
        
        sample["frame_id"] = int(frames_info["frame_id"].replace("frame_", ""))
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["person_id"] = int(frames_info["person_id"].replace("face_", ""))
        sample["data_id"] = 5
        sample["head_bbox"] = [ frames_info["other"]['head_bbox_xyxy'] ]
        sample["bbox_strategy"] = "fixed_center"

        # task gaze
        # gazes = [
        #     frames_info["other"]["gaze_vector"]
        # ]  # need to process the gaze cf gaze360 git
        # if gazes[0] is None :
        #     sample["task_gaze_yawpitch"] = None
        #     sample["task_gaze_vector"] = None
        # else:
        #     gaze_xy = torch.Tensor(np.array(gazes))
        #     gaze_xy = nn.functional.normalize(gaze_xy, p=2, dim=1)
        #     # check if nan in target_vector
        #     if torch.isnan(gaze_xy).any():
        #         print('dataset target_vector has nan')

            # only the gaze of the middle frame
        sample["task_gaze_yawpitch"] = -1
        sample["task_gaze_vector"] = -1

        if self.transform:
            sample = self.transform(sample)

        return sample 

class EyediapImage(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None):
        super().__init__('Eyediap',image_db_path, sample_path, split, head_bbox_name, transform)

    def __getitem__(self, idx):

        frames_info = self.data[self.sample.iloc[idx]["image_id"]]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path"])) 
        ]
        
        sample["frame_id"] = frames_info["frame"]
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["person_id"] = 1
        sample["data_id"] = 6
        sample["head_bbox"] = [ frames_info["other"]['head_bbox_yolo_xyxy'] ]
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
    
class EyediapVideo(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None):
        super().__init__('Eyediap',image_db_path, sample_path, split, head_bbox_name, transform)
        self.strategy = "followed"
        self.window_size = 7
        self.window_stride = 3

    def get_clip_frame(self, key):
        key_split = key.split("_frame_")
        frame = int(key_split[-1])
        clip = int(key_split[0].replace("clip_", ""))
        return clip, frame
    
    def __getitem__(self, idx):

        image_key = self.sample.iloc[idx]["image_id"]
        clip, frame = self.get_clip_frame(image_key)

        # jitter randomly the frame if training
        if self.split == "train":
            frame += np.random.randint(-1, 1)
        
        middle_frame_infos = self.data[image_key]
        sequence_keys = [f"clip_{clip:08d}_frame_{i:08d}" for i in 
                             create_window(frame, self.window_size, self.window_stride)]
        valid_keys = [ key in self.data for key in sequence_keys ]
        frames_infos = [ self.data[key] if valid else middle_frame_infos.copy() 
                        for key, valid in zip(sequence_keys, valid_keys)]
        
        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path"])) 
            for frames_info in frames_infos
        ]
        sample["images"] = [ image if valid else np.zeros_like(image) 
                            for image, valid in zip(sample["images"], valid_keys)]
        
        sample["frame_id"] = middle_frame_infos["frame"]
        sample["clip_id"] = int(middle_frame_infos["clip_id"].replace("clip_", ""))
        sample["person_id"] = 1
        sample["data_id"] = 11
        #sample["head_bbox"] = [ middle_frame_infos["other"]['head_bbox_yolo_xyxy'] ]
        sample["bbox_strategy"] = self.strategy

        if self.strategy == "followed":
            sample["head_bbox"] = [ frames_info["other"]['head_bbox_yolo_xyxy']
                                for frames_info in frames_infos]
        else : 
            raise ValueError(f"strategy {self.strategy} not implemented")
        
        # task gaze
        gazes = [
            middle_frame_infos["other"]["gaze_direction"]
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
    


class VATImage(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None, 
                additonal_data = []):
        
        super().__init__('VAT',image_db_path, sample_path, split, head_bbox_name, transform,
                         additonal_data)
    
    def __getitem__(self, idx):
        
        key = self.sample.iloc[idx]["image_id"]
        frames_info = self.data[key]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"])) 
        ]
        sample["frame_id"] = int(frames_info["frame_id"].replace("frame_", ""))
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["person_id"] = int(frames_info["person_id"].replace("face_", ""))
        sample["data_id"] = 9
        sample["head_bbox"] = [ frames_info["other"]['head_bbox_yolo_in_head_crop'] ]
        sample["bbox_strategy"] = "fixed_center"

        if "gaze" in self.additonal_data:
            
            # task gaze
            gazes = [
                self.additonal_data["gaze"].get_data(key)
            ]
            # check if nan in target_vector
            gaze_float = torch.Tensor(np.array(gazes))
            normalized_gazes = nn.functional.normalize(gaze_float)
            spherical_vector = cartesial2spherical(normalized_gazes)
            
            # only the gaze of the middle frame
            sample["task_gaze_yawpitch"] = spherical_vector[0]
            sample["task_gaze_vector"] = normalized_gazes[0]

        else:
        
            sample["task_gaze_yawpitch"] = -1
            sample["task_gaze_vector"] = -1

        if self.transform:
            sample = self.transform(sample)

        return sample 
    

class VATVideo(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None,
                additonal_data = [],
                strategy = "fixed_center"):
        
        super().__init__('VAT',image_db_path, sample_path, split, head_bbox_name, transform,
                         additonal_data)
        self.strategy = strategy
        self.window_size = 7
        self.window_stride = 3
        #self.filter_invalide_gaze()
        if split in ["train", "validation"]:
            #self.filter_existing_sequence()
            self.downsample_data()

    def get_clip_frame_face(self, key):
        key_split = key.split("_frame_")
        split_frame_face = key_split[-1]
        frame = int( split_frame_face.split("_face_")[0])
        face = int(split_frame_face.split("_face_")[1])
        clip = int(key_split[0].replace("clip_", ""))
        return clip, frame, face
        
    def downsample_data(self): 

        keys = self.sample['image_id'].to_list()
        # sort the keys
        keys.sort()
        # sample with a stride of 3
        keys = keys[::3]
        self.sample = self.sample[self.sample['image_id'].isin(keys)]
        self.sample.reset_index(drop=True, inplace=True)
    
    def __getitem__(self, idx):
        
        key = self.sample.iloc[idx]["image_id"]
        frames_info = self.data[key]

        image_key = self.sample.iloc[idx]["image_id"]
        clip, frame, face = self.get_clip_frame_face(image_key)

        # jitter randomly the frame if training
        if self.split == "train":
            frame += np.random.randint(-1, 1)
        
        middle_frame_infos = self.data[image_key]
        sequence_keys = [f"clip_{clip:08d}_frame_{i:08d}_face_{face:08d}" for i in 
                             create_window(frame, self.window_size, self.window_stride)]
        # print(sequence_keys)
        valid_keys = [ key in self.data for key in sequence_keys ]
        frames_infos = [ self.data[key] if valid else middle_frame_infos.copy() 
                        for key, valid in zip(sequence_keys, valid_keys)]
        
        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path_crop"])) 
            for frames_info in frames_infos 
        ]
        sample["images"] = [ image if valid else np.zeros_like(image) 
                            for image, valid in zip(sample["images"], valid_keys)]
        
        sample["frame_id"] = int(middle_frame_infos["frame_id"].replace("frame_", ""))
        sample["clip_id"] = int(middle_frame_infos["clip_id"].replace("clip_", ""))
        sample["person_id"] = int(middle_frame_infos["person_id"].replace("face_", ""))
        sample["data_id"] = 10
        
        sample["bbox_strategy"] = self.strategy

        if self.strategy == "followed":
            sample["head_bbox"] = [
                f["other"]["head_bbox_yolo_in_head_crop"]
                        for f in frames_infos
                ]
        else :
            raise ValueError(f"strategy {self.strategy} not implemented")
        
        if "gaze" in self.additonal_data:
            
            # task gaze
            gazes = [
                self.additonal_data["gaze"].get_data(image_key)
            ]
            # check if nan in target_vector
            gaze_float = torch.Tensor(np.array(gazes))
            normalized_gazes = nn.functional.normalize(gaze_float)
            spherical_vector = cartesial2spherical(normalized_gazes)
            
            # only the gaze of the middle frame
            sample["task_gaze_yawpitch"] = spherical_vector[0]
            sample["task_gaze_vector"] = normalized_gazes[0]

        else:
        
            sample["task_gaze_yawpitch"] = -1
            sample["task_gaze_vector"] = -1

        if self.transform:
            sample = self.transform(sample)

        return sample 
    

class MPIIFaceImage(BaseImage): 

    def __init__(self, 
                image_db_path,
                sample_path,
                split,
                head_bbox_name,
                transform=None):
        super().__init__('MPIIFace',image_db_path, sample_path, split, head_bbox_name, transform)

    def __getitem__(self, idx):

        frames_info = self.data[self.sample.iloc[idx]["image_id"]]

        # input setup
        sample = {}
        sample["images"] = [ 
            self.read_image(self.get_path_data(frames_info["image_path"])) 
        ]
        
        sample["frame_id"] = int(frames_info["image_id"].split("_")[1])
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["person_id"] = 1
        sample["data_id"] = 12
        sample["head_bbox"] = [ frames_info["other"]['head_bbox_yolo_xyxy'] ]
        sample["bbox_strategy"] = "fixed_center"

        # task gaze
        gazes = [
            frames_info["other"]["gaze_vector"]
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