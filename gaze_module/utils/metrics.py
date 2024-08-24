import math
import warnings
import pickle
import os 
import numpy as np
import scipy
import torch
from sklearn.exceptions import UndefinedMetricWarning
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from gaze_module.utils.metrics_utils import (
    compute_angular_error,
    compute_angular_error_cartesian,
    spherical2cartesial
)
from gaze_module.data.components.gaze_dataset import DATASET_ID

# filter the warnings about ill-defined P,R and F1
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

class AngularError(Metric):
    """Angular error metric for gaze estimation used in Gaze360"""

    def __init__(self, mode="spherical"):
        super().__init__()
        self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.mode = mode

    def update(self, input, target):
        if self.mode == "spherical":
            self.total_error += compute_angular_error(input, target)
        elif self.mode == "cartesian":
            self.total_error += compute_angular_error_cartesian(input, target)
        self.total_samples += input.size(0)

    def compute(self):
        return self.total_error / self.total_samples


class PredictionSave(Metric):
    def __init__(self) -> None:
        super().__init__(compute_on_cpu=True, compute_with_cache=False)

        self.add_state("frame_pred", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("frame_gt", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("frame_id", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("video_id", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("person_id", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("data_id", default=[], dist_reduce_fx="cat", persistent=True)

    def update(self, pred, gt, frame_id, video_id, person_id, data_id):
        self.frame_pred += pred
        self.frame_gt += gt
        self.frame_id += frame_id
        self.video_id += video_id
        self.person_id += person_id
        self.data_id += data_id

    def compute(self):
        if len(self.frame_pred) == 0:
            return {"frame_pred": None, "frame_gt": None, "frame_id": None, 
                    "video_id": None, "person_id": None, "data_id": None}

        frame_pred = dim_zero_cat(self.frame_pred).cpu()
        frame_gt = dim_zero_cat(self.frame_gt).cpu()
        frame_id = dim_zero_cat(self.frame_id).cpu()
        video_id = dim_zero_cat(self.video_id).cpu()
        person_id = dim_zero_cat(self.person_id).cpu()
        data_id = dim_zero_cat(self.data_id).cpu()

        return {
            "frame_pred": frame_pred,
            "frame_gt": frame_gt,
            "frame_id": frame_id,
            "video_id": video_id,
            "person_id": person_id,
            "data_id": data_id,
        }


def compute_gaze_results(exp_results,mode_angular = 'spherical'):

    angular_error_all = {}
    k = 2 if mode_angular == 'spherical' else 3
    exp_results['frame_pred'] = exp_results['frame_pred'].view(-1, k).numpy()
    exp_results['frame_gt'] = exp_results['frame_gt'].view(-1, k).numpy()
    exp_results['frame_id'] = exp_results['frame_id'].view(-1).numpy()
    exp_results['video_id'] = exp_results['video_id'].view(-1).numpy()
    exp_results['person_id'] = exp_results['person_id'].view(-1).numpy()
    exp_results['data_id'] = exp_results['data_id'].view(-1).numpy()

    with open('/idiap/temp/pvuillecard/projects/face_analyser/datasets/Gaze360/gaze360_image_database.pkl', 'rb') as f:
        image_db_gaze360 = pickle.load(f)
    
    # include the face information in the prediction 
    for k in image_db_gaze360.keys():
        face_info = image_db_gaze360[k]['other']['person_face_bbox']
        is_face = face_info[0] != -1
        image_db_gaze360[k]['face_info'] = 1 if is_face else 0
        gaze_dir = image_db_gaze360[k]['other']['gaze_dir']
        # compute the angular error with center (0,0,-1)
        angular_error = 180/np.pi*np.arccos(np.dot(gaze_dir, np.array([0, 0, -1])) / (np.linalg.norm(gaze_dir) * np.linalg.norm(np.array([0, 0, -1]))))
        image_db_gaze360[k]['angular_error'] = angular_error

    with open('/idiap/temp/pvuillecard/projects/face_analyser/datasets/GFIE/gfie_image_database.pkl', 'rb') as f:
        image_db_gfie = pickle.load(f)
    for k in image_db_gfie.keys():
        gaze_dir = image_db_gfie[k]['other']['gaze_direction']
        # compute the angular error with center (0,0,-1)
        angular_error = 180/np.pi*np.arccos(np.dot(gaze_dir, np.array([0, 0, -1])) / (np.linalg.norm(gaze_dir) * np.linalg.norm(np.array([0, 0, -1]))))
        image_db_gfie[k]['angular_error'] = angular_error

    with open('/idiap/temp/pvuillecard/projects/face_analyser/datasets/MPSGaze/mpsgaze_image_database.pkl', 'rb') as f:
        image_db_mpsgaze = pickle.load(f)
    for k in image_db_mpsgaze.keys():
        
        gaze_dir = image_db_mpsgaze[k]['other']['gaze_pitch_yaw'][::-1]
        gaze_dir = spherical2cartesial(torch.Tensor(np.array([gaze_dir]))).numpy()[0]
        # compute the angular error with center (0,0,-1)
        angular_error = 180/np.pi*np.arccos(np.dot(gaze_dir, np.array([0, 0, -1])) / (np.linalg.norm(gaze_dir) * np.linalg.norm(np.array([0, 0, -1]))))
        image_db_mpsgaze[k]['angular_error'] = angular_error

        face_info = image_db_mpsgaze[k]['other']['face_bbox_xyxy']
        face_size = face_info[2] - face_info[0]
        if face_size < 30:
            size = None
        elif face_size < 60:
            size = 30
        elif face_size < 90:
            size = 60
        elif face_size < 120:
            size = 90
        elif face_size < 150:
            size = 120
        elif face_size < 180:
            size = 150
        elif face_size < 210:
            size = 180
        elif face_size < 240:
            size = 210
        else :
            size = 240
        image_db_mpsgaze[k]['face_size'] = size


    with open('/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/gaze360_model/run_2024-06-19_11-13-54/logs/train/runs/run_0/prediction/pred_gaze_gazefollow.pkl', 'rb') as f:
        image_db_follow = pickle.load(f)

    for k in image_db_follow.keys():
        gaze_dir = image_db_follow[k]['gaze_vector_pred']
        # compute the angular error with center (0,0,-1)
        angular_error = 180/np.pi*np.arccos(np.dot(gaze_dir, np.array([0, 0, -1])) / (np.linalg.norm(gaze_dir) * np.linalg.norm(np.array([0, 0, -1]))))
        image_db_follow[k]['angular_error'] = angular_error
    
    with open('/idiap/temp/pvuillecard/projects/face_analyser/datasets/Eyediap/eyediap_image_database.pkl', 'rb') as f:
        image_db_eyediap = pickle.load(f)
    
    # with open('/idiap/temp/pvuillecard/projects/face_analyser/datasets/MPIIFace/mpiiface_database.pkl', 'rb') as f:
    #     image_db_mpiiface = pickle.load(f)
    #################################################################################################
    # Gaze360 TESTING
    #################################################################################################

    pred_gaze360 = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 1:
            pred_gaze360[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }
    if len(pred_gaze360) > 0:
        # gaze 360 error
        angular_error_full = AngularError(mode = mode_angular)
        angular_error_back = AngularError(mode = mode_angular)
        angular_error_180 = AngularError(mode = mode_angular)
        angular_error_20 = AngularError(mode = mode_angular)
        angular_error_det = AngularError(mode = mode_angular)
        angular_error_det_180 = AngularError(mode = mode_angular)
        angular_error_det_20 = AngularError(mode = mode_angular)

        for k in pred_gaze360.keys():

            angular_error_full.update(torch.from_numpy(pred_gaze360[k]['frame_gt']).unsqueeze(0),
                                torch.from_numpy(pred_gaze360[k]['frame_pred']).unsqueeze(0))
            if image_db_gaze360[k]['angular_error'] <= 90:
                angular_error_180.update(torch.from_numpy(pred_gaze360[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gaze360[k]['frame_pred']).unsqueeze(0))
            if image_db_gaze360[k]['angular_error'] > 90:
                angular_error_back.update(torch.from_numpy(pred_gaze360[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gaze360[k]['frame_pred']).unsqueeze(0))
            if image_db_gaze360[k]['angular_error'] <= 20:
                angular_error_20.update(torch.from_numpy(pred_gaze360[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gaze360[k]['frame_pred']).unsqueeze(0))
            if image_db_gaze360[k]['face_info'] == 1:
                angular_error_det.update(torch.from_numpy(pred_gaze360[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gaze360[k]['frame_pred']).unsqueeze(0))
                if image_db_gaze360[k]['angular_error'] <= 90:
                    angular_error_det_180.update(torch.from_numpy(pred_gaze360[k]['frame_gt']).unsqueeze(0),
                                                torch.from_numpy(pred_gaze360[k]['frame_pred']).unsqueeze(0))
                if image_db_gaze360[k]['angular_error'] <= 20:
                    angular_error_det_20.update(torch.from_numpy(pred_gaze360[k]['frame_gt']).unsqueeze(0),
                                                torch.from_numpy(pred_gaze360[k]['frame_pred']).unsqueeze(0))
                
        angular_error_all["Gaze360_full"] = angular_error_full.compute()
        angular_error_all["Gaze360_back"] = angular_error_back.compute()
        angular_error_all["Gaze360_180"] = angular_error_180.compute()
        angular_error_all["Gaze360_20"] = angular_error_20.compute()
        angular_error_all["Gaze360_face"] = angular_error_det.compute()
        angular_error_all["Gaze360_face_180"] = angular_error_det_180.compute()
        angular_error_all["Gaze360_face_20"] = angular_error_det_20.compute()

    pred_gaze360video = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 7:
            pred_gaze360video[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }
    if len(pred_gaze360video) > 0:
        # gaze 360 error
        angular_error_full = AngularError(mode = mode_angular)
        angular_error_back = AngularError(mode = mode_angular)
        angular_error_180 = AngularError(mode = mode_angular)
        angular_error_20 = AngularError(mode = mode_angular)
        angular_error_det = AngularError(mode = mode_angular)
        angular_error_det_180 = AngularError(mode = mode_angular)
        angular_error_det_20 = AngularError(mode = mode_angular)

        for k in pred_gaze360video.keys():

            angular_error_full.update(torch.from_numpy(pred_gaze360video[k]['frame_gt']).unsqueeze(0),
                                torch.from_numpy(pred_gaze360video[k]['frame_pred']).unsqueeze(0))
            if image_db_gaze360[k]['angular_error'] <= 90:
                angular_error_180.update(torch.from_numpy(pred_gaze360video[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gaze360video[k]['frame_pred']).unsqueeze(0))
            if image_db_gaze360[k]['angular_error'] > 90:
                angular_error_back.update(torch.from_numpy(pred_gaze360video[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gaze360video[k]['frame_pred']).unsqueeze(0))
            if image_db_gaze360[k]['angular_error'] <= 20:
                angular_error_20.update(torch.from_numpy(pred_gaze360video[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gaze360video[k]['frame_pred']).unsqueeze(0))
            if image_db_gaze360[k]['face_info'] == 1:
                angular_error_det.update(torch.from_numpy(pred_gaze360video[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gaze360video[k]['frame_pred']).unsqueeze(0))
                if image_db_gaze360[k]['angular_error'] <= 90:
                    angular_error_det_180.update(torch.from_numpy(pred_gaze360video[k]['frame_gt']).unsqueeze(0),
                                                torch.from_numpy(pred_gaze360video[k]['frame_pred']).unsqueeze(0))
                if image_db_gaze360[k]['angular_error'] <= 20:
                    angular_error_det_20.update(torch.from_numpy(pred_gaze360video[k]['frame_gt']).unsqueeze(0),
                                                torch.from_numpy(pred_gaze360video[k]['frame_pred']).unsqueeze(0))
                
        angular_error_all["Gaze360video_full"] = angular_error_full.compute()
        angular_error_all["Gaze360video_back"] = angular_error_back.compute()
        angular_error_all["Gaze360video_180"] = angular_error_180.compute()
        angular_error_all["Gaze360video_20"] = angular_error_20.compute()
        angular_error_all["Gaze360video_face"] = angular_error_det.compute()
        angular_error_all["Gaze360video_face_180"] = angular_error_det_180.compute()
        angular_error_all["Gaze360video_face_20"] = angular_error_det_20.compute()
    

    #################################################################################################
    # GFIE TESTING
    #################################################################################################
    pred_gifie = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 2:
            pred_gifie[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }
    angular_error_full = AngularError(mode = mode_angular)
    angular_error_180 = AngularError(mode = mode_angular)
    angular_error_20 = AngularError(mode = mode_angular)
    
    if len(pred_gifie) > 0:
        for k in pred_gifie.keys():

            angular_error_full.update(torch.from_numpy(pred_gifie[k]['frame_gt']).unsqueeze(0),
                                torch.from_numpy(pred_gifie[k]['frame_pred']).unsqueeze(0))
            if image_db_gfie[k]['angular_error'] <= 90:
                angular_error_180.update(torch.from_numpy(pred_gifie[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gifie[k]['frame_pred']).unsqueeze(0))
            if image_db_gfie[k]['angular_error'] <= 20:
                angular_error_20.update(torch.from_numpy(pred_gifie[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gifie[k]['frame_pred']).unsqueeze(0))
            
        angular_error_all["GFIE_full"] = angular_error_full.compute()
        angular_error_all["GFIE_180"] = angular_error_180.compute()
    
    pred_gifievideo = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 8:
            pred_gifievideo[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }
    angular_error_full = AngularError(mode = mode_angular)
    angular_error_180 = AngularError(mode = mode_angular)
    angular_error_20 = AngularError(mode = mode_angular)
    
    if len(pred_gifievideo) > 0:
        for k in pred_gifievideo.keys():

            angular_error_full.update(torch.from_numpy(pred_gifievideo[k]['frame_gt']).unsqueeze(0),
                                torch.from_numpy(pred_gifievideo[k]['frame_pred']).unsqueeze(0))
            if image_db_gfie[k]['angular_error'] <= 90:
                angular_error_180.update(torch.from_numpy(pred_gifievideo[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gifievideo[k]['frame_pred']).unsqueeze(0))
            if image_db_gfie[k]['angular_error'] <= 20:
                angular_error_20.update(torch.from_numpy(pred_gifievideo[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_gifievideo[k]['frame_pred']).unsqueeze(0))
            
        angular_error_all["GFIEvideo_full"] = angular_error_full.compute()
        angular_error_all["GFIEvideo_180"] = angular_error_180.compute()
        
    #################################################################################################
    # GazeFollow TESTING
    #################################################################################################
    pred_follow = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 4:
            pred_follow[f'frame_{exp_results["frame_id"][i]:08d}_face_{exp_results["person_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }
    angular_error_full = AngularError(mode = mode_angular)
    angular_error_180 = AngularError(mode = mode_angular)
    angular_error_20 = AngularError(mode = mode_angular)
    angular_error_2D = AngularError(mode = mode_angular)
    
    if len(pred_follow) > 0:
        for k in pred_follow.keys():
            
            angular_error_2D.update(torch.from_numpy(pred_follow[k]['frame_gt']).unsqueeze(0)[:,:2],
                                torch.from_numpy(pred_follow[k]['frame_pred']).unsqueeze(0)[:,:2])
            
            angular_error_full.update(torch.from_numpy(pred_follow[k]['frame_gt']).unsqueeze(0),
                                torch.from_numpy(pred_follow[k]['frame_pred']).unsqueeze(0))
            if image_db_follow[k]['angular_error'] <= 90:
                angular_error_180.update(torch.from_numpy(pred_follow[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_follow[k]['frame_pred']).unsqueeze(0))
            if image_db_follow[k]['angular_error'] <= 20:
                angular_error_20.update(torch.from_numpy(pred_follow[k]['frame_gt']).unsqueeze(0),
                                            torch.from_numpy(pred_follow[k]['frame_pred']).unsqueeze(0))
            
        angular_error_all["GazeFollow_full"] = angular_error_full.compute()
        angular_error_all["GazeFollow_180"] = angular_error_180.compute()
        angular_error_all["GazeFollow_20"] = angular_error_20.compute()
        angular_error_all["GazeFollow_2D"] = angular_error_2D.compute()

    #################################################################################################
    # MPSGAZE TESTING
    #################################################################################################
    pred_mpsgaze = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 3:
            pred_mpsgaze[f'frame_{exp_results["frame_id"][i]:08d}_face_{exp_results["person_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }
    
    angular_error_full = AngularError(mode = mode_angular)
    angular_error = { 30*i: AngularError(mode = mode_angular) for i in range(1, 9)}

    for k in pred_mpsgaze.keys():
        face_size = image_db_mpsgaze[k]['face_size']
        angular_error_full.update(torch.from_numpy(pred_mpsgaze[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_mpsgaze[k]['frame_pred']).unsqueeze(0))
        if face_size is not None:
            angular_error[face_size].update(torch.from_numpy(pred_mpsgaze[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_mpsgaze[k]['frame_pred']).unsqueeze(0))
    
    angular_error_all[f"MPSGaze_all"] = angular_error_full.compute()
    for k in angular_error.keys():
        angular_error_all[f"MPSGaze_{k}"] = angular_error[k].compute()
    
    #################################################################################################
    # Eyediap TESTING
    #################################################################################################
    
    pred_eyediap = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 6:
            pred_eyediap[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }
    
    angular_error_ft_M = AngularError(mode = mode_angular)
    angular_error_ft_S = AngularError(mode = mode_angular)
    angular_error_ft_all = AngularError(mode = mode_angular)

    angular_error_cs = AngularError(mode = mode_angular)

    for k in pred_eyediap.keys():
        
        if image_db_eyediap[k]['task'] == 'FT':
            angular_error_ft_all.update(torch.from_numpy(pred_eyediap[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_eyediap[k]['frame_pred']).unsqueeze(0))
            if image_db_eyediap[k]['static'] == 'M':
                angular_error_ft_M.update(torch.from_numpy(pred_eyediap[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_eyediap[k]['frame_pred']).unsqueeze(0))
            else:
                angular_error_ft_S.update(torch.from_numpy(pred_eyediap[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_eyediap[k]['frame_pred']).unsqueeze(0))
        else:
            angular_error_cs.update(torch.from_numpy(pred_eyediap[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_eyediap[k]['frame_pred']).unsqueeze(0))
            
    angular_error_all[f"Eyediap_FT"] = angular_error_ft_all.compute()
    angular_error_all[f"Eyediap_CS"] = angular_error_cs.compute()
    angular_error_all[f"Eyediap_FT_M"] = angular_error_ft_M.compute()
    angular_error_all[f"Eyediap_FT_S"] = angular_error_ft_S.compute()
    
    pred_eyediap_video = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 11:
            pred_eyediap_video[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }
    
    angular_error_ft_M = AngularError(mode = mode_angular)
    angular_error_ft_S = AngularError(mode = mode_angular)
    angular_error_ft_all = AngularError(mode = mode_angular)

    angular_error_cs = AngularError(mode = mode_angular)

    for k in pred_eyediap_video.keys():
        
        if image_db_eyediap[k]['task'] == 'FT':
            angular_error_ft_all.update(torch.from_numpy(pred_eyediap_video[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_eyediap_video[k]['frame_pred']).unsqueeze(0))
            if image_db_eyediap[k]['static'] == 'M':
                angular_error_ft_M.update(torch.from_numpy(pred_eyediap_video[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_eyediap_video[k]['frame_pred']).unsqueeze(0))
            else:
                angular_error_ft_S.update(torch.from_numpy(pred_eyediap_video[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_eyediap_video[k]['frame_pred']).unsqueeze(0))
        else:
            angular_error_cs.update(torch.from_numpy(pred_eyediap_video[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_eyediap_video[k]['frame_pred']).unsqueeze(0))
            
    angular_error_all[f"Eyediapvideo_FT"] = angular_error_ft_all.compute()
    angular_error_all[f"Eyediapvideo_CS"] = angular_error_cs.compute()
    angular_error_all[f"Eyediapvideo_FT_M"] = angular_error_ft_M.compute()
    angular_error_all[f"Eyediapvideo_FT_S"] = angular_error_ft_S.compute()


    #################################################################################################
    # MPIIFace TESTING
    #################################################################################################
    
    pred_mpiiface = {}
    for i in range(len(exp_results['frame_id'])):
        if exp_results['data_id'][i] == 12:
            pred_mpiiface[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}'] = {
                'frame_pred': exp_results['frame_pred'][i],
                'frame_gt': exp_results['frame_gt'][i]
            }

    angular_error = AngularError(mode = mode_angular)

    for k in pred_mpiiface.keys():
        angular_error.update(torch.from_numpy(pred_mpiiface[k]['frame_gt']).unsqueeze(0),
                            torch.from_numpy(pred_mpiiface[k]['frame_pred']).unsqueeze(0))
            
    angular_error_all[f"MPIIFace"] = angular_error.compute()
 
    # round the results 
    for k in angular_error_all.keys():
        angular_error_all[k] = round(angular_error_all[k].item(),2)
    
    return angular_error_all


def save_pred_gaze_results(exp_results,output_dir,mode_angular = 'spherical'):

    k = 2 if mode_angular == 'spherical' else 3
    exp_results['frame_pred'] = exp_results['frame_pred'].view(-1, k).numpy()
    exp_results['frame_gt'] = exp_results['frame_gt'].view(-1, k).numpy()
    exp_results['frame_id'] = exp_results['frame_id'].view(-1).numpy()
    exp_results['video_id'] = exp_results['video_id'].view(-1).numpy()
    exp_results['person_id'] = exp_results['person_id'].view(-1).numpy()
    exp_results['data_id'] = exp_results['data_id'].view(-1).numpy()

    # get unique data_id 
    data_ids = np.unique(exp_results['data_id'])

    for data_id in data_ids:
        data_save = {}
        for i in range(len(exp_results['frame_id'])):
            if exp_results['data_id'][i] == data_id:
                if data_id in [1,2,6,11]:
                    data_save[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}'] = {
                        'gaze_vector_pred': exp_results['frame_pred'][i]
                    }
                elif data_id in [3,4]:
                    data_save[f'frame_{exp_results["frame_id"][i]:08d}_face_{exp_results["person_id"][i]:08d}'] = {
                        'gaze_vector_pred': exp_results['frame_pred'][i]
                    }
                elif data_id in [5,9,10,13]:
                    data_save[f'clip_{exp_results["video_id"][i]:08d}_frame_{exp_results["frame_id"][i]:08d}_face_{exp_results["person_id"][i]:08d}'] = {
                        'gaze_vector_pred': exp_results['frame_pred'][i]
                    }

        with open(os.path.join(output_dir,f'pred_gaze_{DATASET_ID[data_id]}.pkl'), 'wb') as f:
            pickle.dump(data_save, f)
    