
from numpy import var
import torch 
import math
from gaze_module.utils.metrics_utils import cartesial2spherical, spherical2cartesial
import torch.nn.functional as F

class AngularArcossLoss(torch.nn.Module):
    def __init__(self, compute_only_2d = False):
        super(AngularArcossLoss, self).__init__()
        self.compute_only_2d = compute_only_2d

    def forward(self, output, target, data_id = None):
        # normalize vectors
        if  self.compute_only_2d and data_id in [4,9,10]:
            # apply only to 2d data
            target_v = F.normalize(target[:,:2], p=2, dim=1, eps=1e-8)
        else: 
            target_v = F.normalize(target, p=2, dim=1, eps=1e-8)
      
        if "cartesian" in output.keys():
            
            if self.compute_only_2d and data_id in [4,9,10]:
                output_v = F.normalize(output["cartesian"][:,:2], p=2, dim=1, eps=1e-8)
            else:
                output_v = F.normalize(output["cartesian"], p=2, dim=1, eps=1e-8)

        elif "spherical" in output.keys():
            output_v = spherical2cartesial(output["spherical"])
            output_v = F.normalize(output_v, p=2, dim=1, eps=1e-8)
        else:
            raise ValueError("Output must contain either cartesian or spherical keys")
            
        sim = F.cosine_similarity(output_v, target_v, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
        loss = torch.acos(sim).mean()*180/math.pi
        
        return loss 

class CosineGazeLoss(torch.nn.Module):
    def __init__(self):
        super(CosineGazeLoss, self).__init__()

    def forward(self, output, target):
        target_v = F.normalize(target, p=2, dim=1, eps=1e-8)

        if "cartesian" in output.keys():
            output_v = F.normalize(output["cartesian"], p=2, dim=1, eps=1e-8)
        elif "spherical" in output.keys():
            output_v = spherical2cartesial(output["spherical"])
            output_v = F.normalize(output_v, p=2, dim=1, eps=1e-8)
        else:
            raise ValueError("Output must contain either cartesian or spherical keys")
        
        sim = F.cosine_similarity(output_v, target_v, dim=1, eps=1e-8)
        loss = torch.mean(1 - sim)
        
        return loss

class VMFDistributionLoss(torch.nn.Module):
    """
    this loss is not stable
    """
    def __init__(self):
        super(VMFDistributionLoss, self).__init__()

    def vMF3_log_likelihood(self, y_true, mu_pred, kappa_pred):
        """ define in the paper """
        cosin_dist = torch.sum(y_true * mu_pred, dim=1)
        log_likelihood = kappa_pred * cosin_dist + torch.log(kappa_pred) - torch.log(1 - torch.exp(-2*kappa_pred)) - kappa_pred
        return log_likelihood
    
    def forward(self, output, target):
        target_v = F.normalize(target, p=2, dim=1, eps=1e-8)

        if "cartesian" in output.keys():
            output_v = F.normalize(output["cartesian"], p=2, dim=1, eps=1e-8)
        elif "spherical" in output.keys():
            output_v = spherical2cartesial(output["spherical"])
            output_v = F.normalize(output_v, p=2, dim=1, eps=1e-8)
        else:
            raise ValueError("Output must contain either cartesian or spherical keys")
        
        var = output["var"]
        cosin_dist = torch.sum(target_v * output_v, dim=1)
        log_likelihood = var * cosin_dist + torch.log(var) - torch.log(1 - torch.exp(-2*var)) - var

        return  - torch.mean(log_likelihood)


class PinBallLoss(torch.nn.Module):
    def __init__(self, mode = "cartesian"):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1 - self.q1
        self.mode = mode

    def forward(self, output, target):
        # print('loss pinball')

        if "cartesian" in output.keys() and self.mode != "no_norm":
            output_v = torch.nn.functional.normalize(output["cartesian"], p=2, dim=1)
            target_v = torch.nn.functional.normalize(target, p=2, dim=1)
        elif "cartesian" in output.keys() and self.mode == "no_norm":
            output_v = output["cartesian"]
            target_v = target
        elif "spherical" in output.keys():
            output_v = output["spherical"]
            target_v = target
        else:
            raise ValueError(f"Invalid mode: {self.mode} in PinBallLoss")
        
        var_o = output["var"]
        
        q_10 = target_v - (output_v - var_o)
        q_90 = target_v - (output_v + var_o)

        loss_10 = torch.max(self.q1 * q_10, (self.q1 - 1) * q_10)
        loss_90 = torch.max(self.q9 * q_90, (self.q9 - 1) * q_90)

        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10 + loss_90

 
class AleotoricLoss(torch.nn.Module):
    """Laplacian losses https://arxiv.org/pdf/2105.09803"""
    def __init__(self, mode = "cartesian"):
        super(AleotoricLoss, self).__init__()
        self.mode = mode

    def forward(self, output_vector, target_vector, var_o):

        k = output_vector.size(1)
        if self.mode == "cartesian":
            output_v = torch.nn.functional.normalize(output_vector, p=2, dim=1)
            target_v = torch.nn.functional.normalize(target_vector, p=2, dim=1)
        elif self.mode == "spherical":
            output_v = cartesial2spherical(output_vector)
            target_v = cartesial2spherical(target_vector)
            
        loss = torch.mean(var_o + (1/torch.exp(var_o))*torch.norm(output_v - target_v, p=2,dim=1))
        return loss

class GazeLoss(torch.nn.Module):
    def __init__(
        self, 
        main : torch.nn.modules = None,
        add : torch.nn.modules = None,
        alpha : float = 1.,
        beta : float = 1.
        ):
        super(GazeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.add = add
        self.main = main

    def forward(self, output_vector, target_vector, var_o):
        
        assert self.add is not None or self.main is not None, "At least one loss function must be provided"

        if self.add is None and self.main is not None:
            loss = self.main(output_vector, target_vector)
        elif self.add is not None and self.main is None:
            loss = self.add(output_vector, target_vector, var_o) 
        else:
            loss = self.alpha*self.main(output_vector, target_vector) + \
                    self.beta*self.add(output_vector, target_vector, var_o)
        
        return loss
    
class GazeLossMulti(torch.nn.Module):

    def __init__(self ):
        super(GazeLossMulti, self).__init__()
        
        self.l1_loss = torch.nn.L1Loss()
        self.cosine = AngularArcossLoss()
        self.pinball = PinBallLoss(mode= "cartesian")

    def forward(self, output_3d, output_2d, target, var_o):
        
        out_3d_2d_norm = torch.nn.functional.normalize(output_3d[:,:2], p=2, dim=1)
        out_2d_norm = torch.nn.functional.normalize(output_2d, p=2, dim=1)

        loss_self = self.l1_loss( out_3d_2d_norm, out_2d_norm)
        if target.shape[1] == 3:
            loss_gaze_3d = self.cosine(output_3d, target)
            loss_gaze_2d = self.cosine(output_2d, target[:,:2]) 
        else:
            loss_gaze_3d = self.cosine(output_3d[:,:2], target[:,:2]) 
            loss_gaze_2d = self.cosine(output_2d, target[:,:2])
        
        loss = loss_self + loss_gaze_3d + loss_gaze_2d
        
        return loss