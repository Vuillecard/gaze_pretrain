
import torch 
import math
from gaze_module.utils.metrics_utils import cartesial2spherical
import torch.nn.functional as F

class AngularArcossLoss(torch.nn.Module):
    def __init__(self):
        super(AngularArcossLoss, self).__init__()

    def forward(self, output_vector, target_vector):
        # normalize vectors
        output_v = F.normalize(output_vector, p=2, dim=1, eps=1e-8)
        target_v = F.normalize(target_vector, p=2, dim=1, eps=1e-8)

        sim = F.cosine_similarity(output_v, target_v, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
        loss = torch.acos(sim).mean()*180/math.pi
        
        return loss 

class CosineGazeLoss(torch.nn.Module):
    def __init__(self):
        super(CosineGazeLoss, self).__init__()

    def forward(self, output_vector, target_vector):
        # normalize vectors
        output_v = F.normalize(output_vector, p=2, dim=1, eps=1e-8)
        target_v = F.normalize(target_vector, p=2, dim=1, eps=1e-8)

        sim = F.cosine_similarity(output_v, target_v, dim=1, eps=1e-8)
        loss = torch.mean*(1 - sim)
        
        return loss


class PinBallLoss(torch.nn.Module):
    def __init__(self, mode = "cartesian"):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1 - self.q1
        self.mode = mode

    def forward(self, output_vector, target_vector, var_o):

        if self.mode == "cartesian":
            output_v = torch.nn.functional.normalize(output_vector, p=2, dim=1)
            target_v = torch.nn.functional.normalize(target_vector, p=2, dim=1)
        elif self.mode == "cartesian_to_spherical":
            output_v = cartesial2spherical(output_vector)
            target_v = cartesial2spherical(target_vector)
            output_v =  output_v * 180/math.pi
            target_v =  target_v * 180/math.pi
            var_o = math.pi * torch.nn.Tanh()(var_o)
        elif self.mode == "spherical":
            output_v = output_vector
            target_v = target_vector
        else:
            raise ValueError(f"Invalid mode: {self.mode} in PinBallLoss")
        
        #print(output_v.size(), target_v.size(), var_o.size())
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
            
        loss = torch.mean(var_o*k + (1/torch.exp(var_o))*torch.norm(output_v - target_v, p=1,dim=1))
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