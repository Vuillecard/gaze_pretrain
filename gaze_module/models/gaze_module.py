from typing import Any, Dict, Tuple
import os 
import pickle
import json
import math

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric, MeanMetric

from omegaconf import DictConfig
import torch.optim as optim
from torch import nn

from gaze_module.data.components.gaze_dataset import DATASET_ID
from gaze_module.utils.metrics import (
    AngularError, 
    PredictionSave,
    compute_gaze_results,
    save_pred_gaze_results
)

class GazeModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        solver: DictConfig,
        loss: torch.nn.Module,
        compile: bool,
        output_path: str,
        mode_angular: str = "spherical",
        pretrained_path: str = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # set paths
        self.output_path = output_path
        self.val_path = os.path.join(output_path, "metric", "val")
        os.makedirs(self.val_path, exist_ok=True)
        self.test_path = os.path.join(output_path, "metric", "test")
        os.makedirs(self.test_path, exist_ok=True)
        
        self.pred_path = os.path.join(output_path, "prediction")
        os.makedirs(self.pred_path, exist_ok=True)

        #self.net = net.load_model()
        self.net = net
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            weight = torch.load(pretrained_path)['state_dict']
            weight = {k: v for k, v in weight.items() if k.startswith('net')}
            weight = {k.replace('net.', ''): v for k, v in weight.items()}
            self.net.load_state_dict(weight)
        
        #self.output_type = net.output if hasattr(net, "output") else None
        #self.output_type = net.output
        # loss function
        #self.criterion = torch.nn.L1Loss()
        self.criterion = loss

        # metric objects for calculating and averaging accuracy across batches
        self.mode_angular = mode_angular
        self.train_angular = AngularError(mode=self.mode_angular)

        
        self.val_angular = nn.ModuleDict({
            'all': AngularError(mode=self.mode_angular),
            'gaze360': AngularError(mode=self.mode_angular),
            'gaze360video': AngularError(mode=self.mode_angular),
            'gfie': AngularError(mode=self.mode_angular),
            'gfievideo': AngularError(mode=self.mode_angular),
            'mpsgaze': AngularError(mode=self.mode_angular),
            'gazefollow': AngularError(mode=self.mode_angular),
            'eyediap': AngularError(mode=self.mode_angular),
            'eyediapvideo': AngularError(mode=self.mode_angular),
            'vat': AngularError(mode=self.mode_angular),
            'vatvideo': AngularError(mode=self.mode_angular),
            'mpiiface': AngularError(mode=self.mode_angular),
        })
        self.test_angular = nn.ModuleDict({
            'gaze360': AngularError(mode=self.mode_angular),
            'gaze360video': AngularError(mode=self.mode_angular),
            'gfie': AngularError(mode=self.mode_angular),
            'gfievideo': AngularError(mode=self.mode_angular),
            'mpsgaze': AngularError(mode=self.mode_angular),
            'gazefollow': AngularError(mode=self.mode_angular),
            'eyediap': AngularError(mode=self.mode_angular),
            'eyediapvideo': AngularError(mode=self.mode_angular),
            'vat': AngularError(mode=self.mode_angular),
            'vatvideo': AngularError(mode=self.mode_angular),
            'mpiiface': AngularError(mode=self.mode_angular),
        })

        self.test_pred = PredictionSave()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_ang_best = MinMetric()

    def forward(self, x: torch.Tensor, data_id:torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x,data_id)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        for k in self.val_angular: 
            self.val_angular[k].reset()
        self.val_ang_best.reset()

    def model_step(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        data_id = list(set(batch["data_id"].tolist()))
        assert len(data_id) == 1
        data_id = data_id[0]
             
        output = self.forward(batch["images"], data_id)
        
        if "cartesian" in output.keys():
            # if data_id == 4:
            #     k = 2
            # else: 
            #     k = 3
            #output["cartesian"] = output["cartesian"][:]
            target = batch["task_gaze_vector"]
            
        elif "spherical" in output.keys():
            target = batch["task_gaze_yawpitch"]
        else:
            raise ValueError(f"Invalid mode: {self.mode_angular}")
        
        loss = self.criterion(output, target,data_id)
        pred = output["cartesian"] if "cartesian" in output.keys() else output["spherical"]
        return loss, pred, target
        

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        # print(preds.size(), targets.size())
        # print(preds , targets)
        # print(loss)
        # update and log metrics
        self.train_loss(loss)
        self.train_angular(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/angular", self.train_angular, on_step=True, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)

        self.val_angular['all'](preds, targets)
        self.log("val/angular_all", self.val_angular['all'], on_step=False, on_epoch=True, prog_bar=False)

        data_id = list(set(batch["data_id"].cpu().tolist()))
        assert len(data_id) == 1
        self.val_angular[DATASET_ID[data_id[0]]](preds, targets)
        self.log(f"val/angular_{DATASET_ID[data_id[0]]}", self.val_angular[DATASET_ID[data_id[0]]],
                  on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_angular['all'].compute()  # get current val acc
        self.val_ang_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/angular_best", self.val_ang_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)
        data_id = list(set(batch["data_id"].cpu().tolist()))
        assert len(data_id) == 1
        data_id = data_id[0]
        
        if data_id in [1, 2, 3, 4, 6,7,8,11,12]:
            self.test_pred.update(preds, targets, batch["frame_id"], 
                                batch["clip_id"], batch["person_id"], batch["data_id"])
        
        self.test_angular[DATASET_ID[data_id]](preds, targets)
        self.log(f"test/angular_{DATASET_ID[data_id]}", 
                self.test_angular[DATASET_ID[data_id]],
                on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        torch.save(
            self.test_pred.state_dict(),
            os.path.join(
                self.test_path, f"prediction_test_gaze_epoch_{self.current_epoch}.pth"
            ),
        )
        save_prediction = self.test_pred.compute()
        with open(
            os.path.join(
                self.test_path, f"prediction_test_gaze_epoch_{self.current_epoch}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(save_prediction, f)
        angular_results = compute_gaze_results(save_prediction,self.mode_angular )
        for k, v in angular_results.items():
            self.log(f"test_datasets/{k}", v, on_step=False, on_epoch=True, prog_bar=False)
        with open(
            os.path.join(
                self.test_path, f"angular_error_testdata_epoch_{self.current_epoch}.json"
            ),
            "w",
        ) as f:
            json.dump(angular_results, f, indent=4)
        self.test_pred.reset()
    
    def predict_step(self, batch, batch_idx):
        #_, preds, targets = self.model_step(batch)

        data_id = list(set(batch["data_id"].cpu().tolist()))
        assert len(data_id) == 1
        data_id = data_id[0]

        output = self(batch["images"], 1)
        preds = output["cartesian"] if "cartesian" in output.keys() else output["spherical"]
        targets = torch.zeros(preds.size(0), 3) if "cartesian" in output.keys() else torch.zeros(preds.size(0), 2)
        
        self.test_pred.update(preds, targets, batch["frame_id"], 
                                batch["clip_id"], batch["person_id"], batch["data_id"])

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        save_prediction = self.test_pred.compute()
        with open(
            os.path.join(
                self.pred_path, f"prediction_gaze_epoch_{self.current_epoch}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(save_prediction, f)
        save_pred_gaze_results(save_prediction,self.pred_path, self.mode_angular)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def get_param_groups(self):
        def _get_layer_decay(name):
            layer_id = None
            if name in ("encoder.class_token", "encoder.pose_token", "encoder.mask_token"):
                layer_id = 0
            elif ("_encoder" in name):
                layer_id = 0
            elif ("_head" in name):
                layer_id = self.hparams.transformer.depth + 1
            elif name.startswith("encoder.pos_embedding"):
                layer_id = 0
            elif name.startswith("encoder.transformer1.layers"):
                layer_id = int(name.split("encoder.transformer1.layers.")[1].split(".")[0]) + 1
            else:
                layer_id = self.hparams.transformer.depth + 1
            layer_decay = self.hparams.solver.layer_decay ** (self.hparams.transformer.depth + 1 - layer_id)
            return layer_id, layer_decay

        non_bn_parameters_count = 0
        zero_parameters_count = 0
        no_grad_parameters_count = 0
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, p in self.named_parameters():
            if not p.requires_grad:
                group_name = "no_grad"
                no_grad_parameters_count += 1
                continue
            name = name[len("module."):] if name.startswith("module.") else name
            if ((len(p.shape) == 1 or name.endswith(".bias")) and self.hparams.solver.ZERO_WD_1D_PARAM):
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "zero")
                weight_decay = 0.0
                zero_parameters_count += 1
            else:
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "non_bn")
                weight_decay = self.hparams.solver.weight_decay
                non_bn_parameters_count += 1

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.hparams.solver.lr * layer_decay,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.hparams.solver.lr * layer_decay,
                }
            parameter_group_names[group_name]["params"].append(name)
            parameter_group_vars[group_name]["params"].append(p)

        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params
   
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        # linear learning rate scaling for multi-gpu
        if(self.trainer.num_devices * self.trainer.num_nodes>1 and self.hparams.solver.apply_linear_scaling):
            self.lr_scaler = self.trainer.num_devices * self.trainer.num_nodes * self.trainer.accumulate_grad_batches * self.hparams.train_batch_size / 256
        else:
            self.lr_scaler = 1

        optim_params = [{'params': filter(lambda p: p.requires_grad, self.trainer.model.parameters()), 'lr': self.hparams.solver.lr * self.lr_scaler}]

        if(self.hparams.solver.name=="AdamW"):
            optimizer = optim.AdamW(params=optim_params, weight_decay=self.hparams.solver.weight_decay, betas=(0.9, 0.95))
        elif(self.hparams.solver.name=="Adam"):
            optimizer = optim.Adam(params=optim_params, weight_decay=self.hparams.solver.weight_decay, betas=(0.9, 0.95))
        elif(self.hparams.solver.name=="SGD"):
            optimizer = optim.SGD(params=optim_params, momentum=self.hparams.solver.momentum, weight_decay=self.hparams.solver.weight_decay)
        else:
            raise NotImplementedError("Unknown solver : " + self.hparams.solver.name)
        
        def warm_start_and_cosine_annealing(epoch):
            lr_scale_min = 0.01
            if epoch < self.hparams.solver.warmup_epochs:
                lr = (epoch+1) / self.hparams.solver.warmup_epochs
            else:
                lr = lr_scale_min + 0.5 *(1-lr_scale_min)*(1. + math.cos(math.pi * ((epoch+1) - self.hparams.solver.warmup_epochs) \
                                                                         / (self.trainer.max_epochs - self.hparams.solver.warmup_epochs )))
            return lr

        if(self.hparams.solver.scheduler == "cosine"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_start_and_cosine_annealing for _ in range(len(optim_params))], verbose=False)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.solver.decay_steps, gamma=self.hparams.solver.decay_gamma, verbose=False)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval" : "epoch",
                'frequency': 1,
            }
        }


