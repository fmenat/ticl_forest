import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Precision, Recall, F1Score

from src.evaluation.metrics import OA, ConfusionStatsOverall
from .lossess import SpatialMaskedCrossEntropyLoss, MaskedDiceLoss, MaskedFocalLoss

class PLBase(pl.LightningModule):
    def __init__(self, n_classes, no_data_value=255, label_name="", lr=1e-3, weight_decay=1e-5, optimizer="adam", extra_optimizer_kwargs={}, BS=32):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer.lower()
        self.extra_optimizer_kwargs = extra_optimizer_kwargs
        self.BS = BS
        self.n_classes = n_classes
        self.no_data_value = no_data_value
        self.label_name = label_name   

    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        elif self.hparams.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS
        return optimizer(self.parameters(),lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, **self.hparams.extra_optimizer_kwargs)
  
    def on_train_epoch_end(self):
        dic_metrics = self.train_metrics.compute()
        for k, v in dic_metrics.items():
            self.log(f"{k}", v, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=self.BS)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        dic_metrics = self.val_metrics.compute()
        for k, v in dic_metrics.items():
            self.log(f"{k}", v, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=self.BS)
        self.val_metrics.reset()

    def apply_softmax(self, y: torch.Tensor, dim=-1) -> torch.Tensor:
        return torch.nn.Softmax(dim=dim)(y)
    
    def apply_sigmoid(self, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.Sigmoid()(y)


class ForestMaskSegmentationBase(PLBase):
    def __init__(self, n_classes, no_data_value=255, label_name="", lr=1e-3, weight_decay=1e-5, optimizer="adam", extra_optimizer_kwargs={}, BS=32, loss_args={}, macro_loss_args={}):
        super().__init__(n_classes=n_classes, no_data_value=no_data_value, label_name=label_name, lr=lr, weight_decay=weight_decay, optimizer=optimizer, extra_optimizer_kwargs=extra_optimizer_kwargs, BS=BS)
        self.save_hyperparameters()
        self.train_metrics = MetricCollection([ 
            OA(no_data_val=self.no_data_value), 
            ]).clone(postfix="/train")
        self.val_metrics = MetricCollection([ 
            OA(no_data_val=self.no_data_value), 
            ConfusionStatsOverall(num_classes=self.n_classes, no_data_val=self.no_data_value, metrics_calculate=["f1", "rec"])  #ConfusionStatsBatch
            ]).clone(postfix="/val")
        
        self.set_loss_args(loss_args, macro_loss_args)
        self.pad_h = self.pad_w = 0 #initialize padding attributes to avoid attribute error in prediction_trimming if not set during training/validation

    def set_loss_args(self, loss_args: dict, macro_loss_args: dict):
        self.loss_args = loss_args.copy()
        if "weight" in loss_args:
            self.loss_args["weight"] = torch.tensor(self.loss_args["weight"], dtype=torch.float32)

        self.macro_loss_args = macro_loss_args.copy()
        if "weight" not in macro_loss_args:
            self.macro_loss_args["weight"] = {}

        self.criteria = SpatialMaskedCrossEntropyLoss(self.no_data_value, **self.loss_args)
        if self.macro_loss_args["weight"].get("dice", 0) > 0:
            self.dice_criteria = MaskedDiceLoss(no_data_value=self.no_data_value, weight=self.loss_args.get("weight", None), target_class=self.macro_loss_args.get("dice_target_class", -1))
        if self.macro_loss_args["weight"].get("focal", 0) > 0:
            self.focal_criteria = MaskedFocalLoss(no_data_value=self.no_data_value, gamma=self.macro_loss_args.get("focal_gamma", 2.0), target_class=self.macro_loss_args.get("focal_target_class", -1))
        

    def training_step(self, batch: dict, batch_idx):
        loss = self.loss_batch(batch)
        for k, v in loss.items():
            if k not in ["pred", "valid_mask", "pred_logits"]:
                self.log(k+"/train", v, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=self.BS)
        
        if len(batch["label"][self.label_name].shape) > len(loss["pred"].shape):
            labels = batch["label"][self.label_name].argmax(dim=1)
        else:
            labels = batch["label"][self.label_name]
        self.train_metrics.update(loss["pred"], labels, loss["valid_mask"])
        return loss["loss"]

    def validation_step(self, batch: dict, batch_idx):
        loss = self.loss_batch(batch)
        if len(batch["label"][self.label_name].shape) > len(loss["pred"].shape):
            labels = batch["label"][self.label_name].argmax(dim=1)
        else:
            labels = batch["label"][self.label_name]
        self.val_metrics.update(loss["pred"], labels, loss["valid_mask"])
        return -1 
    
    def prepare_batch(self, batch: dict):
        label_dic = {k: v.to(self.device) for k, v in batch["label"].items()}
        
        batch_dict = {}
        for k, v in batch["input"].items():
            if k not in self.encoder:
                continue
            batch_dict[k] = v.to(self.device)
            if batch_dict[k].ndim == 5:  #spatio-temporal data
                batch_dict[k] = batch_dict[k].permute(0, 1, 4, 2, 3)  # Ensure shape is [B, T, C, H, W]

            if batch_dict[k].ndim > 3:  #spatial data
                h, w = batch_dict[k].shape[-2:]
                final_pooling = self.encoder[k].final_pooling if hasattr(self.encoder[k], 'final_pooling') else 1
                N = (2**(self.encoder[k].n_layers-1) if self.encoder[k].n_layers>1 else 0) + (final_pooling if final_pooling > 1 else 0)
                
                if N > 0 and (h % N != 0 or w % N != 0): #pad to be divisible by N
                    self.pad_h = (N - h % N) % N
                    self.pad_w = (N - w % N) % N
                    batch_dict[k] = torch.nn.functional.pad(batch_dict[k], (0, self.pad_w, 0, self.pad_h), mode='constant', value=0)
                else:
                    self.pad_h = self.pad_w = 0

        return batch_dict, label_dic

    def prediction_trimming(self, pred: torch.Tensor):
        if self.pad_h > 0 or self.pad_w > 0:
            h_ref, w_ref = pred.shape[-2]-self.pad_h, pred.shape[-1]-self.pad_w
            pred = pred[..., :h_ref, :w_ref]
        return pred
    
    def mask_calculation(self, reference_labels: dict) -> dict:
        masks = {
                "event": reference_labels[self.label_name] == 1, #event can be 255
                "recovering": reference_labels[f"sincelast_disturbances"] != 0, #from start to year-1
               }
        if "forest_mask_static" in reference_labels:
            masks["forest"] = reference_labels["forest_mask_static"].bool()
        else:
            masks["forest"] = torch.ones_like(masks["event"]).bool() #if no forest mask consider all areas as forest (but still can be masked out by other masks)
        if "future_disturbance" in reference_labels:
            masks["future_event"] = reference_labels["future_disturbance"].bool() #if future disturbance mean the forest is already recovered in the current step
        if "past_disturbance" in reference_labels:
            masks["past_event"] = reference_labels["past_disturbance"].bool()


        if self.training:
            training_masking = self.macro_loss_args.get("training_masking", "forest")
            epochs_stable = self.macro_loss_args.get("epochs_stable", 0)
            if epochs_stable > 0 and self.current_epoch <= epochs_stable:
                training_masking = "stable"  # start training with stable areas only

            if training_masking =="all":
                masks["valid_mask"] = torch.ones_like(masks["event"]).bool()  #consider all areas during training
            elif training_masking =="forest":
                masks["valid_mask"] = masks["forest"]
            elif training_masking =="stable":
                masks["valid_mask"] = (masks["forest"] & ~masks["recovering"]) | (masks["past_event"] | masks["event"] | masks["future_event"])
            
        else: #teacher forcing that recovering areas are not valid (is input but still can be predicted as disturbance)
            validation_masking = self.macro_loss_args.get("validation_masking", "stable")
            if validation_masking =="all":
                masks["valid_mask"] = torch.ones_like(masks["event"]).bool()  
            elif validation_masking =="forest":
                masks["valid_mask"] = masks["forest"]
            elif validation_masking =="stable":
                masks["valid_mask"] = (masks["forest"] & ~masks["recovering"]) | (masks["past_event"] | masks["event"] | masks["future_event"])
            
        return masks
        
    
    def loss_batch(self, batch: dict, val_loss=False) -> dict: 
        input_sensor, reference_labels = self.prepare_batch(batch)
        masks = self.mask_calculation(reference_labels)
        
        disturb_pred = self(input_sensor) #model prediction
        return_dic = {
             "pred": disturb_pred["logits"].argmax(dim=1).detach(),
             "valid_mask": masks["valid_mask"].detach(),
             "pred_logits": disturb_pred["logits"].detach(),
             }
        if not self.training and not val_loss:
            return return_dic

        loss_dic = {}
        if self.macro_loss_args["weight"].get("ce", 1) > 0:
            loss_dic["ce"] = self.criteria(disturb_pred["logits"], reference_labels[self.label_name], masks["valid_mask"])        

        #regular losses: dice, focal
        if self.macro_loss_args["weight"].get("dice", 0) > 0:
            loss_dic["dice"] = self.dice_criteria(disturb_pred["logits"], reference_labels[self.label_name], masks["valid_mask"])
        if self.macro_loss_args["weight"].get("focal", 0) > 0:
            loss_dic["focal"] = self.focal_criteria(disturb_pred["logits"], reference_labels[self.label_name], masks["valid_mask"])
        
        #temporal consistency losses
        if self.macro_loss_args["weight"].get("temp_cons_full", 0) > 0 or self.macro_loss_args["weight"].get("temp_cons_full_exp", 0) > 0:
            prev_disturb_mask = reference_labels["sincelast_disturbances"] != 0 
            if "last_maximum" in self.macro_loss_args:
                prev_disturb_mask = prev_disturb_mask & (reference_labels["sincelast_disturbances"] <= self.macro_loss_args["last_maximum"])                
            alpha = self.macro_loss_args.get("temp_cons_full_alpha", 1)
            
            if self.macro_loss_args["weight"].get("temp_cons_full", 0) > 0:
                loss_n = "temp_cons_full"
                spatial_weights = 1/(reference_labels["sincelast_disturbances"]**alpha +1e-8)
            elif self.macro_loss_args["weight"].get("temp_cons_full_exp", 0) > 0:
                loss_n = "temp_cons_full_exp"
                spatial_weights = torch.exp( (1-reference_labels["sincelast_disturbances"])/alpha )
            loss_dic[loss_n] = self.criteria(disturb_pred["logits"], reference_labels[self.label_name], masks["valid_mask"] & prev_disturb_mask, 
                                                       spatial_weights=spatial_weights) #focus on temporal consistency with since last disturbance weighting
        
        final_loss = loss_dic.get("ce", 0.0)
        for v in self.macro_loss_args["weight"]:
            if type(self.macro_loss_args["weight"][v]) != str and self.macro_loss_args["weight"][v] > 0 and v != "ce":
                final_loss += self.macro_loss_args["weight"][v] * loss_dic[v]
        
        return_dic["loss"] = final_loss
        return dict(**return_dic, **({k: v for k, v in loss_dic.items()} if len(self.macro_loss_args["weight"])>0 else {}) )
  