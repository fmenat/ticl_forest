import torch
from torchmetrics import Metric

class OA(Metric):
    """custom overall accuracy metric implementation following torchmetric framework"""
    def __init__(self, no_data_val, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.long), dist_reduce_fx="sum")
        self.no_data_val = no_data_val

    def update(self, preds, target: torch.Tensor, valid_mask: torch.Tensor):
        assert preds.shape == target.shape == valid_mask.shape
        upd_valid_mask = valid_mask & (target != self.no_data_val) 

        self.correct += (((preds == target)*upd_valid_mask).float().sum(axis=[1,2]) / (upd_valid_mask.sum(axis=[1,2]).float() + 1e-8) ).sum()
        self.total += target.shape[0]

    def compute(self):
        return self.correct / self.total


class ConfusionStatsBatch(Metric):
    """    Base metric that accumulates TP, FP, FN calculated over images. """
    def __init__(self, num_classes: int, no_data_val: int, metrics_calculate=[]):
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.no_data_val = no_data_val
        self.metrics_calculate = metrics_calculate
        if len(self.metrics_calculate) == 0:
            self.metrics_calculate=["pre", "rec", "f1"]

        self.add_state("pre", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("f1", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("f2", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("iou", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("ior", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("ti", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("yj", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor): #require categorical predictions
        assert preds.shape == target.shape == valid_mask.shape

        upd_valid_mask = valid_mask & (target != self.no_data_val) 
        for c in range(self.num_classes):
            pred_c = preds == c
            true_c = target == c
            tp_c = torch.sum(pred_c & true_c & upd_valid_mask, axis=[1,2])
            fp_c = torch.sum(pred_c & ~true_c & upd_valid_mask, axis=[1,2])
            fn_c = torch.sum(~pred_c & true_c & upd_valid_mask, axis=[1,2])
            
            if "pre" in self.metrics_calculate or "f1" in self.metrics_calculate:
                pre_c = tp_c / (tp_c + fp_c + 1e-8)
                self.pre[c] += pre_c.sum()
            if "rec" in self.metrics_calculate or "f1" in self.metrics_calculate:
                rec_c = tp_c / (tp_c + fn_c + 1e-8)
                self.rec[c] += rec_c.sum()
            if "f1" in self.metrics_calculate:
                self.f1[c] += 2*(pre_c * rec_c / (pre_c + rec_c + 1e-8)).sum()
            if "iou" in self.metrics_calculate:
                self.iou[c] += (tp_c / (tp_c + fp_c + fn_c + 1e-8)).sum()
            if "ior" in self.metrics_calculate:
                self.ior[c] += (tp_c / (tp_c + fn_c + 1e-8)).sum()
            if "f2" in self.metrics_calculate:
                self.f2[c] += (5*pre_c * rec_c / (4*pre_c + rec_c + 1e-8)).sum()
            if "ti" in self.metrics_calculate:
                self.ti[c] += (tp_c / (tp_c + 0.7*fp_c + 0.3*fn_c + 1e-8)).sum()
        self.total += target.shape[0]
        if "yj" in self.metrics_calculate:
            self.yj += self.rec.sum() -1

    def compute(self):
        return_stats = {}
        for m in self.metrics_calculate:
            stats_m = eval(f"self.{m}") / self.total
            return_stats[f"ma_{m}"] = stats_m.mean()

            for c in range(self.num_classes): 
                if m == "yj":
                    return_stats[f"{m}"] = stats_m
                else:   
                    return_stats[f"{c}_{m}"] = stats_m[c]
        return return_stats


class ConfusionStatsOverall(Metric):
    """    Base metric that accumulates TP, FP, FN calculated at pixel-level along all batches. """
    def __init__(self, num_classes: int, no_data_val: int, metrics_calculate=[], prefix=""):
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.no_data_val = no_data_val
        self.metrics_calculate = metrics_calculate
        self.prefix = prefix
        if len(self.metrics_calculate) == 0:
            self.metrics_calculate=["pre", "rec", "f1"]

        self.add_state("tp", default=torch.zeros(self.num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(self.num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(self.num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("true_pixels", default=torch.zeros(self.num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("pred_pixels", default=torch.zeros(self.num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("valid_pixels", default=torch.tensor(0.0, dtype=torch.long), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor): #require categorical predictions
        assert preds.shape == target.shape == valid_mask.shape
        
        upd_valid_mask = valid_mask & (target != self.no_data_val) 
        for c in range(self.num_classes):
            pred_c = preds == c
            true_c = target == c
            self.tp[c] += torch.sum(pred_c & true_c & upd_valid_mask)
            self.fp[c] += torch.sum(pred_c & ~true_c & upd_valid_mask)
            self.fn[c] += torch.sum(~pred_c & true_c & upd_valid_mask)
            self.true_pixels[c] += torch.sum(true_c & upd_valid_mask)
            if "counthat" in self.metrics_calculate:
                self.pred_pixels[c] += torch.sum(pred_c & upd_valid_mask)
        self.valid_pixels += torch.sum(upd_valid_mask)

    def compute(self):
        return_stats = {}
        nan_classess = []
        for c in range(self.num_classes):
            if "count" in self.metrics_calculate:
                return_stats[f"{c}_count"] = self.true_pixels[c]/self.valid_pixels
            if "counthat" in self.metrics_calculate:
                return_stats[f"{c}_counthat"] = self.pred_pixels[c]/self.valid_pixels

            if self.true_pixels[c].item() == 0:
                nan_classess.append(c)
                return_stats[f"{c}_pre"] = torch.tensor(float('nan'), device=self.tp.device)
                return_stats[f"{c}_rec"] = torch.tensor(float('nan'), device=self.tp.device)
                return_stats[f"{c}_f1"] = torch.tensor(float('nan'), device=self.tp.device)
                return_stats[f"{c}_iou"] = torch.tensor(float('nan'), device=self.tp.device)
                return_stats[f"{c}_ior"] = torch.tensor(float('nan'), device=self.tp.device)
                return_stats[f"{c}_f2"] = torch.tensor(float('nan'), device=self.tp.device)
                return_stats[f"{c}_ti"] = torch.tensor(float('nan'), device=self.tp.device)
                continue

            if "pre" in self.metrics_calculate or "f1" in self.metrics_calculate:
                prec_c = self.tp[c] / (self.tp[c] + self.fp[c] + 1e-8)
                return_stats[f"{c}_pre"] = prec_c
            if "rec" in self.metrics_calculate or "f1" in self.metrics_calculate:
                rec_c = self.tp[c] / (self.tp[c] + self.fn[c] + 1e-8)
                return_stats[f"{c}_rec"] = rec_c
            if "f1" in self.metrics_calculate:
                return_stats[f"{c}_f1"] = 2*(prec_c * rec_c / (prec_c + rec_c + 1e-8))
            if "iou" in self.metrics_calculate:
                return_stats[f"{c}_iou"] = (self.tp[c] / (self.tp[c] + self.fp[c] + self.fn[c] + 1e-8))
            if "ior" in self.metrics_calculate:
                return_stats[f"{c}_ior"] = (self.tp[c] / (self.tp[c] + self.fn[c] + 1e-8))
            if "f2" in self.metrics_calculate:
                return_stats[f"{c}_f2"] = (5*prec_c * rec_c / (4*prec_c + rec_c + 1e-8))
            if "ti" in self.metrics_calculate:
                return_stats[f"{c}_ti"] = (self.tp[c] / (self.tp[c] + 0.7*self.fp[c] + 0.3*self.fn[c] + 1e-8))
            
        for m in self.metrics_calculate:
            if m not in ["yj", "count", "counthat"] and f"{0}_{m}" in return_stats:
                return_stats[f"ma_{m}"] = torch.nanmean(torch.stack([return_stats[f"{c}_{m}"] for c in range(self.num_classes)]))
        return_stats["yj"] = torch.nan_to_num(return_stats["0_rec"],0) + torch.nan_to_num(return_stats["1_rec"],0) -1

        return {self.prefix+k: v for k, v in return_stats.items()}


class BinaryStatsOverallWindow(ConfusionStatsOverall):
    """    Base metric that accumulates TP, FP, FN calculated at pixel-level along all batches. """
    def __init__(self, no_data_val: int, metrics_calculate=[], prefix=""):
        super().__init__(2, no_data_val, metrics_calculate, prefix)
        
    def update(self, preds: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, pos_in_window: torch.Tensor): #require categorical predictions
        assert preds.shape == target.shape == valid_mask.shape == pos_in_window.shape   
        
        upd_valid_mask = valid_mask & (target != self.no_data_val) 
        for c in range(self.num_classes):
            pred_c = preds == c
            true_c = target == c
            
            if c == 1:
                self.tp[c] += torch.sum(pred_c & pos_in_window & upd_valid_mask) #true positive if model predicts disturbance in any where disturbance-window
            else:
                self.tp[c] += torch.sum(pred_c & true_c & upd_valid_mask)

            self.fp[c] += torch.sum(pred_c & ~true_c & upd_valid_mask)
            self.fn[c] += torch.sum(~pred_c & true_c & upd_valid_mask)
            self.true_pixels[c] += torch.sum(true_c & upd_valid_mask)
            if "counthat" in self.metrics_calculate:
                self.pred_pixels[c] += torch.sum(pred_c & upd_valid_mask)
        self.valid_pixels += torch.sum(upd_valid_mask)

    

if __name__ == "__main__":
    mmm = ConfusionStatsBatch(num_classes=2, no_data_val= -9999)
    pred = torch.randn(2, 32, 32)
    target = torch.randn(2, 32, 32)
    mmm(pred, target, torch.ones_like(target).bool())
    res = mmm.compute()
    print(res)

    mmm = ConfusionStatsOverall(num_classes=2, no_data_val= -9999)
    pred = torch.randn(2, 32, 32)
    target = torch.randn(2, 32, 32)
    mmm(pred, target, torch.ones_like(target).bool())
    res = mmm.compute()
    print(res)

    mmm = OA(no_data_val= -9999)
    pred = torch.randn(2, 32, 32)
    target = torch.randn(2, 32, 32)
    mmm(pred,  target, torch.ones_like(target).bool())
    res = mmm.compute()
    print(res)

    mmm = BinaryStatsOverallWindow(no_data_val= -9999)
    pred = torch.randn(2, 32, 32)
    target = torch.randint(2, (5, 2, 32, 32))
    pos_in_window = pos_in_window_tensor(target, window_size=3, dim=0)
    mmm(pred,  target[0], torch.ones_like(pred).bool(), pos_in_window[0])
    res = mmm.compute()
    print(res)
