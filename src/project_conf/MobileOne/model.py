import pytorch_lightning as pl 
import torch
import torchmetrics


from torch import nn 
from mobileone import mobileone


class SkcMobileNet(pl.LightningModule):
    def __init__(self, checkpoint_path, num_classes, gpu_nodes):
        super(SkcMobileNet, self).__init__()
        self.model = mobileone(variant='s4', inference_mode=False)
        self.checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')

        state_dict = self.checkpoint['state_dict'] if 'state_dict' in self.checkpoint else self.checkpoint
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('linear.')}
        
        # Load weights
        self.model.load_state_dict(state_dict, strict=False)

        # Modify the classifier layer
        num_ftrs = self.model.linear.in_features  
        self.model.linear = nn.Sequential(
                        nn.Linear(in_features=num_ftrs, out_features=512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(in_features=512, out_features=num_classes))
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Setup Metrics
        self.accuracy = torchmetrics.Accuracy(task="binary") 
        self.f1_score = torchmetrics.F1Score(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        
        self.save_hyperparameters(ignore=["model"])
        
        self.sync_dist = True if gpu_nodes > 1 else False

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=2,
                threshold=3e-2,
                threshold_mode='rel',
                min_lr=1e-5
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1            # Apply scheduler after every epoch
        }
        
        return [optimizer], [scheduler]

    def _common_step(self, batch, batch_idx, compute_extra_metrics=False):
        X, y = batch
        y = y.float()
        outputs = self.forward(X).squeeze()
        loss = self.loss_fn(outputs, y)
    
        y_pred = torch.round(torch.sigmoid(outputs))
        accuracy = self.accuracy(y_pred, y.int())
    
        if compute_extra_metrics:
            precision = self.precision(y_pred, y.int())
            recall = self.recall(y_pred, y.int())
            f1_score = self.f1_score(y_pred, y.int())
        else:
            precision = recall = f1_score = None
    
        return loss, accuracy, precision, recall, f1_score

    def training_step(self, batch, batch_idx):
        loss, accuracy, _, _, _ = self._common_step(batch, batch_idx, compute_extra_metrics=False)
        self._log_metrics(prefix="", loss=loss, accuracy=accuracy, precision=None, recall=None, f1_score=None, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, precision, recall, f1_score = self._common_step(batch, batch_idx, compute_extra_metrics=True)
        self._log_metrics(prefix="val_", loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score, on_step=False, on_epoch=True)
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        loss, accuracy, precision, recall, f1_score = self._common_step(batch, batch_idx, compute_extra_metrics=True)
        self._log_metrics(prefix="test_", loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score, on_step=False, on_epoch=True)
        return {'test_loss': loss}

    def _log_metrics(self, prefix: str, loss, accuracy, precision, recall, f1_score, on_step: bool, on_epoch: bool):
        metrics = {f"{prefix}loss": loss, f"{prefix}accuracy": accuracy}
        if precision is not None: metrics[f"{prefix}precision"] = precision
        if recall is not None: metrics[f"{prefix}recall"] = recall
        if f1_score is not None: metrics[f"{prefix}f1_score"] = f1_score
    
        self.log_dict(metrics,
                      on_step=on_step,
                      on_epoch=on_epoch,
                      prog_bar=True,
                      logger=True,
                      sync_dist=self.sync_dist)