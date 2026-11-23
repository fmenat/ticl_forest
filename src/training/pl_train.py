
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import torch

def train_model_year(dl_model, train_dataloader, val_dataloader, full_exp_name_year, STORAGE_PATH, DEVICE, 
                     metric_stopval, mode_stopval, delta_stopval, pat_stopval, EPOCHS, prefix="", store_checkpoints=True, early_stopping=True, only_return_trainer=False):
    timer = Timer()
    logger = TensorBoardLogger(f"{STORAGE_PATH}/", name=  f"{prefix}-{full_exp_name_year}" if prefix != "" else full_exp_name_year)
    if store_checkpoints:
        checkpoint_callback = ModelCheckpoint(monitor=f"{metric_stopval}/val", mode=mode_stopval, every_n_epochs=1, save_top_k=1 ,dirpath=logger.log_dir, filename='model_{epoch}') 
    if early_stopping:
        early_stop_callback = EarlyStopping(monitor=f"{metric_stopval}/val", mode=mode_stopval, min_delta=delta_stopval, patience=pat_stopval)
    
    
    callbacks = [timer]
    if early_stopping:
        callbacks.append(early_stop_callback)
    if store_checkpoints:
        callbacks.append(checkpoint_callback)
    enable_progress_bar = store_checkpoints  # show progress bar if we have callbacks that might stop early or save checkpoints
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="cuda", devices=DEVICE, callbacks=callbacks, logger=logger, enable_progress_bar=enable_progress_bar, enable_checkpointing=store_checkpoints)
    if only_return_trainer:
        return trainer
    trainer.fit(dl_model, train_dataloader, val_dataloaders=val_dataloader)

    n_epochs_trained = trainer.current_epoch
    if store_checkpoints:
        print("Model best weights are at", checkpoint_callback.best_model_path)
    print(f"Training took {timer.time_elapsed('train')} s, with time per epoch {timer.time_elapsed('train')/n_epochs_trained} s and finished in epoch {n_epochs_trained}.")

    return trainer, timer.time_elapsed("train")/n_epochs_trained



def get_val_f1_from_tb(trainer: pl.Trainer, which: str, tag: str = "1_f1/val") -> float:
    """
    Reads the TensorBoard event file written by the trainer and returns the
    most recent value of the scalar tag `val_f1`.
    """
    # 1️⃣ Find the TB log directory
    tb_logger = trainer.logger  # -> TensorBoardLogger instance (if you used it)
    tb_dir = Path(tb_logger.log_dir)  # e.g. ".../lightning_logs/version_0"

    # 2️⃣ Find the *newest* events file (there may be several)
    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {tb_dir}")

    # TensorBoard keeps writing to the *most recent* file; use the last one
    latest_event_file = str(event_files[-1])

    # 3️⃣ Load the file with EventAccumulator – it lazily parses the binary
    ea = event_accumulator.EventAccumulator(latest_event_file,
                                            size_guidance={  # optional: speed up loading
                                                event_accumulator.SCALARS: 0,  # load all scalars
                                            })
    ea.Reload()                               # actually parses the file

    # 4️⃣ Pull the scalar series for the tag we care about
    if tag not in ea.Tags()["scalars"]:
        raise RuntimeError(f"Scalar tag `{tag}` not found in TensorBoard logs. with {ea.Tags()['scalars']} "
                           f"Did you log it with `self.log('{tag}', ...)`?")

    scalar_events = ea.Scalars(tag)      # list of Event objects, ordered by step
    if which == "last":
        sel_event = scalar_events[-1]
    elif which == "best":
        sel_event = max(scalar_events, key=lambda e: e.value)

    return float(sel_event.value)            # plain Python float