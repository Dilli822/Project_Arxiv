import comet_ml
import os 
import argparse
import torch
import pytorch_lightning as pl 

from model import SkcMobileNet
from dataset import SkinCancerDataModule

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from pytorch_lightning.loggers import CometLogger

# Load API
from dotenv import load_dotenv
load_dotenv()


def main(args):
    comet_logger = CometLogger(api_key=os.getenv('API_KEY'), 
                               project=os.getenv('PROJECT_NAME'))
    
    dataloader = SkinCancerDataModule(train_dir=args.train_dir,
                                      val_dir=args.val_dir,
                                      batch_size=args.batch_size, 
                                      num_workers=args.data_workers)
    
    # Call setup to initialize datasets
    dataloader.setup('fit')  
    num_classes = dataloader.get_num_classes()

    # Initialize the model
    model = SkcMobileNet(checkpoint_path=args.checkpoint_path,
                      num_classes=num_classes,
                      gpu_nodes=args.gpus).to(args.device)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./saved_checkpoint/",       
        filename='model-{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}',                                             
        save_top_k=5,
        mode='min'
    )

    # Trainer Parameters
    trainer_args = {
        'accelerator': args.device,                                     # Device to use for training
        'devices': args.gpus,                                           # Number of GPUs to use for training
        'max_epochs': args.epochs,                                      # Maxm. no. of epochs to run                               
        'precision': args.precision,                                    # Precision to use for training
        'check_val_every_n_epoch': 1,                                   # No. of epochs to run validation
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),    # Callbacks to use for training
                      EarlyStopping(monitor="val_loss", patience=5),
                      checkpoint_callback],
        'logger': comet_logger,                                         # Logger to use for training
    }

    if args.gpus > 1:
        trainer_args['strategy'] = args.dist_backend
    trainer = pl.Trainer(**trainer_args)

    # Create a Trainer instance for managing the training process.
    trainer = pl.Trainer(**trainer_args)

    # Fit the model to the training data using the Trainer's fit method.
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Train")

    # Train Device Hyperparameters
    parser.add_argument('-d', '--device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'],
                        help='device to use for training, default cuda')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=0, type=int,
                        help='n data loading workers, default 0 = main process only')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='which distributed backend to use for aggregating multi-gpu train')

    # Train and Test Directory Params
    parser.add_argument('--train_dir', default=None, required=True, type=str,
                        help='Folder path to load training data')
    parser.add_argument('--val_dir', default=None, required=True, type=str,
                        help='Folder path to load validation data')

    parser.add_argument('--checkpoint_path', default=None, required=True, type=str,
                        help='Path to the model checkpoint to load')
    
    
    # General Train Hyperparameters
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('--precision', default='32-true', type=str, help='precision')
    
    args = parser.parse_args()
    main(args)