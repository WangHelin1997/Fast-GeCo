import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from fastgeco.backbones.shared import BackboneRegistry
from geco.data_module import SpecsDataModule
from geco.sdes import SDERegistry
from fastgeco.model import ScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     parser = ArgumentParser()
     parser.add_argument("--batch_size", type=int, default=32,  help="During training take at least N_min reverse steps")
     parser.add_argument("--N_min", type=int, default=1,  help="During training take at least N_min reverse steps")
     parser.add_argument("--N_max", type=int, default=1,  help="During training take at most N_max reverse steps")
     parser.add_argument("--t_rsp_min", type=float, default=0.5,  help="During training take at least N_min reverse steps")
     parser.add_argument("--t_rsp_max", type=float, default=0.5,  help="During training take at most N_max reverse steps")
     parser.add_argument("--pre_ckpt", type=str, default='./logs/u0kwl5bj/epoch=13-si_sdr=8.10.ckpt',  help="Load ckpt")
     parser.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
     parser.add_argument("--lr", type=float, default=5e-5, help="The learning rate (1e-4 by default)")
     parser.add_argument("--loss_type", type=str, default="default", help="The type of loss function to use.")
     parser.add_argument("--stop_iteration_random", type=str, choices=['random', 'last', 'epoch', 'residual_mag'], default="last", help="0 means it is fix, 1 means random stop iterations")
     parser.add_argument("--inference_N", type=int, default=1, help="inference N")
     parser.add_argument("--inference_start", type=float, default=0.5, help="inference start")


     args = parser.parse_args()
     checkpoint_file = args.pre_ckpt

    # Load score model
     model = ScoreModel.load_from_checkpoint(
        checkpoint_file,
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
     model.add_para(args.N_min, args.N_max, args.t_rsp_min, args.t_rsp_max, 
                    args.batch_size, args.loss_type, args.lr, args.stop_iteration_random,
                    args.inference_N, args.inference_start)
     model.cuda()
     model.to('cuda:0')
     
     

     if not args.nolog:
          #this needs to be changed accordingly to your wandb settings
          wandb.login(key='YOUR KEY')
          logger = WandbLogger(project="fastgeco", log_model=True, save_dir="logs")
          logger.experiment.log_code(".")
          savedir_ck = f'./logs/{logger.version}' #change your folder, where to save files
          if not os.path.isdir(savedir_ck):
               os.makedirs(os.path.join(savedir_ck))
     else:
          logger = None

     # Set up callbacks for logger
     if logger != None:
          callbacks = [ModelCheckpoint(dirpath=savedir_ck, save_last=True, filename='{epoch}-last')]
          checkpoint_callback_last = ModelCheckpoint(dirpath=savedir_ck,
               save_last=True, filename='{epoch}-last')
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=savedir_ck,
               save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
          checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=savedir_ck,
               save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
          callbacks = [checkpoint_callback_last, checkpoint_callback_pesq,
               checkpoint_callback_si_sdr]
     # Initialize the Trainer and the DataModule
     if logger != None:
          trainer = pl.Trainer(strategy=DDPPlugin(find_unused_parameters=False), logger=logger,
               log_every_n_steps=10, num_sanity_val_steps=1, accelerator="gpu", devices="auto",
               callbacks=callbacks
          )
     else:
          trainer = pl.Trainer(strategy=DDPPlugin(find_unused_parameters=False),
          log_every_n_steps=10, num_sanity_val_steps=1, accelerator="gpu", devices="auto"
     )


     # Train model
     trainer.fit(model)
