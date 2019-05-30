import torch
import torch.nn as nn
import datetime
import time
import os
from .logger import Logger


def logging(train_logger, result, step):
    for tag, value in result.items():
        train_logger.scalar_summary(tag, value, step)


class TrainerBase:
    def __init__(self, trainer_config, model_config):
        # Setup directory for checkpoint saving
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(model_config['checkpoint'],
                                           self.start_time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Setup device
        self.device, self.device_ids = self._prepare_device(trainer_config['n_gpu'])
        self.train_logger = Logger(trainer_config['logs_train'])
        self.valid_logger = Logger(trainer_config['logs_valid'])

        self.trainer_config = trainer_config
        self.model_config = model_config
        self.epochs = trainer_config['epochs']
        self.save_freq = trainer_config['save_freq']
        self.star_epoch = 1

    def _config_model_and_optimizer(self, model, optimizer):
        self.model = model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer

    def train(self):
        for epoch in range(self.star_epoch, self.epochs + self.star_epoch):
            print("\n\n-------------------------------------------------------------------------------------------")
            start_time = time.time()
            result_train = self._train(epoch)
            end_time = time.time()
            print("Epoch train in {:.2f}s\n\n".format(end_time - start_time))
            result_eval = self._eval(epoch)
            result = {**result_train, **result_eval}

            # Log metrics
            if (self.train_logger is not None) or \
                    (self.valid_logger is not None):
                for key, value in result.items():
                    if key == "train_metrics":
                        logging(self.train_logger, value, epoch)
                    elif key == "valid_metrics":
                        logging(self.valid_logger, value, epoch)

            # Save checkpoints
            self._save_checkpoint(epoch, save_best=True)

    @staticmethod
    def _prepare_device(n_gpu_use):
        """
        Setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, "
                  "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, "
                  "but only {} are available on this machine."
                  .format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _train(self, epoch_index):
        raise NotImplementedError

    def _eval(self, epoch_index):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "trainer_config": self.trainer_config,
            "model_config": self.model_config,
        }
        # Save checkpoint
        if self.save_freq is not None:
            if epoch % self.save_freq == 0:
                file_name = os.path.join(self.checkpoint_dir,
                                         'epoch{}.pth'.format(epoch))
                torch.save(state, file_name)
        elif save_best:
            # Save the best checkpoints
            best_file = os.path.join(self.checkpoint_dir,
                                     'best_model.pth')
            torch.save(state, best_file)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        """
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # Load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.model_config['arch']:
            print("Warning: Architecture configuration given in config file "
                  "is different from that of checkpoint. This may yield an "
                  "exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("Checkpoint loaded. Resume training from epoch {}"
              .format(self.start_epoch))
