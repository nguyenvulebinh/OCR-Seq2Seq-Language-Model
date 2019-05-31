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
        self.checkpoint_dir = os.path.join(model_config['checkpoint'])
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Setup device
        self.device, self.device_ids = self._prepare_device(trainer_config['n_gpu'])
        self.train_logger = Logger(trainer_config['logs_train'])
        self.valid_logger = Logger(trainer_config['logs_valid'])

        self.trainer_config = trainer_config
        self.model_config = model_config
        self.epochs = trainer_config['epochs']
        self.start_epoch = 1
        self.metadata_checkpoint = self._load_metadata_checkpoint(model_config['metadata_checkpoint'])

    def _config_model_and_optimizer(self, model, optimizer):
        self.model = model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
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
            logging(self.train_logger, {"learning_rate": self.optimizer.get_lr()}, epoch)
            # Save checkpoints
            checkpoint_file_path = self._save_checkpoint(epoch)
            self._update_metadata_checkpoint({
                "epoch": epoch,
                "result": result,
                "file_path": checkpoint_file_path
            })

            self.scheduler.step(result_eval['loss'], epoch)

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

    def _load_metadata_checkpoint(self, file_path):
        if os.path.exists(file_path):
            return torch.load(file_path)
        return {
            "best_model": None,
            "epochs": []
        }

    def _update_metadata_checkpoint(self, metadata):
        self.metadata_checkpoint['epochs'].append(metadata)

        self.metadata_checkpoint['epochs'] = sorted(self.metadata_checkpoint['epochs'],
                                                    key=lambda k: k['result']['valid_metrics']['loss'])
        self.metadata_checkpoint['best_model'] = self.metadata_checkpoint['epochs'][0]
        # delete models if need
        epochs_info = sorted(self.metadata_checkpoint['epochs'], key=lambda k: k['epoch'], reverse=True)

        if len(epochs_info) > self.trainer_config['max_models_keep']:
            self.metadata_checkpoint['epochs'] = epochs_info[:self.trainer_config['max_models_keep']]
            for del_index in range(self.trainer_config['max_models_keep'], len(epochs_info)):
                epoch = epochs_info[del_index]
                if epoch['epoch'] == self.metadata_checkpoint['best_model']['epoch']:
                    self.metadata_checkpoint['epochs'].append(self.metadata_checkpoint['best_model'])
                else:
                    os.remove(os.path.join(self.model_config['checkpoint'], epoch['file_path']))

        torch.save(self.metadata_checkpoint, self.model_config['metadata_checkpoint'])

    def _save_checkpoint(self, epoch):
        # Save checkpoint
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "trainer_config": self.trainer_config,
            "model_config": self.model_config,
            'optimizer': self.optimizer.state_dict(),
        }
        file_name = os.path.join(self.checkpoint_dir,
                                 'epoch_{}.pth'.format(epoch))
        torch.save(state, file_name)
        return 'epoch_{}.pth'.format(epoch)

    def _resume_specific_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        """
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded. Resume training from epoch {}"
              .format(self.start_epoch))

    def _resume_checkpoint(self):
        if len(self.metadata_checkpoint['epochs']) > 0:
            latest_epoch = sorted(self.metadata_checkpoint['epochs'],
                                  key=lambda k: k['epoch'], reverse=True)[0]
            self._resume_specific_checkpoint(os.path.join(self.model_config['checkpoint'], latest_epoch['file_path']))
