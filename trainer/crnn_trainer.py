from .trainer_base import TrainerBase
import torch
from data_loader import ocr_loader
from models.crnn import CRNN
from tqdm import tqdm
from torch.nn import CTCLoss
import random
from utils import metrics


class CRNNTrainer(TrainerBase):

    def __init__(self, trainer_config, model_config, data_config, checkpoint_name=None):
        super(CRNNTrainer, self).__init__(trainer_config, model_config)
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.data_config = data_config

        self.vocab = ocr_loader.get_or_create_vocab(data_config['train']['root_data_path'],
                                                    keep_tone=model_config["keep_tone"],
                                                    vocab_file=model_config.get('vocab_file'))
        # Define model
        self.model = CRNN(model_config.get('image_width'),
                          model_config.get('image_height'),
                          model_config.get('image_channel'),
                          model_config.get('rnn_hidden_size'),
                          len(self.vocab))
        print(self.model)
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=trainer_config["lr"],
                                          betas=(0.9, 0.98), eps=1e-9)

        self.loss_func = CTCLoss(blank=self.vocab.get_blank_id())
        super(CRNNTrainer, self)._config_model_and_optimizer(self.model, self.optimizer)
        self.train_data_loader = ocr_loader.get_dataloader(data_config['train']['root_data_path'],
                                                           model_config.get('image_width'),
                                                           model_config.get('image_height'),
                                                           self.vocab,
                                                           keep_tone=model_config["keep_tone"],
                                                           batch_size=trainer_config['batch_size'])
        self.valid_data_loader = ocr_loader.get_dataloader(data_config['valid']['root_data_path'],
                                                           model_config.get('image_width'),
                                                           model_config.get('image_height'),
                                                           self.vocab,
                                                           keep_tone=model_config["keep_tone"],
                                                           batch_size=trainer_config['batch_size'])

        if checkpoint_name is not None:
            self._resume_checkpoint(checkpoint_name)

    def show_sample(self, output, ground_truth, num_sample=3):
        output_indices = torch.argmax(output, dim=-1).cpu().numpy()
        ground_truth_indices = ground_truth.cpu().numpy()
        list_index_show = random.sample(range(len(ground_truth)), min(num_sample, len(ground_truth)))
        print("\nShow sample:")
        for idx, index in enumerate(list_index_show):
            print("\t{}, \toutput: {}\n\t\tground: {}".format(idx,
                                                              self.vocab.ids_to_sentence(output_indices[index]),
                                                              self.vocab.ids_to_sentence(ground_truth_indices[index])))

    def _train(self, epoch_index):
        self.model.train()
        epoch_loss = 0
        epoch_acc_word = 0
        epoch_acc_char = 0
        output = None
        labels = None
        train_bar = tqdm(enumerate(self.train_data_loader),
                         total=len(self.train_data_loader))
        for batch_index, data_dict in train_bar:
            for name, tensor in data_dict.items():
                data_dict[name] = data_dict[name].to(self.device)
            images = data_dict['images']
            labels = data_dict['labels']
            labels_lengths = data_dict['labels_lengths']
            self.optimizer.zero_grad()
            output = self.model(images)
            batch_size, steps_lengths, vocab_size = output.size()
            loss = self.loss_func(output.transpose(0, 1),
                                  labels,
                                  torch.full((batch_size,), steps_lengths, dtype=torch.long).to(device=self.device),
                                  labels_lengths)

            loss.backward()
            self.optimizer.step()

            epoch_loss += (loss.item() - epoch_loss) / (batch_index + 1)
            epoch_acc_word += (metrics.accuracy_calculate(output,
                                                          labels,
                                                          self.vocab) - epoch_acc_word) / (batch_index + 1)
            epoch_acc_char += (metrics.accuracy_calculate(output,
                                                          labels,
                                                          self.vocab,
                                                          is_word_level=False) - epoch_acc_char) / (batch_index + 1)

            # show progress
            train_bar.set_description(desc="Train epoch {}".format(epoch_index), refresh=True)
            train_bar.set_postfix_str(
                s="loss step: {:.2f}, loss: {:.2f}, acc_word: {:.2f}%, acc_char: {:.2f}%".format(loss.item(),
                                                                                                 epoch_loss,
                                                                                                 epoch_acc_word,
                                                                                                 epoch_acc_char),
                refresh=True)
        self.show_sample(output, labels)

        return {
            'train_metrics': {
                'loss': epoch_loss,
                'acc_word': epoch_acc_word,
                'acc_char': epoch_acc_char,
            }
        }

    def _eval(self, epoch_index):
        self.model.eval()
        epoch_loss = 0
        epoch_acc_word = 0
        epoch_acc_char = 0
        output = None
        labels = None

        valid_bar = tqdm(enumerate(self.valid_data_loader),
                         total=len(self.valid_data_loader))

        for batch_index, data_dict in valid_bar:
            for name, tensor in data_dict.items():
                data_dict[name] = data_dict[name].to(self.device)
            images = data_dict['images']
            labels = data_dict['labels']
            labels_lengths = data_dict['labels_lengths']

            output = self.model(images)
            batch_size, steps_lengths, vocab_size = output.size()
            loss = self.loss_func(output.transpose(0, 1),
                                  labels,
                                  torch.full((batch_size,), steps_lengths, dtype=torch.long).to(device=self.device),
                                  labels_lengths)

            epoch_loss += (loss.item() - epoch_loss) / (batch_index + 1)
            epoch_acc_word += (metrics.accuracy_calculate(output,
                                                          labels,
                                                          self.vocab) - epoch_acc_word) / (batch_index + 1)
            epoch_acc_char += (metrics.accuracy_calculate(output,
                                                          labels,
                                                          self.vocab,
                                                          is_word_level=False) - epoch_acc_char) / (batch_index + 1)

            # show progress
            valid_bar.set_description(desc="Valid epoch {}".format(epoch_index), refresh=True)
            valid_bar.set_postfix_str(
                s="loss step: {:.2f}, loss: {:.2f}, acc_word: {:.2f}%, acc_char: {:.2f}%".format(loss.item(),
                                                                                                 epoch_loss,
                                                                                                 epoch_acc_word,
                                                                                                 epoch_acc_char),
                refresh=True)

        self.show_sample(output, labels)

        return {
            'valid_metrics': {
                'loss': epoch_loss,
                'acc_word': epoch_acc_word,
                'acc_char': epoch_acc_char,
            }
        }
