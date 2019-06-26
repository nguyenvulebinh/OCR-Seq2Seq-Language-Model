from .trainer_base import TrainerBase
import torch
from data_loader import ocr_loader
from models.gru_encoder_decoder import GRUEncodeDecode
from tqdm import tqdm
import random
from utils import metrics


class GRUEnDeTrainer(TrainerBase):

    def __init__(self, trainer_config, model_config, data_config, checkpoint_name=None):
        super(GRUEnDeTrainer, self).__init__(trainer_config, model_config, data_config)
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.data_config = data_config

        # Define model
        self.model = GRUEncodeDecode(image_width=model_config.get('image_width'),
                                     image_height=model_config.get('image_height'),
                                     image_channel=model_config.get('image_channel'),
                                     encoder_hidden_size=model_config['encoder_hidden_size'],
                                     decoder_hidden_size=model_config['decoder_hidden_size'],
                                     embedding_dim=model_config['embedding_dim'],
                                     vocab_size=len(self.vocab),
                                     max_target_len=model_config['max_target_len'],
                                     vocab=self.vocab,
                                     num_layers=model_config["num_layers"],
                                     drop=0.3,
                                     device=self.device)
        print(self.model)
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=trainer_config["lr"],
                                          betas=(0.9, 0.98), eps=1e-9)

        super(GRUEnDeTrainer, self)._config_model_and_optimizer(self.model, self.optimizer)
        self.train_data_loader = ocr_loader.get_dataloader(data_config['train']['root_data_path'],
                                                           model_config.get('image_width'),
                                                           model_config.get('image_height'),
                                                           self.vocab,
                                                           keep_tone=model_config["keep_tone"],
                                                           batch_size=trainer_config['batch_size'],
                                                           max_target_len=model_config['max_target_len'])
        self.valid_data_loader = ocr_loader.get_dataloader(data_config['valid']['root_data_path'],
                                                           model_config.get('image_width'),
                                                           model_config.get('image_height'),
                                                           self.vocab,
                                                           keep_tone=model_config["keep_tone"],
                                                           batch_size=trainer_config['batch_size'],
                                                           max_target_len=model_config['max_target_len'])

        if checkpoint_name is not None:
            self._resume_specific_checkpoint(checkpoint_name)
        else:
            self._resume_checkpoint()

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
            teach_force_labels = data_dict['teach_force_labels']
            self.optimizer.zero_grad()

            # Determine if we are using teacher forcing this iteration
            output = self.model(images,
                                teach_force_labels,
                                teacher_forcing_ratio=self.trainer_config['teacher_forcing_ratio'])
            # use_teacher_forcing = True if random.random() < self.trainer_config['teacher_forcing_ratio'] else False
            # if use_teacher_forcing:
            #     output = self.model(images, teach_force_labels)
            # else:
            #     output = self.model(images)
            loss = metrics.sequence_loss(output, labels, self.vocab.get_pad_id())

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

            output = self.model(images)

            loss = metrics.sequence_loss(output, labels, self.vocab.get_pad_id())

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
