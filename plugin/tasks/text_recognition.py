import os
import torchvision.transforms as transforms

from fairseq.tasks import FairseqTask, register_task
from fairseq.data import Dictionary
import utils as ocr_utils

from plugin.data import CTCLossDictionary, TextRecognitionDataset
from plugin.data.transforms import ResizeWithPad


@register_task('text_recognition')
class TextRecognitionTask(FairseqTask):
    """
    Train a text recognition model.

    Args:
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target text
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--height', type=int, default=32,
                            help='image height size used for training (default: 32)')
        parser.add_argument('--width', type=int, default=200,
                            help='image width size used for training (default: 200)')
        parser.add_argument('--keep-ratio', action='store_true',
                            help='keep image size ratio when training')
        parser.add_argument('--no-token-pin-memory', default=False, action='store_true',
                            help='training using pined memory')
        # fmt: on

    def __init__(self, args, tgt_dict, transform=None):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.transform = transform

    @classmethod
    def load_dictionary(cls, filename, use_ctc_loss):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if use_ctc_loss:
            return CTCLossDictionary.load(filename)
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, ocr_utils.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def build_transform(cls, args):

        transform = transforms.Compose([
            ResizeWithPad(width=args.width, height=args.height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        use_ctc_loss = True if args.criterion == 'ctc_loss' else False
        tgt_dict = cls.load_dictionary(os.path.join(args.data, 'dict.txt'), use_ctc_loss)
        print('| target dictionary: {} types'.format(len(tgt_dict)))

        # build transform
        transform = cls.build_transform(args)

        return cls(args, tgt_dict, transform=transform)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # Read input images and targets
        image_names = []
        targets = []
        targets_simply = []
        target_lengths = []
        image_root = self.args.data
        label_path = os.path.join(self.args.data, '{}.txt'.format(split))
        with open(label_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line_simply = ocr_utils.remove_tone_line(line).split()
                line = line.strip().split()
                image_names.append(os.path.join(image_root, split, line[0]))
                targets.append(line[1:])
                targets_simply.append(line_simply[1:])
                target_lengths.append(len(line[1:]))

        assert len(image_names) == len(targets) == len(targets_simply) == len(target_lengths)
        for item_target, item_target_simply in zip(targets, targets_simply):
            assert len(item_target) == len(item_target_simply)
        print('| {} {} {} images'.format(self.args.data, split, len(image_names)))

        shuffle = True if split == 'train' else False
        use_ctc_loss = True if self.args.criterion == 'ctc_loss' else False
        self.datasets[split] = TextRecognitionDataset(
            image_names, targets, targets_simply, self.tgt_dict, tgt_sizes=target_lengths,
            shuffle=shuffle, transform=self.transform, use_ctc_loss=use_ctc_loss,
            input_feeding=True, append_eos_to_target=True,
        )

    def build_generator(self, args):
        if args.criterion == 'ctc_loss':
            from plugin.ctc_loss_generator import CTCLossGenerator
            return CTCLossGenerator(self.target_dictionary)
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                temperature=args.temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and
        # In order to use `CuDNN`, the "target" has max length 256,
        return (self.args.max_positions, 256)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
