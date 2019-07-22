import sys
import utils as ocr_utils
import os
from fairseq import options, tasks, utils
import json
import re
from shutil import copyfile


def main(args):
    utils.import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)

    task = tasks.get_task(args.task)

    def refactor_label_file(split, file_path):
        if file_path.endswith('.json'):
            with open(os.path.join(args.destdir, split + '.txt'), 'w', encoding='utf-8') as file_label_process:
                with open(file_path, 'r', encoding='utf-8') as file_label:
                    data_label = json.loads(file_label.read())
                    for image_name, label in data_label.items():
                        # change space to |
                        label = re.sub(' ', '|', label)
                        # add space to separate character (char model)
                        label = re.sub('(.)', r'\1 ', label)
                        file_label_process.write("{} {}\n".format(image_name, label))
        else:
            copyfile(file_path, os.path.join(args.destdir, split + '.txt'))
        return os.path.join(args.destdir, split + '.txt')

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    if args.trainpref:
        args.trainpref = refactor_label_file('train', args.trainpref)
    if args.validpref:
        args.validpref = refactor_label_file('valid', args.validpref)

    if not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.tgtdict:
        tgt_dict = task.load_dictionary(args.tgtdict)
    else:
        assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
        tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)

    if tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    print("| Wrote preprocessed data to {}".format(args.destdir))


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    ocr_utils.import_user_module('./plugin')

    sys.argv += [
        '--task', 'text_recognition',
        '--user-dir', './plugin',
        '--trainpref', 'data-bin/ocr-dataset/train/labels.json',
        '--validpref', 'data-bin/ocr-dataset/valid/labels.json',
        '--padding-factor', '8',
        '--destdir', 'data-bin/ocr-dataset'
    ]

    cli_main()
