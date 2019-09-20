from fairseq import checkpoint_utils, options, tasks
import utils as ocr_utils
import torch
import sys
from torchvision import transforms
from plugin.data.transforms import ResizeWithPad
from plugin.data import data_utils
import os
from tqdm import tqdm
from plugin.modules import ctc_beam_search
from fairseq import utils

checkpoint_path = './checkpoints/crnn-encoder/checkpoint_best.pt'


# image_path = './data-bin/test/'


def get_image(image_path):
    transform = transforms.Compose([
        ResizeWithPad(width=1280, height=64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = data_utils.default_loader(image_path)
    return transform(image).unsqueeze(0)


def load_model():
    # load param
    ocr_utils.import_user_module('./plugin')
    sys.argv += [
        './dicts/ocr',
        '--user-dir', './plugin',
        '--task', 'text_recognition',
        '--criterion', 'ctc_loss',
    ]
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)

    use_cuda = torch.cuda.is_available() and not args.cpu

    task_ocr = tasks.setup_task(args)
    print('| loading model from {}'.format(checkpoint_path))
    models, _model_args = checkpoint_utils.load_model_ensemble([checkpoint_path], task=task_ocr)
    model_ocr = models[0]
    if use_cuda:
        model_ocr.cuda()
    print(model_ocr)

    return model_ocr, task_ocr, use_cuda


def infer(image_path, model, task_ocr, use_cuda):
    dict = task_ocr.tgt_dict
    src_tokens = get_image(image_path).cuda() if use_cuda else get_image(image_path)
    crnn_output = model(src_tokens)['encoder_output'].squeeze(1)
    crnn_output_prob = utils.softmax(crnn_output, dim=1)
    label_out = ctc_beam_search.get_output(crnn_output_prob.detach().numpy(), dict.symbols, dict.blank())
    return label_out


if __name__ == '__main__':
    model_ocr, task_ocr, use_cuda = load_model()

    labels, images_list = ocr_utils.load_raw_dataset('./data-bin/ocr-dataset/valid/')
    cer = 0
    wer = 0
    for i, _ in enumerate(tqdm(images_list)):
        result_infer = infer(images_list[i], model_ocr, task_ocr, use_cuda)
        if labels is not None:
            wer += (ocr_utils.acc_pair_string(result_infer, labels[i], is_word_level=True) - wer) / (i + 1)
            cer += (ocr_utils.acc_pair_string(result_infer, labels[i], is_word_level=False) - cer) / (i + 1)
        print(images_list[i][images_list[i].rindex(os.path.sep):], result_infer)
    if labels is not None:
        print('cer: ', cer)
        print('wer: ', wer)
