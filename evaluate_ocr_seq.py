from fairseq import checkpoint_utils, data, options, tasks, utils
import utils as postag_utils
import utils as ocr_utils
import torch
from tqdm import tqdm
import sys
from torchvision import transforms
from plugin.data.transforms import ResizeWithPad
from plugin.data import data_utils
import os
from tqdm import tqdm

checkpoint_path = './checkpoints/ocrseq/checkpoint_best.pt'


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


def infer(image_path, model, dict, use_cuda):
    src_tokens = get_image(image_path).cuda() if use_cuda else get_image(image_path)
    cnn_feature = model.cnn_encoder(src_tokens)
    rnn_feature = model.rnn_encoder(cnn_feature)
    # encoder_out = model.projector_middle(rnn_feature)
    # log_probs = model.get_normalized_probs(encoder_out, log_probs=False)
    # print(dict.string(
    #     dict.ctc_string(log_probs.squeeze().argmax(-1).detach().cpu().numpy()))
    #       .replace(' ', '')
    #       .replace('|', ' '))
    decoder_input = {
        'encoder_out': rnn_feature,
        'encoder_padding_mask': None
    }
    output_tokens = [dict.eos()]
    for i in range(512):
        prev_output_tokens = torch.Tensor(output_tokens).unsqueeze(0).long()
        if use_cuda:
            prev_output_tokens = prev_output_tokens.cuda()
        decoder_out, _ = model.decoder(prev_output_tokens, encoder_out=decoder_input)
        output_tokens += [decoder_out.squeeze(0)[-1].argmax(-1).item()]
        if output_tokens[-1] == dict.eos():
            break
    return dict.string(torch.Tensor(output_tokens).long()[1:-1]).replace(' ', '').replace('|', ' ')


if __name__ == '__main__':
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

    labels, images_list = ocr_utils.load_raw_dataset('./data-bin/ocr-dataset/valid/')
    cer = 0
    wer = 0
    for i, _ in enumerate(tqdm(images_list)):
        result_infer = infer(images_list[i], model_ocr, task_ocr.tgt_dict, use_cuda)
        if labels is not None:
            wer += (ocr_utils.acc_pair_string(result_infer, labels[i], is_word_level=True) - wer) / (i + 1)
            cer += (ocr_utils.acc_pair_string(result_infer, labels[i], is_word_level=False) - cer) / (i + 1)
        print(images_list[i][images_list[i].rindex(os.path.sep):], result_infer)
    if labels is not None:
        print('cer: ', cer)
        print('wer: ', wer)

    # generator = task_ocr.build_generator(args)

    # translations = task_ocr.inference_step(generator, models, sample)
    # large_data_test_55715.jpg
    # </s> S ố | 3 5 , | H o à | S ư , | P h ư ờ n g | 0 , | Q u ậ n | P h ú | N h u ậ n , | T h à n h | p h ố | H ồ | C h í | M i n h </s>
    # print(translations)

    # print(infer('', model, task_ocr))
    # while True:
    #     img_name = input('\nInput: ')
    #     src_tokens = get_image(image_path + img_name).cuda() if use_cuda else get_image(image_path + img_name)
    #     cnn_feature = model.cnn_encoder(src_tokens)
    #     rnn_feature = model.rnn_encoder(cnn_feature)
    #     encoder_out = model.projector_middle(rnn_feature)
    #     log_probs = model.get_normalized_probs(encoder_out, log_probs=False)
    #     print(task_ocr.tgt_dict.string(
    #         task_ocr.tgt_dict.ctc_string(log_probs.squeeze().argmax(-1).detach().cpu().numpy()))
    #           .replace(' ', '')
    #           .replace('|', ' '))
    #     decoder_input = {
    #         'encoder_out': rnn_feature,
    #         'encoder_padding_mask': None
    #     }
    #     output_tokens = [task_ocr.tgt_dict.eos()]
    #     for i in range(200):
    #         prev_output_tokens = torch.Tensor(output_tokens).unsqueeze(0).long()
    #         if use_cuda:
    #             prev_output_tokens = prev_output_tokens.cuda()
    #         decoder_out, _ = model.decoder(prev_output_tokens, encoder_out=decoder_input)
    #         output_tokens += [decoder_out.squeeze(0)[-1].argmax(-1).item()]
    #         if output_tokens[-1] == task_ocr.tgt_dict.eos():
    #             break
    #     print(task_ocr.tgt_dict.string(torch.Tensor(output_tokens).long()[1:-1])
    #           .replace(' ', '')
    #           .replace('|', ' ')
    #           )
