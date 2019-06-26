import torch
from models.crnn import CRNN
from data_loader import ocr_loader
import os
from utils.ctc_beam_search import CTCBeamSearch
from data_loader.lm.vocabulary import Vocabulary

def load_crnn_model(model_path):
    model_info = torch.load(model_path, map_location='cpu')
    model_config = model_info.get('model_config')
    vocab = Vocabulary.from_serializable(model_info.get('vocab'))
    # vocab = ocr_loader.get_or_create_vocab(vocab_file=model_config.get('vocab_file'))
    # Define model
    model = CRNN(model_config.get('image_width'),
                 model_config.get('image_height'),
                 model_config.get('image_channel'),
                 model_config.get('rnn_hidden_size'),
                 len(vocab),
                 use_vis_attn=model_config.get('use_vis_attn'))
    model.load_state_dict(model_info['state_dict'], strict=True)
    return model, vocab, model_config


if __name__ == '__main__':
    model_crnn, vocab, model_config = load_crnn_model('/Users/nguyenbinh/Programing/Python/OCR-Seq2Seq-Language-Model/'
                                                      'models-bin/crnn/checkpoints/epoch_1.pth')

    image_path = '/Users/nguyenbinh/Programing/Python/OCR-Seq2Seq-Language-Model/' \
                 'data-bin/raw/processed/valid/large_data_test_667.jpg'

    print(model_crnn)
    infer_iterator_data = ocr_loader.get_infer_data(model_config.get('image_width'),
                                                    model_config.get('image_height'),
                                                    # data_path='/Users/nguyenbinh/Programing/Python/OCR-Seq2Seq-'
                                                    #           'Language-Model/data-bin/raw/processed/valid',

                                                    data_path='/Users/nguyenbinh/Programing/Python/OCR-Seq2Seq-'
                                                              'Language-Model/data-bin/raw/1015_Private Test',

                                                    batch_size=4)
    for img_names, img_tensors in infer_iterator_data:
        crnn_outputs = model_crnn(img_tensors, softmax=True)
        ctc_beam_decoder = CTCBeamSearch(vocab)
        output_indices_beam = ctc_beam_decoder.decode(crnn_outputs.detach().numpy())
        output_indices_greedy = torch.argmax(crnn_outputs, dim=-1).cpu().numpy()
        for i in range(len(img_names)):
            label_greedy = vocab.ids_to_sentence(output_indices_greedy[i])
            label_beam = vocab.ids_to_sentence(output_indices_beam[i])
            print("{}:\nBeam result: \t{}\nGreedy result: \t{}".format(img_names[i][img_names[i].rindex(os.path.sep) + 1:],
                                                                   label_beam,
                                                                   label_greedy))
            # print(img_names[i][img_names[i].rindex(os.path.sep) + 1:], label)
        break
