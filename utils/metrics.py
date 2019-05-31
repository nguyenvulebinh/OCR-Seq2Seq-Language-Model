import editdistance
import torch


def acc_pair_string(str_1, str_2, is_word_level=True):
    if is_word_level:
        str_1 = str_1.split()
        str_2 = str_2.split()

    return (1. - editdistance.distance(str_1, str_2) / max(len(str_1), len(str_2))) * 100


def accuracy_calculate(output, ground_truth, vocab, is_word_level=True):
    output_indices = torch.argmax(output, dim=-1).cpu().numpy()
    ground_truth_indices = ground_truth.cpu().numpy()

    output_texts = [vocab.ids_to_sentence(item) for item in output_indices]
    ground_truth_texts = [vocab.ids_to_sentence(item) for item in ground_truth_indices]

    acc = 0
    for index, (o_item, g_item) in enumerate(zip(output_texts, ground_truth_texts)):
        acc += (acc_pair_string(o_item, g_item, is_word_level=is_word_level) - acc) / (index + 1)

    return acc

#
# from data_loader import ocr_loader
#
# output_str = [
#     'toi di choi',
#     'troi dep qua'
# ]
#
# ground_truth_str = [
#     'to di choi',
#     'troi dep qua'
# ]
# vocab = ocr_loader.get_or_create_vocab('', vocab_file='../models-bin/crnn/vocab.json')
# output_indices = torch.tensor([vocab.sentence_to_ids(item, length=200) for item in output_str])
# ground_truth_indices = torch.tensor([vocab.sentence_to_ids(item, length=100) for item in ground_truth_str])
#
# # print(torch.tensor(output_indices).shape)
# # print(ground_truth_indices)
#
#
# print(accuracy_calculate(output_indices, ground_truth_indices, vocab, is_word_level=True))
# print(accuracy_calculate(output_indices, ground_truth_indices, vocab, is_word_level=False))
