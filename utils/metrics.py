import editdistance
import torch
import torch.nn.functional as F
# for unittest
import unittest


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


def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes

    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


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

class TestMetric(unittest.TestCase):

    def test_sequence_loss(self):
        score = torch.tensor([[[0.0, 10000.0, 0.0, 0.0], [0.0, 0.0, 100000.0, 0.0], [0.0, 100000.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0, 10000.0], [0.0, 0.0, 100000.0, 0.0], [0.0, 0.0, 0.0, 100000.0]]])
        y_pred = F.log_softmax(score, dim=-1)
        y_true = torch.tensor([[1, 2, 0], [3, 0, 0]])
        loss = sequence_loss(y_pred, y_true, mask_index=0).item()
        print(loss)
        self.assertEqual(loss, 0.0)


if __name__ == '__main__':
    unittest.main()
