import torch
from fairseq.data import Dictionary


class CTCLossDictionary(Dictionary):
    """
    Dictionary for image captioning tasks. This extends Dictionary by
    adding the blank symbol.
    """

    def __init__(self, blank='<blank>'):
        super().__init__()
        self.blank_word = blank
        self.blank_index = self.add_symbol(blank)
        self.nspecial = len(self.symbols)

    def blank(self):
        """Helper to get index of blank symbol"""
        return self.blank_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)

        sent = ' '.join(self[i] for i in tensor)
        return sent

    def ctc_string(self, indices):
        """
        convert id character to sentence
        :param indices:
        :return:
        """
        sentence = []
        current_index = -1
        for index in indices:
            if index == self.eos():
                break
            if index == self.blank():
                continue
            if index != current_index:
                current_index = index
                if index == self.blank():
                    continue
                else:
                    sentence.append(index)
            else:
                continue
        return sentence
