class Vocabulary(object):
    """
    Class to process text and extract Vocabulary for mapping
    """

    def __init__(self, token_to_idx=None):
        """
        :param token_to_idx:  a pre-existing map of tokens to indices
        """
        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx

        self._idx_to_token = {token: idx for idx, token in self._token_to_idx.items()}

    def to_serializable(self):
        """
        return a dictionary that can be serialized
        :return:
        """
        return {
            'token_to_idx': self._token_to_idx
        }

    @classmethod
    def from_serializable(cls, contents):
        """
        instantiates the Vocabulary from a serialized dictionary
        :param contents:
        :return:
        """
        return cls(**contents)

    def add_token(self, token):
        """
        Update mapping dicts based on the token
        :param token: str, the item to add into the Vocabulary
        :return:
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """
        Retrieve the index associated with the token or the UNK index if token isn't present
        :param token: (str) the token to look up
        :return: the index corresponding to the token
        Notes: 'unk_index' needs to be >= 0 (having been added into the Vocabulary)
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """
       Return the token associated with the index
       :param index: the index to lookup
       :return: the token corresponding to the index
       Raises KeyError if the index is not in the Vocabulary
       """
        if index not in self._idx_to_token:
            raise KeyError("The index {} is not in the Vocabulary".format(index))
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>", end_seq_token="<END>"):
        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({
            'unk_token': self._unk_token,
            'mask_token': self._mask_token,
            "begin_seq_token": self._begin_seq_token,
            'end_seq_token': self._end_seq_token
        })
        return contents

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx.get(token)
