from __future__ import division
from __future__ import print_function
import torch
import unittest
from data_loader.lm.vocabulary import Vocabulary


class BeamEntry:
    "information about one single beam at specific time-step"

    def __init__(self):
        self.prTotal = 0  # blank and non-blank
        self.prNonBlank = 0  # non-blank
        self.prBlank = 0  # blank
        self.prText = 1  # LM score
        self.lmApplied = False  # flag if LM was already applied to this beam
        self.labeling = ()  # beam-labeling


class BeamState:
    "information about the beams at specific time-step"

    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal * x.prText)
        return [x.labeling for x in sortedBeams]


class CTCBeamSearch(object):
    """
    Process beam search
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ctc_output):
        """
        Decode ctc output to get output indicates
        :param ctc_output: softmax matrix with shape (batch_size, max_length, vocab_size)
        :return: batch_size of list indicates
        """
        results = []
        for sample in ctc_output:
            results.append(CTCBeamSearch.item_decode(sample, blankIdx=self.vocab.get_blank_id()))
        return results

    @staticmethod
    def addBeam(beamState, labeling):
        "add beam if it does not yet exist"
        if labeling not in beamState.entries:
            beamState.entries[labeling] = BeamEntry()

    @staticmethod
    def item_decode(mat, beamWidth=3, blankIdx=0):
        "beam search as described by the paper of Hwang et al. and the paper of Graves et al."

        maxT, maxC = mat.shape

        # initialise beam state
        last = BeamState()
        labeling = ()
        last.entries[labeling] = BeamEntry()
        last.entries[labeling].prBlank = 1
        last.entries[labeling].prTotal = 1

        # go over all time-steps
        for t in range(maxT):
            curr = BeamState()

            # get beam-labelings of best beams
            bestLabelings = last.sort()[0:beamWidth]

            # go over best beams
            for labeling in bestLabelings:

                # probability of paths ending with a non-blank
                prNonBlank = 0
                # in case of non-empty beam
                if labeling:
                    # probability of paths with repeated last char at the end
                    prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

                # probability of paths ending with a blank
                prBlank = last.entries[labeling].prTotal * mat[t, blankIdx]

                # add beam at current time-step if needed
                CTCBeamSearch.addBeam(curr, labeling)

                # fill in data
                curr.entries[labeling].labeling = labeling
                curr.entries[labeling].prNonBlank += prNonBlank
                curr.entries[labeling].prBlank += prBlank
                curr.entries[labeling].prTotal += prBlank + prNonBlank
                curr.entries[labeling].prText = last.entries[
                    labeling].prText  # beam-labeling not changed, therefore also LM score unchanged from
                curr.entries[
                    labeling].lmApplied = True  # LM already applied at previous time-step for this beam-labeling

                # extend current beam-labeling
                for c in range(maxC):
                    if c == blankIdx:
                        # Ignore case add a blank
                        continue
                    # add new char to current beam-labeling
                    newLabeling = labeling + (c,)

                    # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    if labeling and labeling[-1] == c:
                        prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                    else:
                        prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                    # add beam at current time-step if needed
                    CTCBeamSearch.addBeam(curr, newLabeling)

                    # fill in data
                    curr.entries[newLabeling].labeling = newLabeling
                    curr.entries[newLabeling].prNonBlank += prNonBlank
                    curr.entries[newLabeling].prTotal += prNonBlank

                    # apply LM

            # set new beam state
            last = curr

        # normalise LM scores according to beam-labeling-length
        last.norm()

        # sort by probability
        bestLabeling = last.sort()[0]  # get most probable labeling

        return bestLabeling


class TestCTCBeamSearch(unittest.TestCase):

    def test_ctc_beam_decode(self):
        ctc_output = torch.tensor([
            [[0.4, 0, 0.6], [0.4, 0, 0.6]],
            [[0.6, 0.4, 0], [0.6, 0.4, 0]]
        ])
        ctc_beam_decoder = CTCBeamSearch(Vocabulary())
        results = ctc_beam_decoder.decode(ctc_output)
        ground_truths = [(2,), (1,)]
        for i in range(len(ctc_output)):
            self.assertEqual(results[i], ground_truths[i])


if __name__ == '__main__':
    unittest.main()
