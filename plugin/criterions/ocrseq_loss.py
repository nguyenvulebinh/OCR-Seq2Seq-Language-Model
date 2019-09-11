import torch
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('ocrseq_loss')
class OCRSeqLossCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super(FairseqCriterion, self).__init__()
        self.args = args
        self.blank_idx = task.target_dictionary.blank()
        self.padding_idx = task.target_dictionary.pad()

    def forward(self, model, sample, reduction='mean'):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        net_output_middle = net_output['encoder_output']
        net_output_final, _ = net_output['decoder_output']
        loss = self.compute_loss_ctc(model, net_output_middle, sample, reduction=reduction) + \
               self.compute_cross_entropy_loss(model, net_output_final, sample)
        sample_size = sample['nsentences'] if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.item(),
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_cross_entropy_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample['target'].view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='mean' if reduce else 'none',
        )
        return loss

    def compute_loss_ctc(
            self, model, net_output, sample,
            reduction='mean', zero_infinity=False,
    ):
        log_probs = model.get_normalized_probs(net_output, log_probs=True)
        targets = torch.cat(sample['target_simply'])  # Expected targets to have CPU Backend
        target_lengths = sample['target_length']
        input_lengths = torch.full((sample['nsentences'],), log_probs.size(0), dtype=torch.int32)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          blank=self.blank_idx, reduction=reduction,
                          zero_infinity=zero_infinity)
        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        # assert len(logging_outputs) == 1
        log = logging_outputs[0]
        loss = log.get('loss', 0)
        ntokens = log.get('ntokens', 0)
        batch_sizes = log.get('nsentences', 0)
        sample_size = log.get('sample_size', 0)
        agg_output = {
            'loss': loss,
            'ntokens': ntokens,
            'nsentences': batch_sizes,
            'sample_size': sample_size,
        }
        return agg_output
