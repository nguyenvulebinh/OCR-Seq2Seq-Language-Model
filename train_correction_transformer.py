from fairseq_cli import train
import sys
import utils as ocr_utils
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    ocr_utils.import_user_module('./plugin')
    sys.argv += [
        './data-bin/address/processed/',
        '--task', 'translation',
        '--source-lang', 'error',
        '--target-lang', 'correct',
        '--arch', 'transformer_iwslt_de_en',

        '--batch-size', '16',

        '--encoder-embed-dim', '100',
        '--encoder-layers', '4',
        '--decoder-embed-dim', '200',
        '--decoder-layers', '4',
        '--encoder-ffn-embed-dim', '512',
        '--decoder-ffn-embed-dim', '512',
        
        
        '--criterion', 'cross_entropy',
        # '--optimizer', 'adam',
        # '--lr-scheduler', 'reduce_lr_on_plateau',
        # '--lr', '0.05',

        '--optimizer', 'adam',
        '--adam-eps', '1e-04',
        '--lr', '0.005',
        '--min-lr', '1e-09',
        '--lr-scheduler', 'reduce_lr_on_plateau',
        '--weight-decay', '0.0001',
        '--adam-betas', '(0.9, 0.98)',
        '--clip-norm', '0.0',
        '--save-interval', '1',
        '--save-dir', 'checkpoints/correction_transformer',
    ]

    train.cli_main()
