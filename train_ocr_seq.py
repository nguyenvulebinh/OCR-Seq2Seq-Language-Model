from fairseq_cli import train
import sys
import utils as ocr_utils
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    ocr_utils.import_user_module('./plugin')
    sys.argv += [
        './data-bin/ocr-dataset/',
        '--user-dir', './plugin',
        '--task', 'text_recognition',
        '--arch', 'ocrseq',
        '--decoder-layers', '2',
        # '--backbone', 'vgg16_bn',
        '--backbone', 'cnn_baseline',
        '--batch-size', '1',

        '--height', '64',
        '--width', '1280',

        '--max-epoch', '51',
        '--criterion', 'ocrseq_loss',
        # '--num-workers', '1',

        '--optimizer', 'adam',
        '--adam-eps', '1e-04',
        '--lr', '0.005',
        '--min-lr', '1e-09',
        '--lr-scheduler', 'reduce_lr_on_plateau',
        '--weight-decay', '0.0001',
        # '--warmup-updates', '1000',
        # '--warmup-init-lr', '1e-07',

        '--adam-betas', '(0.9, 0.98)',
        '--clip-norm', '0.0',
        # '--weight-decay', '0.0',
        '--save-interval', '1',
        '--save-dir', 'checkpoints/ocrseq',
        # '--dataset-impl', 'lazy',
    ]

    train.cli_main()
