import sys
from collections import namedtuple
from preprocess_correction import clean_input_correction, clean_output_correction
import torch

from fairseq import checkpoint_utils, options, tasks, utils

# checkpoint_path = './checkpoints/correction_lstm/checkpoint_best.pt'
input_list = [
    "t i e n | d u o n g , | d g | a n h , | h a | n o i",
    "t h o n | t h o | h a , | x a | q u n g | s o n , | t h i | x a | b a | d o n | t n h | u a n g | b i n h"
]
Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    for src_str in input:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def load_model(checkpoint_path='./checkpoints/correction_lstm/checkpoint_best.pt',
               dict_path='dicts/correction/'):
    # load param
    sys.argv += [
        '--path', checkpoint_path,
        '--beam', '5',
        '--source-lang', 'error',
        '--target-lang', 'correct',
        '--buffer-size', '2',
        '--batch-size', '2',
        '--print-alignment',
        dict_path
    ]

    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)

    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    # print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)
    return args, task, models, generator


def infer(input_raw, args, task, models, generator):
    # input_raw = [clean_input_correction(item) for item in input_list]
    output_raw = []
    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    use_cuda = torch.cuda.is_available() and not args.cpu
    # # Hack to support GPT-2 BPE
    # if args.remove_bpe == 'gpt2':
    #     from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
    #     decoder = get_encoder(
    #         'fairseq/gpt2_bpe/encoder.json',
    #         'fairseq/gpt2_bpe/vocab.bpe',
    #     )
    #     encode_fn = lambda x: ' '.join(map(str, decoder.encode(x)))
    # else:
    decoder = None
    encode_fn = lambda x: x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )
    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0
    for inputs in buffered_read(input_raw, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                # print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                if decoder is not None:
                    hypo_str = decoder.decode(map(int, hypo_str.strip().split()))
                output_raw.append(hypo_str)
                # print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                # print('P-{}\t{}'.format(
                #     id,
                #     ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                # ))
                # if args.print_alignment:
                #     print('A-{}\t{}'.format(
                #         id,
                #         ' '.join(map(lambda x: str(utils.item(x)), alignment))
                #     ))

        # update running id counter
        start_id += len(inputs)
    return [clean_output_correction(item) for item in output_raw]


if __name__ == '__main__':
    args, task, models, generator = load_model()
    # input_list = [clean_input_correction(item) for item in input_list]
    output_list = infer(input_list, args, task, models, generator)
    for str_in, str_out in zip(input_list, output_list):
        print(str_in, '\n', str_out)
    # cli_main()
