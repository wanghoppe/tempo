import tensorflow as tf
import numpy as np
import pandas as pd
import os
from magenta.music.midi_io import sequence_proto_to_midi_file
from magenta.common import count_records
from string import ascii_uppercase

from generate_from_list import RANDOM_RATE, TEMPERATURE
from eval_ import TRAINING_DIR, EVAL_DIR, EVAL_DATA, TRAINING_DATA
from training import get_build_graph_fn, TRAINING_DIR, EVAL_DIR, TRAINING_DATA
from encoder_decoder import MelodyEncoderDecoder
from melody_generator import MelodyGenerator

GENERATE_DIR = 'generate/generate_from_eval'

CHECKPOINT_DIR = TRAINING_DIR
RANDOM_SAMPLE = True
BATCH_SIZE = 1 #TODO current only support one at a time

def main():
    encoder = MelodyEncoderDecoder()
    build_graph_fn = get_build_graph_fn(encoder_decoder = encoder,
        sequence_example_file_paths=EVAL_DATA, mode = 'generate_from_eval',
        batch_size = 1)
    generator =  MelodyGenerator(
                        build_graph_fn = build_graph_fn,
                        checkpoint_dir = CHECKPOINT_DIR,
                        encoder_decoder = encoder,
                        random_rate = RANDOM_RATE,
                        temperature = TEMPERATURE,
                        mode = 'generate_from_eval')

    num_input = count_records(EVAL_DATA)
    out_dir_id1 = 0
    out_dir_id2 = -1
    out_dir = ascii_uppercase[out_dir_id1] + '0'

    for i1 in range(num_input):
        if i1 % 20 == 0:
            print("now processing: {0}/ {1}".format((i1+1), num_input), end = '\r')
            if out_dir.endswith('9'):
                out_dir_id1 += 1
                out_dir_id2 = 0
            else:
                out_dir_id2 += 1
            out_dir = ascii_uppercase[out_dir_id1] + str(out_dir_id2)

        #每个生成3个旋律
        gens, ori, pitch = generator.generate_from_eval_sequence(num_output=3)

        out_dir_name = os.path.join(GENERATE_DIR, out_dir, str(i1 % 20))
        if not os.path.exists(out_dir_name):
            os.makedirs(out_dir_name)

        for i3, mel in enumerate(gens):
            sequence_proto_to_midi_file(mel.to_sequence(),
                            os.path.join(out_dir_name, 'gen{0}.mid'.format(i3)))
        sequence_proto_to_midi_file(ori.to_sequence(),
                        os.path.join(out_dir_name, 'ori.mid'))
        sequence_proto_to_midi_file(pitch.to_sequence(),
                        os.path.join(out_dir_name, 'pitch.mid'))

if __name__ == '__main__':
    main()
