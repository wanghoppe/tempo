import tensorflow as tf
import numpy as np
import pandas as pd
from magenta.music.midi_io import sequence_proto_to_midi_file
import copy
import pickle
import os

from melody_lib import MelodyEvent, MelodySequence
from melody_lib import VELOCITY_VALUE
from melody_lib import MAX_SILENCE, MIN_SILENCE, MAX_DURATION, MIN_DURATION
from melody_lib import DURATION_RANGE, SILENCE_RANGE, VELOCITY_RANGE
from encoder_decoder import LOWEST_MIDI_PITCH, HIGHEST_MIDI_PITCH
from training import TRAINING_DIR
from encoder_decoder import MelodyEncoderDecoder
from training import get_build_graph_fn, TRAINING_DIR, EVAL_DIR, TRAINING_DATA
from eval_ import TRAINING_DIR, EVAL_DIR, EVAL_DATA, TRAINING_DATA
from melody_generator import MelodyGenerator


with open('pitch_lst.object', 'rb') as f:
    PITCH_LST = pickle.load(f)

#只用前100个音
PITCH_LST = PITCH_LST[:100]

# PITCH_LST = [60,61,65,78,65,62,34,55,56,57,58,59,60,62,62,65,65,65,69,69,54]


CHECKPOINT_DIR = TRAINING_DIR
OUT_DIR = 'generate/generate_from_list'

RANDOM_RATE = 0.8
TEMPERATURE = 1.0
NUM_OUTPUT = 10
# RANDOM_SAMPLE = False


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    encoder = MelodyEncoderDecoder()
    build_graph_fn = get_build_graph_fn(encoder_decoder = encoder,
                                        mode = 'generate_from_list',
                                        batch_size = 1)

    generator = MelodyGenerator(build_graph_fn = build_graph_fn,
                                checkpoint_dir = CHECKPOINT_DIR,
                                encoder_decoder = encoder,
                                random_rate = RANDOM_RATE,
                                temperature = TEMPERATURE,
                                mode = 'generate_from_list')
    for i in range(NUM_OUTPUT):
        melody = generator.generate_from_pitch_list(PITCH_LST)
        sequence_proto_to_midi_file(melody.to_sequence(),
                                    OUT_DIR + '/generate{0}.mid'.format(i))
    sequence_proto_to_midi_file(MelodySequence(PITCH_LST).to_sequence(),
                                OUT_DIR + '/ori_pitch.mid')

if __name__ == '__main__':
    main()
