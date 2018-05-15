from training import get_build_graph_fn, TRAINING_DIR, EVAL_DIR, TRAINING_DATA
from eval_ import TRAINING_DIR, EVAL_DIR, EVAL_DATA, TRAINING_DATA
from encoder_decoder import MelodyEncoderDecoder
import tensorflow as tf
import numpy as np
import pandas as pd


from melody_lib import MelodyEvent, MelodySequence
from melody_lib import VELOCITY_VALUE
from melody_lib import MAX_SILENCE, MIN_SILENCE, MAX_DURATION, MIN_DURATION
from melody_lib import DURATION_RANGE, SILENCE_RANGE, VELOCITY_RANGE
from encoder_decoder import LOWEST_MIDI_PITCH


from magenta.music.midi_io import sequence_proto_to_midi_file

CHECKPOINT_DIR = '/media/hoppe/ECCC8857CC881E48/Code/tempo/training/training'
TEMPERATURE = 0.6

def run_generate(build_graph_fn, checkpoint_file):
    with tf.Graph().as_default():
        build_graph_fn()
        saver = tf.train.Saver()
        session = tf.Session()
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_file)
        tf.logging.info('Checkpoint used: %s', checkpoint_file)
        saver.restore(session, checkpoint_file)
        tf.train.start_queue_runners(session)


        global_step = tf.train.get_or_create_global_step()
        gen_ops = []
        generate_list_name = [
            'inputs',
            'duration_flat',
            'silence_flat',
            'velocity_flat',
            'duration_softmax',
            'silence_softmax',
            'velocity_softmax'
        ]
        for name in generate_list_name:
            gen_ops.append(tf.get_collection(name)[0])

        temperature = tf.get_collection('temperature')[0]

        (inputs,
        duration_flat,
        silence_flat,
        velocity_flat,
        duration_softmax,
        silence_softmax,
        velocity_softmax) = session.run(gen_ops, feed_dict = {temperature: TEMPERATURE})

    inputs = np.reshape(inputs, inputs.shape[1:])

    pred_duration = []
    pred_silence = []
    pred_velocity = []

    for i, input_ in enumerate(inputs):
        pred_duration.append(np.random.choice(DURATION_RANGE, p = duration_softmax[i]))
        pred_silence.append(np.random.choice(SILENCE_RANGE, p = silence_softmax[i]))
        pred_velocity.append(np.random.choice(VELOCITY_RANGE, p = velocity_softmax[i]))
        print(duration_softmax[i].argmax())

    print('duration_ori_dis', '\n', pd.Series(duration_flat).value_counts(), '\n\n')
    print('silence_ori_dis', '\n', pd.Series(silence_flat).value_counts(), '\n\n')
    print('velocity_ori_dis', '\n', pd.Series(velocity_flat).value_counts(), '\n\n')

    print('duration_pred_dis', '\n', pd.Series(pred_duration).value_counts(), '\n\n')
    print('silence_pred_dis', '\n', pd.Series(pred_silence).value_counts(), '\n\n')
    print('velocity_pred_dis', '\n', pd.Series(pred_velocity).value_counts(), '\n\n')


    ori_events = []
    pred_events = []
    raw_pitch = []
    for i, input_ in enumerate(inputs):
        pitch = np.argmax(input_) + LOWEST_MIDI_PITCH
        raw_pitch.append(int(pitch))
        ori_event = MelodyEvent(pitch,
                                duration_flat[i] + MIN_DURATION,
                                silence_flat[i] + MIN_SILENCE,
                                VELOCITY_VALUE[velocity_flat[i]])

        pred_event = MelodyEvent(pitch,
                                pred_duration[i] + MIN_DURATION,
                                pred_silence[i] + MIN_SILENCE,
                                80)
        ori_events.append(ori_event)
        pred_events.append(pred_event)

    ori_melody = MelodySequence(ori_events)
    pred_melody = MelodySequence(pred_events)
    raw_melody = MelodySequence(raw_pitch)

    sequence_proto_to_midi_file(ori_melody.to_sequence(), 'generate/ori_melody.mid')
    sequence_proto_to_midi_file(pred_melody.to_sequence(), 'generate/pred_melody.mid')
    sequence_proto_to_midi_file(raw_melody.to_sequence(), 'generate/raw_pitch.mid')
    print("generation completed, please go to ./generate dir to see the results")

def main():
    encoder = MelodyEncoderDecoder()
    build_graph_fn = get_build_graph_fn(encoder_decoder = encoder,
        sequence_example_file_paths=EVAL_DATA, mode = 'generate',
        batch_size = 1)
    run_generate(build_graph_fn, CHECKPOINT_DIR)

if __name__ == '__main__':
    main()
