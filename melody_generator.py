import tensorflow as tf
import copy
import numpy as np

from melody_lib import MelodyEvent, MelodySequence
from melody_lib import VELOCITY_VALUE
from melody_lib import MAX_SILENCE, MIN_SILENCE, MAX_DURATION, MIN_DURATION
from melody_lib import DURATION_RANGE, SILENCE_RANGE, VELOCITY_RANGE
from encoder_decoder import LOWEST_MIDI_PITCH, HIGHEST_MIDI_PITCH

class MelodyGenerator(object):

    def __init__(self, build_graph_fn, checkpoint_dir, encoder_decoder,
                    random_rate, temperature, mode):
        if mode != 'generate_from_list' and mode != 'generate_from_eval':
            raise TypeError('mode must be "generate_from_list" or "generate_from_eval"')
        self._mode = mode
        with tf.Graph().as_default():
            build_graph_fn()
            saver = tf.train.Saver()
            self._session = tf.Session()
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            tf.logging.info('Checkpoint used: %s', checkpoint_file)
            saver.restore(self._session, checkpoint_file)
            if mode == 'generate_from_eval':
                tf.train.start_queue_runners(self._session)
        self._encoder = encoder_decoder
        self._random_rate = random_rate
        self._temperature = temperature

    def generate_from_pitch_list(self, pitch_lst):

        graph_inputs = self._session.graph.get_collection('inputs')[0]
        graph_initial_state = self._session.graph.get_collection('initial_state')[0]
        graph_final_state = self._session.graph.get_collection('final_state')[0]
        graph_duration_softmax = self._session.graph.get_collection('duration_softmax')[0]
        graph_silence_softmax = self._session.graph.get_collection('silence_softmax')[0]
        graph_velocity_softmax = self._session.graph.get_collection('velocity_softmax')[0]
        graph_temperature = self._session.graph.get_collection('temperature')[0]

        # feed_dict = {graph_initial_state}

        initial_state = self._session.run(graph_initial_state)
        melody = MelodySequence([])

        for pitch in pitch_lst:

            random_sample = True if np.random.rand() < self._random_rate else False


            input_ = self._get_graph_inputs(pitch_lst, melody)
            feed_dict = {graph_initial_state: initial_state,
                        graph_inputs: input_,
                        graph_temperature: self._temperature}

            (final_state,
            duration_softmax,
            silence_softmax,
            velocity_softmax) = self._session.run(
                                [graph_final_state,
                                graph_duration_softmax,
                                graph_silence_softmax,
                                graph_velocity_softmax], feed_dict)

            initial_state = final_state
            self._extent_melody(pitch_lst, melody, random_sample,
                                duration_softmax[0],
                                silence_softmax[0],
                                velocity_softmax[0])
        return melody

    def _get_graph_inputs(self, pitch_lst, melody):

        melody_len = len(melody)
        melody_copy = copy.deepcopy(melody)
        melody_copy.append(MelodySequence([pitch_lst[melody_len]])[0])

        input_ = self._encoder.events_to_input(melody_copy, melody_len)
        input_ = np.array(input_).reshape([1, 1, -1])
        return input_

    def _extent_melody(self, pitch_lst, melody,
                        random_sample,
                        duration_softmax,
                        silence_softmax,
                        velocity_softmax):
        if random_sample:
            pred_duration = np.random.choice(DURATION_RANGE, p = duration_softmax)
            pred_silence = np.random.choice(SILENCE_RANGE, p = silence_softmax)
            pred_velocity = np.random.choice(VELOCITY_RANGE, p = velocity_softmax)
        else:
            pred_duration = duration_softmax.argmax()
            pred_silence = silence_softmax.argmax()
            pred_velocity = velocity_softmax.argmax()

        idx = len(melody)
        event = MelodyEvent(pitch = pitch_lst[idx],
                            duration = pred_duration + MIN_DURATION,
                            silence = pred_silence + MIN_SILENCE,
                            velocity = VELOCITY_VALUE[pred_velocity])
        melody.append(event)
        del event

    def generate_from_eval_sequence(self, num_output):
        '''
        return 3 object
        one list of generated melodies, length equals to num_output,
        one is the original melody
        one is pure pitch list
        '''

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
            gen_ops.append(self._session.graph.get_collection(name)[0])

        temperature = self._session.graph.get_collection('temperature')[0]
#TODO make batch size bigger than 1        # lengths = tf.get_collection('lengths')[0]

        (inputs,
        duration_flat,
        silence_flat,
        velocity_flat,
        duration_softmax,
        silence_softmax,
        velocity_softmax) = self._session.run(gen_ops,
                                    feed_dict = {temperature: self._temperature})

        inputs = np.reshape(inputs, inputs.shape[1:])

        pred_melodies = []
        for i in range(num_output):
            pred_melodies.append(self._generate_once_from_eval(inputs,
                                                        duration_softmax,
                                                        silence_softmax,
                                                        velocity_softmax))

        ori_events = []
        raw_pitch = []
        for i, input_ in enumerate(inputs):
            pitch = np.argmax(input_[:self._encoder._pitch_range]) + LOWEST_MIDI_PITCH
            raw_pitch.append(int(pitch))
            ori_event = MelodyEvent(pitch,
                                    duration_flat[i] + MIN_DURATION,
                                    silence_flat[i] + MIN_SILENCE,
                                    VELOCITY_VALUE[velocity_flat[i]])
            ori_events.append(ori_event)

        ori_melody = MelodySequence(ori_events)
        raw_melody = MelodySequence(raw_pitch)

        return  pred_melodies, ori_melody, raw_melody

    def _generate_once_from_eval(self, inputs, duration_softmax,
                                silence_softmax, velocity_softmax):
        pred_duration = []
        pred_silence = []
        pred_velocity = []

        for i, input_ in enumerate(inputs):
            random_sample = True if np.random.rand() < self._random_rate else False
            if random_sample:
                pred_duration.append(
                    np.random.choice(DURATION_RANGE, p = duration_softmax[i]))
                pred_silence.append(
                    np.random.choice(SILENCE_RANGE, p = silence_softmax[i]))
                pred_velocity.append(
                    np.random.choice(VELOCITY_RANGE, p = velocity_softmax[i]))
            else:
                pred_duration.append(duration_softmax[i].argmax())
                pred_silence.append(silence_softmax[i].argmax())
                pred_velocity.append(velocity_softmax[i].argmax())

        pred_events = []
        for i, input_ in enumerate(inputs):
            pitch = np.argmax(input_[:self._encoder._pitch_range]) + LOWEST_MIDI_PITCH
            pred_event = MelodyEvent(pitch,
                                    pred_duration[i] + MIN_DURATION,
                                    pred_silence[i] + MIN_SILENCE,
                                    VELOCITY_VALUE[pred_velocity[i]])
            pred_events.append(pred_event)

        return MelodySequence(pred_events)
