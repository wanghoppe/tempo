import numpy as np
from melody_lib import VELOCITY_VALUE
from melody_lib import MAX_SILENCE, MIN_SILENCE, MAX_DURATION, MIN_DURATION
from melody_lib import DURATION_RANGE, SILENCE_RANGE, VELOCITY_RANGE
from melody_lib import MelodyEvent, MelodySequence, STEPS_PER_SECOND
from sequence_example import make_sequence_example


DENSITY_VALUE = [1, 2, 4, 6, 8, 10, 12, 14, 16, 24, 32]
LOWEST_MIDI_PITCH = 36
HIGHEST_MIDI_PITCH = 84
# PITCH_RANGE = HIGHEST_MIDI_PITCH - LOWEST_MIDI_PITCH + 1

class MelodyEncoderDecoder(object):

    def __init__(self, min_pitch = LOWEST_MIDI_PITCH,
                max_pitch = HIGHEST_MIDI_PITCH):

        self._min_pitch = min_pitch
        self._max_pitch = max_pitch
        self._pitch_range = max_pitch - min_pitch + 1

    @property
    def input_size(self):
        return (self._pitch_range +
                DURATION_RANGE +         #前一个event的duration
                SILENCE_RANGE +          #前一个silence
                VELOCITY_RANGE +         #前一个velocity
                len(DENSITY_VALUE)       #前3s的平均每秒音的个数
                )

    @property
    def output_size(self):
        return (MAX_DURATION - MIN_DURATION + 1 +
                MAX_SILENCE - MIN_SILENCE + 1 +
                len(VELOCITY_VALUE))


    def events_to_input(self, events, position):
        '''Returns the input vector for the given position in the melody.
        Returns a self.input_size length list of floats

        '''
        offset = 0
        input_ = [0.0] * self.input_size
        input_[events[position].pitch - LOWEST_MIDI_PITCH] = 1.0
        if position == 0:
            return input_
        else:
            offset += self._pitch_range
            input_[offset + events[position-1].duration - MIN_DURATION] = 1.0
            offset += DURATION_RANGE
            input_[offset + events[position-1].silence - MIN_SILENCE] = 1.0
            offset += SILENCE_RANGE
            input_[offset + VELOCITY_VALUE.index(events[position-1].velocity)] = 1.0
            offset += VELOCITY_RANGE

            event_num = 0
            second = 0.0
            search_position = position - 1
            while second < 3.0:
                if search_position < 0:
                    break

                second += (events[search_position].duration + \
                            events[search_position].silence)/STEPS_PER_SECOND

                event_num += 1
                search_position -= 1

            density = event_num/ second
            for i, val in enumerate(DENSITY_VALUE):
                if val >= density:
                    density_id = i
                    break

            input_[offset + density_id] = 1.0

        return input_

    def events_to_label_dict(self, events, position):
        '''Returns the label list for the given position in the melody.

        Indices[0, 3]:
        [0]: duration in steps (1/32 note) in that pitch
        [1]: silence in steps (1/32 note) after that pitch
        [2]: one of values in VELOCITY_VALUE, reprenting the velocity
        '''

        label = {'duration':events[position].duration - MIN_DURATION,
                'silence':events[position].silence,
                'velocity':VELOCITY_VALUE.index(events[position].velocity)}

        return label

    def decode_single_softmax(self,
                        input_,
                        duration_softmax,
                        silence_softmax,
                        velocity_softmax):
        '''Decode softmaxed label vector into a MelodyEvent object
        input: three self.outsize length softmax vectors
        output： a MelodyEvent object

        input_.shape(self.input_size,)
        duration_softmax.shape = (65,)
        silence_softmax.shape = (33,)
        velocity_softmax.shape = (15,)
        '''

        assert len(duration_softmax) == 65
        assert len(silence_softmax) == 33
        assert len(velocity_softmax) == 15

        pitch = input_.argmax() + LOWEST_MIDI_PITCH
        duration = duration_softmax.argmax()
        silence = silence_softmax.argmax()
        velocity = VELOCITY_VALUE[velocity_softmax.argmax()]

        return MelodyEvent(pitch  = pitch,
                        duration = duration,
                        silence = silence,
                        velocity = velocity)

    def encode(self, events):
        """Returns a SequenceExample for the given event sequence.

        Args:
          events: A list-like sequence of events.

        Returns:
          A tf.train.SequenceExample containing inputs and labels.
        """
        inputs = []
        labels = []
        for i in range(len(events)):
            inputs.append(self.events_to_input(events, i))
            labels.append(self.events_to_label_dict(events, i))
        return make_sequence_example(inputs, labels)
