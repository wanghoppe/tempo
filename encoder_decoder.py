import numpy as np
from melody_lib import VELOCITY_VALUE
from melody_lib import MAX_SILENCE, MIN_SILENCE, MAX_DURATION, MIN_DURATION
from melody_lib import MelodyEvent, MelodySequence
from sequence_example import make_sequence_example


LOWEST_MIDI_PITCH = 24
HIGHEST_MIDI_PITCH = 105

class MelodyEncoderDecoder(object):

    def __init__(self, min_pitch = LOWEST_MIDI_PITCH,
                max_pitch = HIGHEST_MIDI_PITCH):

        self._min_pitch = min_pitch
        self._max_pitch = max_pitch
        self._pitch_range = max_pitch - min_pitch

    @property
    def input_size(self):
        return (self._pitch_range + 1)

    @property
    def output_size(self):
        return (MAX_DURATION - MIN_DURATION + 1 +
                MAX_SILENCE - MIN_SILENCE + 1 +
                len(VELOCITY_VALUE))


    def events_to_input(self, events, position):
        '''Returns the input vector for the given position in the melody.
        Returns a self.input_size length list of floats

        '''
        input_ = [0.0] * self.input_size
        input_[events[position].pitch - LOWEST_MIDI_PITCH] = 1.0
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
