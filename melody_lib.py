import math
from magenta.music.events_lib import EventSequence
from magenta.music import constants
from magenta.protobuf import music_pb2
from magenta.pipelines import statistics
from magenta.music import sequences_lib
from collections import namedtuple
import numpy as np

MAX_VELOCITY = 127
MAX_PITCH = 105

NONE_PITCH_EVENT = 0
# VELOCITY_VALUE = [8,16,24,32,40,48,56,64,72,80,88,96,104,112,120]
VELOCITY_VALUE = [63,71,79,87,95,103,111,119,127]
MIN_DURATION = 1
MAX_DURATION = 64
MIN_SILENCE = 0
MAX_SILENCE = 32

DEFUALT_VELOCITY = VELOCITY_VALUE[-2]
DEFUALT_DURATION = 8
DEFUALT_SILENCE = 0
STEPS_PER_SECOND = 32
MAX_EVENT_LENGTH = 128

DURATION_RANGE = MAX_DURATION - MIN_DURATION + 1
SILENCE_RANGE = MAX_SILENCE - MIN_SILENCE + 1
VELOCITY_RANGE = len(VELOCITY_VALUE)

STANDARD_PPQ = constants.STANDARD_PPQ


Data = namedtuple('Data', 'pitch start end velocity')

class MelodyEvent(object):


    def __init__(self, pitch, duration, silence, velocity):

        if duration > 64:
            # clip longer duration to 64
            duration = 64
        assert ((MIN_DURATION <= duration) and \
                (MIN_SILENCE <= silence) and \
                (silence <= MAX_SILENCE))
        assert velocity in VELOCITY_VALUE
        self.pitch = pitch
        self.duration = duration
        # self.velocity = self._get_velocity(velocity)
        self.velocity = velocity
        self.silence = silence

    # def _get_velocity(self, ori_velocity):
    #     '''Convert velocity into one of int in velocity_VALUE'''
    #     assert 0 <= ori_velocity <= 127
    #     return math.ceil(ori_velocity / 127 * 15) * 8


    def __repr__(self):
        return ('MelodyEvent(pitch:{0}, duration:{1}, velocity:{2}, ' +
        'silence:{3})')\
        .format(self.pitch, self.duration, self.velocity, self.silence)

    def __eq__(self, other):
        if not isinstance(other, MelodyEvent):
            return False
        return (self.pitch == other.pitch and
                self.duration == other.duration and
                self.velocity == other.velocity)


class MelodySequence(EventSequence):

    def __init__(self, events = [], start_step = 0,
                steps_per_second = STEPS_PER_SECOND):

        if isinstance(events, list):
            self._events = self._from_event_list(events)
        self._start_step = start_step
        self._steps_per_second = steps_per_second
        self._end_step = start_step + self._get_total_steps()

    def _from_event_list(self, events):

        if events and isinstance(events[0], int):
            new_events = []
            for pitch in events:
                assert 0 <= pitch < MAX_PITCH
                new_events.append(
                    MelodyEvent(pitch, DEFUALT_DURATION,
                                DEFUALT_SILENCE, DEFUALT_VELOCITY))
            return new_events
        else:
            return events


    def _get_total_steps(self):
        steps = 0
        for event in self:
            steps += event.duration + event.silence
        return steps

    def __iter__(self):
        return iter(self._events)

    def __getitem__(self, key):
        """Returns the slice or individual item."""
        if isinstance(key, int):
            return self._events[key]
        elif isinstance(key, slice):
            events = self._events.__getitem__(key)
            return type(self)(events=events,
                            start_step=self.start_step + (key.start or 0),
                            duration_step_per_quater = self._steps_per_second)

    def __len__(self):
        return len(self._events)

    def __deepcopy__(self, memo=None):
        return type(self)(events=copy.deepcopy(self._events, memo),
                      start_step=self.start_step,
                      duration_step_per_quater = self._steps_per_second)

    def _reset(self):
        self._events = []
        self._start_step = 0
        self._end_step = 0
        self._steps_per_second = STEPS_PER_SECOND

    def append(self, event):
        self._events.append(event)
        self._end_step += 1

    @property
    def start_step(self):
        return self._start_step

    @property
    def end_step(self):
        return self._end_step

    @property
    def total_steps(self):
        return self._end_step - self._start_step

    @property
    def steps(self):
        return list(range(self._start_step, self._end_step))

    @property
    def steps_per_second(self):
        return self._steps_per_second

    #Advance Method

    def to_sequence(self,
                    qpm = 120,
                    sequence_start_time = 0.0,
                    instrument = 0,
                    program = 0):

        second_per_step = 1/self._steps_per_second

        sequence = music_pb2.NoteSequence()
        sequence.tempos.add().qpm = qpm
        sequence.ticks_per_quarter = STANDARD_PPQ

        sequence_start_time += self.start_step * second_per_step

        time = sequence_start_time
        for i, event in enumerate(self):
            note = sequence.notes.add()
            note.start_time = time
            note.pitch = event.pitch
            note.velocity = event.velocity
            note.instrument = instrument
            note.program = program
            note.end_time = event.duration * second_per_step + time

            time += (event.duration + event.silence) * second_per_step

        if sequence.notes:
          sequence.total_time = sequence.notes[-1].end_time

        return sequence

    def get_note_histogram(self):
        """Gets a histogram of the note occurrences in a melody.

        Returns:
          A list of 12 ints, one for each note value (C at index 0 through B at
          index 11). Each int is the total number of times that note occurred in
          the melody.
        """

        np_melody = np.array([a.pitch for a in self._events], dtype=int)
        return np.bincount(np_melody % 12,
                           minlength=12)


def _get_notes_tuple(quantized_sequence,
                    instrument=0,
                    filter_drums=True):

    # sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

    # no note in the quantized note sequence
    if not quantized_sequence.notes:
        return
    else:
        notes = sorted([n for n in quantized_sequence.notes
                if n.instrument == instrument],
                key=lambda note: note.quantized_start_step)

        last_quantized_step = max([n.quantized_end_step
                for n in quantized_sequence.notes
                if n.instrument == instrument ])

        notes_to_populate = []
        current_notes = []
        cur_start_step = 0
        cur_end_step = 0
        ith_note = 0
        add_note_to_curnotes = True

        while cur_end_step < last_quantized_step:

            if add_note_to_curnotes:
                try:
                    current_notes.append(notes[ith_note])
                    if notes[ith_note].quantized_start_step:
                        cur_start_step = \
                            max(notes[ith_note].quantized_start_step, cur_start_step)
                    else:
                        pass

                    while notes[ith_note+1].quantized_start_step == cur_start_step:
                        current_notes.append(notes[ith_note+1])
                        ith_note += 1

                except IndexError:
                    pass
                add_note_to_curnotes = False

            for cur_n in current_notes.copy():
                if cur_n.quantized_end_step <= cur_start_step:
                    current_notes.remove(cur_n)

            cur_note = sorted(current_notes, key = lambda a: a.pitch)[-1]

            # curnotes_earlest_end_step = \
            #     sorted(current_notes,
            #         key = lambda a: a.quantized_end_step)[0].quantized_end_step

            try:
                next_note_start_step = notes[ith_note+1].quantized_start_step
            except IndexError:
                next_note_start_step = quantized_sequence.total_quantized_steps

            cur_end_step = min(next_note_start_step,
                                # curnotes_earlest_end_step,
                                cur_n.quantized_end_step)

            if cur_end_step == next_note_start_step or \
                cur_end_step == cur_n.quantized_end_step:
                add_note_to_curnotes = True
                ith_note += 1

            if notes_to_populate and \
                cur_note.pitch == notes_to_populate[-1].pitch and \
                cur_note.quantized_start_step < cur_start_step:
                    last_note = notes_to_populate[-1]

                    notes_to_populate[-1] = Data(last_note.pitch,
                                            last_note.start,
                                            cur_end_step,
                                            last_note.velocity)
            else:
                notes_to_populate.append(Data(cur_note.pitch,
                                        cur_start_step,
                                        cur_end_step,
                                        cur_note.velocity))

            cur_start_step = cur_end_step

    return notes_to_populate

def extract_melodies(quantized_sequence,
                    gap_step = 32,
                    min_unique_pitches=4,
                    max_melody_events=MAX_EVENT_LENGTH,
                    min_melody_events=9,
                    filter_drums=True):
    '''return a list of MelodySequence, used to great 'datainfo.txt' file'''

    sequences_lib.assert_is_quantized_sequence(quantized_sequence)

    melodies = []
    stats = dict()

    stats['melody_lengths_in_events'] = statistics.Histogram(
        'melody_lengths_in_events', [9, 20, 30, 40, 50, 100, 200, 500])
    stats['total_length_in_pitch'] = statistics.Counter('total_length_in_pitch')
    stats['total_length_in_steps'] = statistics.Counter('total_length_in_steps')

    instruments = set([n.instrument for n in quantized_sequence.notes])

    orig_melodies = []
    for instrument in instruments:
        lst = _get_notes_tuple(quantized_sequence,
                                instrument = instrument,
                                filter_drums = True)
        lst = make_relative_velocity(lst)

        events = []
        for i in range(len(lst) - 1):

            if lst[i+1].start - lst[i].end > gap_step or \
            len(events) == max_melody_events:

                orig_melodies.append(events)
                events = []
                # print('yes')
            else:
                events.append(MelodyEvent(lst[i].pitch,
                                        lst[i].end - lst[i].start,
                                        lst[i+1].start - lst[i].end,
                                        lst[i].velocity))

            if i == len(lst)-2 and (lst[i+1].start - lst[i].end <= gap_step):
                events.append(MelodyEvent(lst[i].pitch,
                                        lst[i].end - lst[i].start,
                                        lst[i+1].start - lst[i].end,
                                        lst[i].velocity))

                orig_melodies.append(events)

    for mel in orig_melodies:
        melody = MelodySequence(mel)
        if len(melody) < min_melody_events:
            continue

        # Require a certain number of unique pitches.
        note_histogram = melody.get_note_histogram()
        unique_pitches = np.count_nonzero(note_histogram)
        if unique_pitches < min_unique_pitches:
            continue

        stats['melody_lengths_in_events'].increment(len(melody))

        stats['total_length_in_steps'].increment(melody.total_steps)
        stats['total_length_in_pitch'].increment(len(melody))

        melodies.append(melody)

    return melodies, list(stats.values())

def extract_melodies_for_info(quantized_sequence,
                            gap_step = 32,
                            min_unique_pitches=4,
                            max_melody_events=MAX_EVENT_LENGTH,
                            min_melody_events=9,
                            filter_drums=True):
    '''return a list of MelodySequence'''

    sequences_lib.assert_is_quantized_sequence(quantized_sequence)

    melodies = []
    stats = dict([(stat_name, statistics.Counter(stat_name)) for stat_name in
                ['melodies_discarded_too_short',
                 'melodies_discarded_too_few_pitches',
                 'melodies_discarded_too_repeated',
                 'melodies_discarded_too_repeated_long']])

    stats['melody_lengths_in_steps'] = statistics.Histogram(
        'melody_lengths_in_steps', [a*32 for a in
        [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, 7 // 2, 7,
        7 + 1, 7 - 1]])

    stats['melody_lengths_in_events'] = statistics.Histogram(
        'melody_lengths_in_events', [9, 20, 30, 40, 50, 100, 200, 500, 512, 513, 514])

    stats['pitch_dis'] = statistics.Histogram(
        'pitch_dis', list(range(129)))
    stats['duration_dis'] = statistics.Histogram(
        'duration_dis', list(range(66)))
    stats['silence_dis'] = statistics.Histogram(
        'silence_dis', list(range(34)))
    stats['velocity_dis'] = statistics.Histogram(
        'velocity_dis', VELOCITY_VALUE)

    instruments = set([n.instrument for n in quantized_sequence.notes])

    orig_melodies = []
    for instrument in instruments:
        lst = _get_notes_tuple(quantized_sequence,
                                instrument = instrument,
                                filter_drums = True)
        lst = make_relative_velocity(lst)
        events = []
        for i in range(len(lst) - 1):

            if lst[i+1].start - lst[i].end > gap_step or \
            len(events) == max_melody_events:

                orig_melodies.append(events)
                events = []
                # print('yes')
            else:
                events.append(MelodyEvent(lst[i].pitch,
                                        lst[i].end - lst[i].start,
                                        lst[i+1].start - lst[i].end,
                                        lst[i].velocity))

            if i == len(lst)-2 and (lst[i+1].start - lst[i].end <= gap_step):
                events.append(MelodyEvent(lst[i].pitch,
                                        lst[i].end - lst[i].start,
                                        lst[i+1].start - lst[i].end,
                                        lst[i].velocity))

                orig_melodies.append(events)

    for mel in orig_melodies:
        melody = MelodySequence(mel)
        if len(melody) < min_melody_events:
            stats['melodies_discarded_too_short'].increment()
            continue



        # Require a certain number of unique pitches.
        note_histogram = melody.get_note_histogram()
        unique_pitches = np.count_nonzero(note_histogram)
        if unique_pitches < min_unique_pitches:
            stats['melodies_discarded_too_few_pitches'].increment()
            continue
        if discard_repeated(mel):
            stats['melodies_discarded_too_repeated'].increment()
            continue

        if discard_repeated_long(mel):
            stats['melodies_discarded_too_repeated_long'].increment()
            continue

        stats['melody_lengths_in_steps'].increment(melody.total_steps)
        stats['melody_lengths_in_events'].increment(len(melody))
        for event in melody:
            stats['pitch_dis'].increment(event.pitch)
            stats['duration_dis'].increment(event.duration)
            stats['silence_dis'].increment(event.silence)
            stats['velocity_dis'].increment(event.velocity)

        melodies.append(melody)

    return melodies, list(stats.values())

def make_relative_velocity(tuple_lst):

    new_lst = []
    max_vel = max(map(lambda a: a.velocity, tuple_lst))
    min_vel = min(map(lambda a: a.velocity, tuple_lst))
    ori_range = max_vel - min_vel
    correct_range = VELOCITY_VALUE[-1] - VELOCITY_VALUE[0]

    if ori_range == 0:
        new_lst = [Data(tu.pitch, tu.start, tu.end, VELOCITY_VALUE[-2])
                    for tu in new_lst]
    else:
        for tu in tuple_lst:
            vel_idx = round(((tu.velocity - min_vel)/ ori_range) * correct_range / 8)
            vel = VELOCITY_VALUE[vel_idx]
            new_tu = Data(tu.pitch,
                        tu.start,
                        tu.end,
                        vel)
            new_lst.append(new_tu)
    return new_lst

def discard_repeated(melody):
    i = 0
    leng = len(melody) - 4
    count = 0
    while True:
        if count == 3:
            return True
        if i >= leng:
            break
        if melody[i].pitch == melody[i+1].pitch:
            if melody[i+1].pitch == melody[i+2].pitch:
                if melody[i+2].pitch == melody[i+3].pitch:
                    if melody[i+2].pitch == melody[i+3].pitch:
                        count += 1
                    i+=4
                else:
                    i+=3
            else:
                i+=2
        else:
            i+=1
    return False

def discard_repeated_long(melody):
    i = 0
    leng = len(melody) - 3
    num = (len(melody) // 4) // 2
    count = 0
    while True:
        if count == num:
            return True
        if i >= leng:
            break
        if melody[i].duration >= 48:
            if melody[i+1].duration >= 48:
                if melody[i+2].duration >= 48:
                    if melody[i+3].duration >= 48:
                        count += 1
                    i+=4
                else:
                    i+=3
            else:
                i+=2
        else:
            i+=1
    return False
