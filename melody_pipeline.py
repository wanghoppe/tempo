from melody_lib import MelodySequence, extract_melodies, extract_melodies_for_info
from magenta.protobuf import music_pb2
from magenta.pipelines import pipeline
from magenta.pipelines import statistics

class MelodyExtractor(pipeline.Pipeline):
    def __init__(self,
                min_unique_pitches=5,
                max_melody_events=512,
                min_melody_events=9,
                filter_drums=True,
                name=None):

        super(MelodyExtractor, self).__init__(
                input_type=music_pb2.NoteSequence,
                output_type=MelodySequence,
                name=name)

        self._min_unique_pitches = min_unique_pitches
        self._max_melody_events = max_melody_events
        self._min_melody_events = min_melody_events
        self._filter_drums = filter_drums

    def transform(self, quantized_sequence):
        try:
            melodies, stats = extract_melodies(
                        quantized_sequence,
                        min_unique_pitches=self._min_melody_events,
                        max_melody_events=self._max_melody_events,
                        min_melody_events=self._min_melody_events,
                        filter_drums=self._filter_drums)
        except Exception as e:
            print('Skipped sequence:', str(e))
            melodies = []
            stats = [statistics.Counter('unknow_error', 1)]
        self._set_stats(stats)
        return melodies


class MelodyExtractorInfo(pipeline.Pipeline):
    '''used to get the whole dataset info, not used in training'''
    def __init__(self,
                min_unique_pitches=5,
                max_melody_events=512,
                min_melody_events=9,
                filter_drums=True,
                name=None):

        super(MelodyExtractorInfo, self).__init__(
                input_type=music_pb2.NoteSequence,
                output_type=MelodySequence,
                name=name)

        self._min_unique_pitches = min_unique_pitches
        self._max_melody_events = max_melody_events
        self._min_melody_events = min_melody_events
        self._filter_drums = filter_drums

    def transform(self, quantized_sequence):
        try:
            melodies, stats = extract_melodies_for_info(
                        quantized_sequence,
                        min_unique_pitches=self._min_melody_events,
                        max_melody_events=self._max_melody_events,
                        min_melody_events=self._min_melody_events,
                        filter_drums=self._filter_drums)
        except Exception as e:
            print('Skipped sequence:', str(e))
            melodies = []
            stats = [statistics.Counter('unknow_error', 1)]
        self._set_stats(stats)
        return melodies
