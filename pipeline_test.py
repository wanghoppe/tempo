

from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.protobuf import music_pb2
from magenta.pipelines import statistics

from melody_pipeline import MelodyExtractorInfo

from magenta.music.midi_io import sequence_proto_to_midi_file

from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.protobuf import music_pb2
from magenta.pipelines import statistics

from melody_pipeline import MelodyExtractorInfo
from melody_lib import STEPS_PER_SECOND
from encoder_decoder import LOWEST_MIDI_PITCH, HIGHEST_MIDI_PITCH
from encoder_decoder import MelodyEncoderDecoder
from melody_lib import MelodySequence
import tensorflow as tf

# Stretch by -10%, 0%, 10%, and 15%.
stretch_factors = [0.9, 1.0, 1.1, 1.15]

transposition_range = range(-3, 4)

class EncoderPipeline(pipeline.Pipeline):
    """A Module that converts monophonic melodies to a model specific encoding."""

    def __init__(self, name):
        """Constructs an EncoderPipeline.

          name: A unique pipeline name.
        """
        super(EncoderPipeline, self).__init__(
                input_type=MelodySequence,
                output_type=tf.train.SequenceExample,
                name=name)
        self._melody_encoder_decoder = MelodyEncoderDecoder()

    def transform(self, melody):

        encoded = self._melody_encoder_decoder.encode(melody)
        return [encoded]

def get_pipeline():

    stretch_pipeline = note_sequence_pipelines.StretchPipeline(
        stretch_factors, name='StretchPipeline')

    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_second=STEPS_PER_SECOND, name='Quantizer')

    transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range, min_pitch=LOWEST_MIDI_PITCH,
        max_pitch=HIGHEST_MIDI_PITCH, name= 'TranspositionPipeline')

    melody_extractor = MelodyExtractorInfo(name='MelodyExtractor')

    encoder_pipeline = EncoderPipeline(name='EncoderPipeline')

    dag ={stretch_pipeline: dag_pipeline.DagInput(music_pb2.NoteSequence)}
    dag[quantizer] = stretch_pipeline
    dag[transposition_pipeline] = quantizer
    dag[melody_extractor] = transposition_pipeline
    dag[encoder_pipeline] = melody_extractor
    dag[dag_pipeline.DagOutput('Output')] = encoder_pipeline

    return dag_pipeline.DAGPipeline(dag)


def main():
    pipeline_instance = get_pipeline()
    input_iterator = pipeline.tf_record_iterator(
                    '/home/hoppe/Code/tempo/tmp/tmp.tfrecord',
                    pipeline_instance.input_type)
    total_output = 0
    stats = []

    for i, input_ in enumerate(input_iterator):
        print(i, end = '\r')
        total_output += len(pipeline_instance.transform(input_)['Output'])
        stats = statistics.merge_statistics(stats + pipeline_instance.get_stats())
        if i%20 == 0:
            with open('test/datainfo.txt', 'w') as f:
                f.write('input melody:' + str(i) + '\n\n')
                f.write('total melody:' + str(total_output) + '\n\n')
                for i in stats:
                    f.write(str(i)+'\n\n')

    with open('test/datainfo.txt', 'w') as f:
        f.write('input melody:' + str(i) + '\n\n')
        f.write('total melody:' + str(total_output) + '\n\n')
        for i in stats:
            f.write(str(i)+'\n\n')

if __name__ == '__main__':
    main()
