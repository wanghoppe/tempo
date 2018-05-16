

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

# Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
stretch_factors = [0.95, 0.975, 1.0, 1.025, 1.05]

transposition_range = range(-3, 4)

def get_pipeline():

    stretch_pipeline = note_sequence_pipelines.StretchPipeline(
        stretch_factors, name='StretchPipeline')
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_second=16, name='Quantizer')
#     melody_extractor = MelodyExtractorInfo(name='MelodyExtractorInfo')

    transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range, name= 'TranspositionPipeline')

    dag ={stretch_pipeline: dag_pipeline.DagInput(music_pb2.NoteSequence)}
    dag[quantizer] = stretch_pipeline
    dag[transposition_pipeline] = quantizer
#     dag[melody_extractor] = quantizer
    dag[dag_pipeline.DagOutput('Output')] = transposition_pipeline

    return dag_pipeline.DAGPipeline(dag)


def main():
    pipeline_instance = get_pipeline()
    input_iterator = pipeline.tf_record_iterator(
                    '/media/hoppe/ECCC8857CC881E48/Code/tempo/note_sequence.tfrecord',
                    pipeline_instance.input_type)
    total_output = 0
    stats = []

    for i, input_ in enumerate(input_iterator):
        print(i, end = '\r')
        total_output += len(pipeline_instance.transform(input_)['Output'])
        stats = statistics.merge_statistics(stats + pipeline_instance.get_stats())
        if i%20 == 0:
            with open('test/datainfo2.txt', 'w') as f:
                f.write('input melody:' + str(i) + '\n\n')
                f.write('total melody:' + str(total_output) + '\n\n')
                for i in stats:
                    f.write(str(i)+'\n\n')

    with open('test/datainfo2.txt', 'w') as f:
        f.write('input melody:' + str(i) + '\n\n')
        f.write('total melody:' + str(total_output) + '\n\n')
        for i in stats:
            f.write(str(i)+'\n\n')

if __name__ == '__main__':
    main()
