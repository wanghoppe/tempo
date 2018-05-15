import os
import subprocess

def midi2wav(file):
    out_name = file.split('.')[0] + '.wav'
    if os.path.exists(out_name):
        pass
    else:
        lst = [
            'timidity',
            file,
            '-Ow', '-o', out_name
        ]
        subprocess.call(lst)

def main():
    midi_lst = []
    for dirpath, dirnames, filename in os.walk('generate'):
        for file in filename:
            midi_lst.append(os.path.join(dirpath, file))

    for i in midi_lst:
        midi2wav(i)

if __name__ == '__main__':
    main()
