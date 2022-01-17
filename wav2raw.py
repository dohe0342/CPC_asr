from scipy.io import wavfile
import os 
import h5py

#trainroot = ['./../LibriSpeech/train-clean-100/', \
#        './../LibriSpeech/train-clean-360/', \
#        './../LibriSpeech/train-other-500/']
#trainroot = ['./../LibriSpeech/train-clean-100/']

#devroot = ['./../LibriSpeech/dev-clean/', './../LibriSpeech/dev-other/']
devroot = ['./../LibriSpeech/dev-clean/']
#testroot = ['./../LibriSpeech/test-clean/']

"""convert wav files to raw wave form and store them in the disc 
"""

# store train 
if 0:
    h5f = h5py.File('train-clean-100.h5', 'w')
    for rootdir in trainroot:
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                #if file.endswith('.wav'):
                if file.endswith('.wav'):
                    fullpath = os.path.join(subdir, file)
                    fs, data = wavfile.read(fullpath)
                    h5f.create_dataset(file[:-4], data=data)
                    print(file[:-4])
    h5f.close()

if 1:
    # store dev 
    h5f = h5py.File('dev-clean.h5', 'w')
    for rootdir in devroot:
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if file.endswith('.wav'):
                    fullpath = os.path.join(subdir, file)
                    fs, data = wavfile.read(fullpath)
                    h5f.create_dataset(file[:-4], data=data)
                    print(file[:-4])
    h5f.close()

if 0:
    # store test
    h5f = h5py.File('test-Librispeech.h5', 'w')
    for rootdir in testroot:
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if file.endswith('.wav'):
                    fullpath = os.path.join(subdir, file)
                    fs, data = wavfile.read(fullpath)
                    h5f.create_dataset(file[:-4], data=data)
                    print(file[:-4])
    h5f.close()

