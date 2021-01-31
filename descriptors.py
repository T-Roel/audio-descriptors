# -*- coding: utf-8 -*-
"""
Demonstration code for Descriptors
Thales Roel - thalesroel@hotmail.com

Usage: 
python3 descriptors.py input1.wav output.pdf   

Example:
python3 descriptors.py bell.wav figname.pdf

Within Spyder, use:
runfile('your_current_dir/descriptors.py', wdir='your_current_dir',
        args='input1.wav output.pdf')
"""

import librosa

def spectral_centroid(audio_file, sr, frame_size, hop_length):
  spectral_centroid = librosa.feature.spectral_centroid(y=audio_file,
                                                        sr=sr,
                                                        n_fft=frame_size,
                                                        hop_length=hop_length)[0]
  
  return spectral_centroid

def spectral_rolloff(audio_file, frame_size, hop_length):
  spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file,
                                                      n_fft=frame_size,
                                                      hop_length=hop_length)[0]
  
  return spectral_rolloff

def frames_to_time(descriptor_param):
  frames = range(len(descriptor_param))
  time = librosa.frames_to_time(frames)

  return time

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    
    title = input("The title of your plot: ")
    color_one = input("The color of the first plot: ")
    color_two = input("The color of the second plot: ")

    # Default parameters
    FRAME_SIZE = 1024
    HOP_LENGTH = 512
    
    print(sys.argv[1], sys.argv[2])
    
    audio, sr = librosa.load(sys.argv[1], sr=44100)

    sc_audio = spectral_centroid(audio, sr, FRAME_SIZE, HOP_LENGTH)
    rolloff_audio = spectral_rolloff(audio, FRAME_SIZE, HOP_LENGTH)

    time = frames_to_time(sc_audio)

    plt.figure(figsize=(25,10))
    plt.subplot(2, 1, 1)
    plt.title(title)
    plt.plot(time, sc_audio , color_one)
    plt.subplot(2, 1, 2)
    plt.plot(time, rolloff_audio , color_two)
    plt.savefig(sys.argv[2])
    plt.show()