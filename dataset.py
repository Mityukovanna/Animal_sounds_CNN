from torch.utils.data import Dataset
import pandas as pd
import torch
import torchaudio
import os
import numpy as np

"""
The dataset is one-hot encoded. The labels.csv file created by running the 'data_preprocessing' file
has structure:
---------------------------------------------------------------------------------
| filename | class_ Cow | class_ Dog | class_ Frog | class_ Pig | class_ Rooster|
---------------------------------------------------------------------------------
All files in this dataset have same length --> same sample rate --> NO NEED IN RESAMPLING
The dataset class is shown below. 

Link to the tutorials I used:
https://www.youtube.com/watch?v=88FFnqt5MNI&list=LL (needed to adjust it to the one-hot encoding)
https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
https://www.youtube.com/watch?v=lhF_RVa7DLE&t=3s mel-spectrogram transform
"""
class AnimalSoundsDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
        # read the .csv file
        self.annotations = pd.read_csv(annotations_file)

        # get directory to the sound files
        self.audio_dir = audio_dir

        # transformation can be any transformation used with torchaudio.
        # in our case it is mel-spectrogram transformation
        self.transformation = transformation

        # sample rate we are trying to achieve
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        # size of the dataset
        return len(self.annotations)
    
    def __getitem__(self, index):
        # path to the audio file
        audio_sample_path = self._get_audio_sample_path(index)

        # check if file exists before loading
        if not os.path.exists(audio_sample_path):
            raise FileNotFoundError(f"Audio file not found: {audio_sample_path}")

        # returns label (in our case it will be one-hot encoded table)
        label = self._get_audio_sample_label(index)
        # label is extracted from the table
        label = torch.tensor(label, dtype=torch.long) 
        # info about the signal and sample rate(sr)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr) # resampling if sample rate differes from the desired one
        signal = self._mix_down_if_necessary(signal) # mixes down to improve audio quality
        signal = self.transformation(signal) # equivalent of doing mel_spectrogram(signal)
        return signal, label
    
    def _resample_if_necessary(self, signal, sr):
        # link to tutorial: https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html
        # check if existing sample rate matches the desired one
        if sr != self.target_sample_rate:
            # no match --> resample
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        # returning the signal    
        return signal
    
    def _mix_down_if_necessary(self, signal):
        # mixing down the tensor
        # e.g. signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000) [btw it is tensors]
        # not necessary if only have one channel of the signal (mono signal)
        if signal.shape[0] > 1: # more than one channel e.g (2, 1000)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        # use pandas .iloc[] method to get name of the file
        # explanation of .iloc[]: https://www.geeksforgeeks.org/python-extracting-rows-using-pandas-iloc/

        # getting the name of the file by walking through the .csv file
        file_name = self.annotations.iloc[index, 0] 

        # path to the .wav audio file:
        path = os.path.join(self.audio_dir, file_name)

        # Ensure file exists
        if not os.path.exists(path):
            print(f"Warning: File {file_name} is missing at {path}")
        
        return path
    
    def _get_audio_sample_label(self, index):
        # .iloc[] function to get the labels for the audio sample
        # .astype(np.float32) is used to convert numbers to array and effectively store it in memory
        # explanation of .astype(np.float32): https://www.programiz.com/python-programming/numpy/methods/astype
        one_hot_label = self.annotations.iloc[index, 1:].values.astype(np.float32)
        return np.argmax(one_hot_label)  # Convert one-hot to class index

# main function (basically to check if the code is working)
def main():
    ANNOTATIONS_FILE = r"C:\Users\admin\Desktop\python projects\neural_zoo\labels.csv"  
    AUDIO_DIR = r"C:\Users\admin\Desktop\python projects\neural_zoo\Wav_files" 
    SAMPLE_RATE = 16000

    # dataset transforms (building a mel_spectrogram)
    # parameters of teh spectrogram explained: https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels
    # one more link to understand mel spectrograms: https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056#:~:text=Each%20of%20these%20values%20in,second%20there%20are%2022%2C050%20samples.
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024, # frame size ()
        hop_length=512,
        n_mels = 64
    )
    # Ddbug: Check if paths exist
    if not os.path.exists(ANNOTATIONS_FILE):
        raise FileNotFoundError(f"Labels CSV not found at: {ANNOTATIONS_FILE}")

    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError(f"Audio directory not found at: {AUDIO_DIR}")

    # using an asd an example to check if the code is working
    asd = AnimalSoundsDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)
    
    print(f"There are {len(asd)} samples in the dataset.")

    signal, label = asd[0]
    # signal shape represents now mel spectrogram in tensor form
    # signal shape is torch.Size([number_of_channels, number_of_mel_freq_bins, time_frames])
    print(f"Signal shape: {signal.shape}, Label: {label}")

if __name__ == "__main__":
    main()