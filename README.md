# MLP-Mixer for Multi-Channel 1D Data 
In this project we implemented MLP-Mixer [1] from scratch for application to multi-channel, 1-dimensional (1D), time series data. We test this implementation on an open dataset of audio files, different patching methods, and optimize hyperparameters for each. 

## Phoneme Dataset 
The Phoneme Dataset [2] contains 6,668 examples of audio waveforms representing phonemes in the English language, collected through Google Translate Audio. Speakers were recorded saying complete words and phonemes were extracted from each audio file by the Penn Phonetics Lab Forced Aligner [3]. A spectrogram with 11 frequency bands was generated from each phoneme instance. There are 39 unique phonemes represented in the dataset. We additionally preprocess this dataset by filling in missing values with zeros and padding all data points with zeros to length 220 from their original (padded) length of 217. 

The machine learning task is to classify the phoneme spectrograms into the correct class. The data are provided from [2] split into training (n = 3,315 examples) and held-out testing sets (n = 3,353). We further split the testing set into truly held-out (n = 1,676 examples) and validation/hyperparameter tuning (n = 1,677) sets. This additional split allows us to tune model performance on the validation set with low risk of reporting inflated performance on the held-out testing set.

## Patching Methods  
The original MLP Mixer paper [1] was made for processing images. As such, the patching worked as the figure to the left shows, with blocks of pixels becoming the patch that is embedded and passed to the model. We wanted to use the idea of image patching on 1D time series data, namely the phoneme spectrograms described in the section above, as input to the MLP mixer. To do this we came up with three 1D patching scenarios, shown below. 

The sequential patching method (a) simply sliced the input time series into patches of size patch_size, where slicing happens along the time axis and time steps are not ‘shuffled’ between patches, except between patch edges. The random patching method (b) shuffles the time series and then patches it sequentially, so that indices are shuffled randomly amongst each other. The cyclical patching method (c) ‘samples’ from the time series cyclically, i.e. every patch_size, and puts these indices beside each other, so that for a patch size of 3 and array length 10 the cyclical index array would look like [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8], as shown above.

## References 
[1] Tolstikhin, Ilya O., et al. "Mlp-mixer: An all-mlp architecture for vision." Advances in neural information processing systems 34 (2021): 24261-24272.
[2] Hamooni, Hossein, and Abdullah Mueen. "Dual-domain hierarchical classification of phonetic time series." 2014 IEEE international conference on data mining. IEEE, 2014.
[3] Jiahong Yuan and Mark Liberman. 2008. Speaker identification on the SCOTUS corpus. Proceedings of Acoustics '08
[4] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
