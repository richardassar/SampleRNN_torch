# SampleRNN_torch

A Torch implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://openreview.net/forum?id=SkxKPDv5xl).

![A visual representation of the SampleRNN architecture](http://deepsound.io/images/samplernn.png)

## Samples

Sample output can be heard [here](https://soundcloud.com/psylent-v/sets/samplernn_torch). Feel free to submit links to any interesting samples you generate as a pull request.

## Dependencies

The following packages are required to run SampleRNN_torch:

- nn
- cunn
- cudnn
- rnn
- optim
- audio
- xlua
- gnuplot

## Datasets

To retrieve and prepare the *piano* dataset, as used in the reference implementation, run:

```
cd datasets/piano/
./create_piano_dataset.sh
```

The violin dataset preparation scripts are located in `datasets/violin/`.

Custom datasets may be created by using `scripts/generate_dataset.lua` to slice multiple audio files into segments for training, audio must be placed in `datasets/[dataset]/data/`.

## Training

To start a training session run `th train.lua -dataset piano`. To view a description of all accepted arguments run `th train.lua -help`.

To view the progress of training run `th generate_plots`, the loss and gradient norm curve will be saved in `sessions/[session]/plots/`.

## Sampling

By default samples are generated at the end of every training epoch but they can also be generated separately using `th train.lua -generate_samples` with the `session` parameter to specify the model.

Multiple samples are generated in batch mode for efficiency, however generating a single audio sample is faster with `th fast_sample.lua`. See `-help` for a description of the arguments.

## Models

A pretrained model of the *piano* dataset is available [here](https://drive.google.com/uc?id=0B5pXFO5X-KJ9Mko3MUZuLUpEQVU&export=download). Download and copy it into your `sessions/` directory and then extract it in place.

More models will be uploaded soon.

## Theano version

This code is based on the reference implementation in Theano by [Soroush Mehri](https://github.com/soroushmehr).

https://github.com/soroushmehr/sampleRNN_ICLR2017
