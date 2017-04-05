# sampleRNN_torch
A Torch implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://openreview.net/forum?id=SkxKPDv5xl).

![A visual representation of the SampleRNN architecture](http://deepsound.io/images/samplernn.png)

## Samples

Sample output can be heard here:

- TODO: soundcloud links

Feel free to submit links to any interesting samples you generate as a pull request.

## Dependencies
The following packages are required to run sampleRNN_torch:

- nn
- cunn
- cudnn
- rnn
- optim
- audio
- xlua
- image *(optional)*
- gnuplot *(optional)*

## Datasets
To retrieve and prepare the *piano* dataset, as used in the reference implementation, run:

```
cd datasets/piano/
./create_piano_dataset.sh
```

Custom datasets may be created by using `scripts/generate_dataset.lua` to slice multiple audio files into segments for training, audio must be placed in `datasets/[dataset]/data/`.

## Training
To start a training session run `th train.lua -dataset piano`. Default arguments are used, to view a description of all accepted arguments run `th train.lua -help`.

## Sampling
By default samples are generated at the end of every training epoch but they can also be generated separately using `th train.lua -generate_samples` with the `session` parameter to specify the model.

Multiple samples are generated in batch mode for efficiency, however generating a single audio sample is faster with `th fast_sample.lua`. See `-help` for a description of the arguments.

## Models
A pretrained model of the *piano* dataset is available [here](TODO: upload model) and can be fetched by running:

```
TODO: download command
```

## Acknowledgements
Special thanks to [Soroush Mehri](https://github.com/soroushmehr) for his discussion and support during the development of this project.