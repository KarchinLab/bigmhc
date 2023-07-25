# BigMHC

BigMHC is a deep learning tool for predicting MHC-I (neo)epitope presentation and immunogenicity.

See the article for more information:
* [Nature Machine Intelligence paper](https://www.nature.com/articles/s42256-023-00694-6)
* [free NMI link](https://rdcu.be/dhkOY)
* [the preprint](https://doi.org/10.1101/2022.08.29.505690)

All data used in this research can be freely downloaded [here](https://doi.org/10.17632/dvmz6pkzvb).

## Major Update (January 19, 2023)

Major updates have been made to the models, code, data, and preprint. Please pull the latest versions for using BigMHC and note the revised training procedure.

## Installation


### Get the BigMHC Source
```
git clone https://github.com/karchinlab/bigmhc.git
```

The repository is about 5GB, so installation generally takes about 3 minutes depending on internet speed.

### Environment and Dependencies

Execution is OS agnostic and does not require GPUs.

Training models with large batch sizes (e.g. 32768) requires significant GPU memory (about 94 GB total). Transfer learning requires minimal GPU memory and can be reasonably conducted on a CPU.

All methods were tested on Debian 11 using Linux 5.10.0-19-amd64, AMD EPYC 7443P, and four RTX 3090 GPUs.

Software depenencies are listed below (the versions used in the paper are parenthesized).

#### Required Dependencies

* [python](https://www.python.org) (3.9.13)
* [numpy](https://numpy.org) (1.21.5)
* [pytorch](https://pytorch.org) (1.13.0)
* [pandas](https://pandas.pydata.org) (1.4.4)
* [psutil](https://pypi.org/project/psutil) (5.9.4)

#### Optional Dependencies

* [cuda](https://developer.nvidia.com/cuda-downloads) (11.7)
  * Required for GPU usage
* [magma](https://developer.nvidia.com/magma) (magma-cuda117 version 2.6.1)
  * Recommended for GPU usage

#### Jupyter Notebook Dependencies

* [scipy](https://scipy.org/) (1.7.3)
* [scikit-learn](https://scikit-learn.org) (1.0.2)
* [matplotlib](https://matplotlib.org/) (3.5.3)
* [seaborn](https://seaborn.pydata.org/) (0.12.1)
* [py3dmol](https://pypi.org/project/py3Dmol/) (2.0.0.post2)
* [logomaker](https://pypi.org/project/logomaker/) (0.8)
* [openpyxl](https://pypi.org/project/openpyxl) (3.1.1)


## Usage

There are two executable Python scripts in src: `predict.py` and `train.py`.

* `predict.py` is used for making predictions using BigMHC EL and BigMHC IM
* `train.py` allows you to train or retrain (transfer learning) BigMHC on new data

Both scripts, which can be run from any directory, offer help text.
* `python predict.py --help`
* `python train.py --help`

#### Examples

From within the `src` dir, you can execute the below examples:

```
python predict.py -i=../data/example1.csv -m=el -t=2 -d="cpu"
python predict.py -i=../data/example2.csv -m=el -a=HLA-A*02:02 -p=0 -c=0 -d="cpu"
```

Predictions will be written to `example1.csv.prd` and `example2.csv.prd` in the data folder. Execution takes a few seconds. Compare your output with `example1.csv.cmp` and `example2.csv.cmp` respectively.

#### Required Arguments
* `-i` or `--input` input CSV file
  * Columns are zero-indexed
  * Must have a column of peptides
  * Can also have a column of of MHC-I allele names
* `-m` or `--model` BigMHC model to load
  * `el` or `bigmhc_el` to load BigMHC EL
  * `im` or `bigmhc_im` to load BigMHC IM
  * Can be a path to a BigMHC model directory
  * Optional for `train.py` (if a model dir is specified, then transfer learn)

#### Required Arguments for Training
* `-t` or `--tgtcol` column index of target values
  * Elements in this column are considered ground truth values.
* `-o` or `--out` output directory
  * Directory to save model parameters for each epoch
  * Optional for transfer learning (defaults to `model` arg)

#### Input Formatting Arguments
* `-a` or `--allele` allele name or allele column
  * If `allele` is a column index, then a single MHC-I allele name must be present in each row
* `-p` or `--pepcol` peptide column
  * Is the column index of a CSV file containing one peptide sequence per row.
* `-c` or `--hdrcnt` header count
  * Skip the first `hdrcnt` rows before consuming `input`

#### Output Arguments
* `-o` or `--out` output file or directory
  * If using `predict.py`, save CSV data to this file
    * Defaults to `input`.prd
  * If using `train.py`, save the retrained BigMHC model to this directory
    * If transfer learning, defaults to the base model dir
* `-z` or `--saveatt` boolean indicating whether to save attention values
  * Only available for `predict.py`
  * Use `1` for true and `0` for false

#### Other Optional Arguments
* `-d` or `--devices` devices on which to run BigMHC
  * Set to `all` to utilize all GPUs
  * To use a subset of available GPUs, provide a comma-separated list of GPU device indices
  * Set to `cpu` to run on CPU (not recommended for large datasets)
* `-v` or `--verbose` toggle verbose printing
  * Use `1` for true and `0` for false
* `-j` or `--jobs` Number of workers for parallel data loading
  * These workers are persistent throughout the script execution
* `-f` or `--prefetch` Number of batches to prefetch per data loader worker
  * Increasing this number can help prevent GPUs waiting on the CPU, but increases memory usage
* `-b` or `--maxbat` Maximum batch size
  * Turn this down if running out of memory
  * If using `predict.py`, defaults to a value that is estimated to fully occupy the device with the least memory
  * If using `train.py`, defaults to `32`
* `-s` or `--pseudoseqs` CSV file mapping MHC to one-hot encoding
* `-l` or `--lr` AdamW optimizer learning rate
  * Only available for `train.py`
* `-e` or `--epochs` number of epochs for transfer learning
  * Only available for `train.py`

## Contact (Gmail)

benjialbert2

## Citation
```
﻿@Article{Albert2023,
	author={Albert, Benjamin Alexander and Yang, Yunxiao and Shao, Xiaoshan M. and Singh, Dipika and Smith, Kellie N. and Anagnostou, Valsamo and Karchin, Rachel},
	title={Deep neural networks predict class I major histocompatibility complex epitope presentation and transfer learn neoepitope immunogenicity},
	journal={Nature Machine Intelligence},
	year={2023},
	month={Jul},
	day={20},
	issn={2522-5839},
	doi={10.1038/s42256-023-00694-6},
	url={https://doi.org/10.1038/s42256-023-00694-6}
}
```

[![DOI](https://zenodo.org/badge/530254502.svg)](https://zenodo.org/badge/latestdoi/530254502)

## License

See the [LICENSE](LICENSE) file
