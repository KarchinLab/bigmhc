# BigMHC

BigMHC is a deep learning tool for predicting MHC-I (neo)epitope presentation and immunogenicity.

See the article for more information:
* [Nature Machine Intelligence article](https://www.nature.com/articles/s42256-023-00694-6)
* [free NMI link](https://rdcu.be/dhkOY)

All data used in this research can be freely downloaded [here](https://doi.org/10.17632/dvmz6pkzvb).

## Installation

```
pip install git+https://github.com/griffithlab/bigmhc.git
```

This will install of the required dependencies.

### Environment and Dependencies

Execution is OS agnostic and does not require GPUs.

Training models with large batch sizes (e.g. 32768) requires significant GPU memory (about 94 GB total). Transfer learning requires minimal GPU memory and can be reasonably conducted on a CPU.

All methods were tested on Debian 11 using Linux 5.10.0-19-amd64, AMD EPYC 7443P, and four RTX 3090 GPUs.

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

There are two commandline entrypoints:  `bigmhc_predict` and `bigmhc_train`.

* `bigmhc_predict` is used for making predictions using BigMHC EL and BigMHC IM
* `bigmhc_train` allows you to train or retrain (transfer learning) BigMHC on new data

Both scripts, which can be run from any directory, offer help text.
* `bigmhc_predict --help`
* `bigmhc_train --help`

#### Examples

```
bigmhc_predict -i=../data/example1.csv -m=el -t=2 -d="cpu"
bigmhc_predict -i=../data/example2.csv -m=el -a=HLA-A*02:02 -p=0 -c=0 -d="cpu"
```

Predictions will be written to `example1.csv.prd` and `example2.csv.prd` in the data folder. Execution takes a few seconds. Compare your output with `example1.csv.cmp` and `example2.csv.cmp` respectively.

#### Supported Alleles

BigMHC only supports MHC-I. In order to handle different MHC naming schemes, BigMHC will perform fuzzy string matching to find the nearest MHC by name. For example, `HLA-A*02:01`, `A*02:01`, `HLAA0201`, and `A0201` are all considered valid and equivalent allele names. Additionally, synonymous substitutions and noncoding fields are handled, so `HLA-A*02:01:01` should be mapped to `HLA-A*02:01`.

We do not validate allele names. BigMHC will make predictions even if given nonsense or MHC-II input, as it will find the nearest valid MHC name to the provided invalid allele name. The list of alleles used in our multiple sequence alignment, to which input is mapped, can be found in the [pseudosequences data file](data/pseudoseqs.csv).

#### Required Arguments
* `-i` or `--input` input CSV file
  * Columns are zero-indexed
  * Must have a column of peptides
  * Can also have a column of of MHC-I allele names
* `-m` or `--model` BigMHC model to load
  * `el` or `bigmhc_el` to load BigMHC EL
  * `im` or `bigmhc_im` to load BigMHC IM
  * Can be a path to a BigMHC model directory
  * Optional for `bigmhc_train` (if a model dir is specified, then transfer learn)

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
  * If using `bigmhc_predict`, save CSV data to this file
    * Defaults to `input`.prd
  * If using `bigmhc_train`, save the retrained BigMHC model to this directory
    * If transfer learning, defaults to the base model dir
* `-z` or `--saveatt` boolean indicating whether to save attention values
  * Only available for `bigmhc_predict`
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
  * If using `bigmhc_predict`, defaults to a value that is estimated to fully occupy the device with the least memory
  * If using `bigmhc_train`, defaults to `32`
* `-s` or `--pseudoseqs` CSV file mapping MHC to one-hot encoding
* `-l` or `--lr` AdamW optimizer learning rate
  * Only available for `bigmhc_train`
* `-e` or `--epochs` number of epochs for transfer learning
  * Only available for `bigmhc_train`

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

## License

See the [LICENSE](LICENSE) file
