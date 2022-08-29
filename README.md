# BigMHC

BigMHC is a deep learning tool for predicting MHC-I (neo)epitope presentation and immunogenicity.

See the paper for more information: TODO

All data used in this research can be freely downloaded here: https://doi.org/10.17632/dvmz6pkzvb.1

## Installation

#### Install Dependencies
* [pytorch](https://pytorch.org)
* [pandas](https://pandas.pydata.org)
* [scikit-learn](https://scikit-learn.org)


#### Get the BigMHC Source
```
git clone https://github.com/KarchinLab/bigmhc.git
```

## Usage

There are two executable Python scripts in src: `predict.py` and `retrain.py`.

* `predict.py` is used for making predictions using BigMHC EL and BigMHC IM
* `retrain.py` allows you to retrain (transfer learning) BigMHC on new data

#### Required Arguments
* `-i` or `--input` input CSV file
  * Columns are zero-indexed
  * Must have a column of peptides
  * Can also have a column of of MHC-I allele names
* `-m` or `--model` BigMHC model to load
  * `el` or `bigmhc_el` to load BigMHC EL
  * `im` or `bigmhc_im` to load BigMHC IM
  * Can be a path to a BigMHC model directory

#### Required Argument for Retraining
* `-t` or `--tgtcol` column index of target values
  * Optional for `predict.py`
  * If using `predict.py`, this column is used to calculate performance metrics.
  * If using `retrain.py`, elements in this column are considered ground truth values.

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
  * If using `retrain.py`, save the retrained BigMHC model to this directory
    * Defaults to creating a new subdir in the `models` dir
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
  * If using `retrain.py`, defaults to `1024`
* `-l` or `--lr` Adam optimizer learning rate
  * Only available for `retrain.py`
* `-e` or `--epochs` number of epochs for transfer learning
  * Only available for `retrain.py`

## Contact (Gmail)

benjialbert2

## Citation

TODO

## License

See the [LICENSE](LICENSE) file
