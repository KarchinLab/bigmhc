# BigMHC

BigMHC is a deep learning tool for predicting MHC-I (neo)epitope presentation and immunogenicity.

See [the paper](https://doi.org/10.1101/2022.08.29.505690) for more information.

All data used in this research can be freely downloaded [here](https://doi.org/10.17632/dvmz6pkzvb).

## Major Update (January 19, 2023)

Major updates have been made to the models, code, data, and preprint. Please pull the latest versions for using BigMHC and note the revised training procedure.

## Installation

All methods were tested on Debian 11 using Linux 5.10.0-19-amd64, AMD EPYC 7443P, and four RTX 3090 GPUs.

After installing dependencies, install BigMHC by cloning this repository. The repository is about 5GB, so installation takes about 5 minutes depending on internet speed.

#### Required Dependencies

Execution is OS agnostic and does not require GPUs.

Training models with large batch sizes (e.g. 32768) requires significant GPU memory (about 94 GB total). Transfer learning requires minimal GPU memory and can be reasonably conducted on a CPU.

The versions used in the paper are parenthesized.

* [python](https://www.python.org/) (3.9.13)
* [pytorch](https://pytorch.org) (1.13.0)
* [pandas](https://pandas.pydata.org) (1.4.4)

#### Optional Dependencies

* [scikit-learn](https://scikit-learn.org) (1.0.2)
  * Required when using the target column CLI argument
* [cuda](https://developer.nvidia.com/cuda-downloads) (11.7)
  * Required for GPU usage
* [magma](https://developer.nvidia.com/magma) (magma-cuda117 version 2.6.1)
  * Recommended for GPU usage

#### Jupyter Notebook Dependencies

* [numpy](https://numpy.org/) (1.21.5)
* [scipy](https://scipy.org/) (1.7.3)
* [matplotlib](https://matplotlib.org/) (3.5.3)
* [seaborn](https://seaborn.pydata.org/) (0.12.1)
* [py3dmol](https://pypi.org/project/py3Dmol/) (2.0.0.post2)
* [logomaker](https://pypi.org/project/logomaker/) (0.8)

#### Get the BigMHC Source
```
git clone https://github.com/KarchinLab/bigmhc.git
```


## Usage

There are two executable Python scripts in src: `predict.py` and `train.py`.

* `predict.py` is used for making predictions using BigMHC EL and BigMHC IM
* `train.py` allows you to train or retrain (transfer learning) BigMHC on new data

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

#### Required Argument for Training and Retraining
* `-t` or `--tgtcol` column index of target values
  * Optional for `predict.py`
  * If using `predict.py`, this column is used to calculate performance metrics.
  * If using `train.py`, elements in this column are considered ground truth values.
* `-o` or `--out` output directory
  * Directory to save model parameters for each epoch

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
* `-l` or `--lr` AdamW optimizer learning rate
  * Only available for `train.py`
* `-e` or `--epochs` number of epochs for transfer learning
  * Only available for `train.py`

## Contact (Gmail)

benjialbert2

## Citation
```
@article {Albert2022.08.29.505690,
	author = {Albert, Benjamin Alexander and Yang, Yunxiao and Shao, Xiaoshan M. and Singh, Dipika and Smith, Kellie N. and Anagnostou, Valsamo and Karchin, Rachel},
	title = {Deep Neural Networks Predict MHC-I Epitope Presentation and Transfer Learn Neoepitope Immunogenicity},
	elocation-id = {2022.08.29.505690},
	year = {2022},
	doi = {10.1101/2022.08.29.505690},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/08/29/2022.08.29.505690},
	journal = {bioRxiv}
}
```

## License

See the [LICENSE](LICENSE) file
