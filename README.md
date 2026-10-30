# BigMHC

BigMHC is a deep learning tool for predicting MHC-I (neo)epitope presentation and immunogenicity.

See the article for more information:
* [Nature Machine Intelligence article](https://www.nature.com/articles/s42256-023-00694-6)
* [free NMI link](https://rdcu.be/dhkOY)

All data used in this research can be freely downloaded [here](https://doi.org/10.17632/dvmz6pkzvb).

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

#### Supported Alleles

BigMHC only supports MHC-I. The alleles it can encode are the 18,929 entries of the [pseudosequences data file](data/pseudoseqs.csv) (HLA-A, -B, -C, -E, -F, -G and several non-human species). Allele names are compared after normalization, which upper-cases the name and removes every character that is not a letter or digit, so `HLA-A*02:01`, `hla-a*02:01`, `HLA-A02:01` and `HLAA0201` all denote the entry `HLA-A*02:01`.

If the normalized name is not in the file, BigMHC does not stop: it substitutes the nearest entry *by name* and predicts for that allele instead. The rules are applied in this order and the first one that yields an entry wins:

1. **HLA prefix.** A name made of one letter followed by digits (`A0201`, `A*02:01`) gets `HLA` prepended; the search then continues with the next rule rather than with an exact lookup.
2. **Truncation.** Characters are removed from the right end of the name one at a time until some entry starts with what is left; the first such entry in file order is used. This maps `HLA-A*02:01:01` to `HLA-A*02:01`, but it also maps an allele that is absent from the file to a neighbour sharing its leading characters, e.g. `HLA-A*02:1040` to `HLA-A*02:104`.
3. **Edit distance.** If no prefix matches, the entry with the smallest Levenshtein distance to the name is used. This always yields some MHC-I allele, also for MHC-II names or nonsense (`HLA-DRB1*01:01` becomes `HLA-B*13:101`, `XYZ123` becomes `HLA-A*01:23`); the match is by spelling only and implies no biological similarity.

Because rule 1 is followed by truncation and not by an exact lookup, a short name without `HLA-` can resolve to a *different* allele than the one meant: `A0201` and `A*02:01` do reach `HLA-A*02:01`, but `A*02:02` resolves to `HLA-A*02:01` (the first entry starting with `HLAA020`) and `B*27:05` to `HLA-B*27:01`, although `HLA-A*02:02` and `HLA-B*27:05` are in the file. This is a known issue; the matching rules have been left unchanged so far. Use full names such as `HLA-A*02:02` to avoid it, and check the conversion report described next.

BigMHC does not validate allele names, but every conversion is reported:

* After the input is read, a summary is printed to stderr (also with `-v=0`): one line per distinct input name that was converted, with the allele actually used, the rule(s) applied and the number of input rows concerned. Names written exactly as in the file are not reported. Edit-distance matches are marked `WARNING`, since they are the ones most likely to be wrong.
* `predict.py` adds the column `mhc_used` after all other output columns: for every row, the name of the allele whose pseudosequence was actually used. The `mhc` column keeps the name exactly as given, so the two can be compared row by row.

For example, with an input file containing

```
mhc,pep
HLA-A*02:02,AAADIVNFL
A*02:02,AAADIVNFL
hla-a*02:02,AAADIVNFL
HLA-DRB1*01:01,AAADIVNFL
```

`python predict.py -i=example.csv -m=el -d="cpu"` prints to stderr

```
MHC name conversions: 3 of 4 distinct input allele names were converted to a name in the pseudosequence file; predictions carry the allele used for each row in the mhc_used column
MHC name conversion: input='A*02:02' used='HLA-A*02:01' reason='HLA prefix + truncation (x=1)' rows=1
MHC name conversion: input='hla-a*02:02' used='HLA-A*02:02' reason='normalization' rows=1
MHC name conversion WARNING: input='HLA-DRB1*01:01' used='HLA-B*13:101' reason='edit distance (d=3)' rows=1 -- the allele used only has the most similar name; this does not imply any biological similarity to the input, so check that the input is a supported MHC-I allele name
```

and writes

```
mhc,pep,tgt,len,BigMHC_EL,mhc_used
HLA-A*02:02,AAADIVNFL,,9,0.8622514,HLA-A*02:02
A*02:02,AAADIVNFL,,9,0.4617879,HLA-A*02:01
hla-a*02:02,AAADIVNFL,,9,0.8622514,HLA-A*02:02
HLA-DRB1*01:01,AAADIVNFL,,9,0.06513106,HLA-B*13:101
```

The second row was scored for `HLA-A*02:01`, not for `HLA-A*02:02` as its `mhc` column might suggest; `mhc_used` makes this visible. Conversions do not change the predictions: for any input, the allele selected and the score are the same as in earlier versions that applied these rules silently.

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

#### Tests

From the repository root, `python -m unittest discover -s tests` runs the test suite (the end-to-end tests load the models and take about a minute; they are skipped if the models are absent).

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
