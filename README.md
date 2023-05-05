# Randomized BPE for machine translation

## Install required packages

### External dependencies
1. Install [`xsv`](https://github.com/BurntSushi/xsv)
2. Install [`GNU parallel`](https://www.gnu.org/software/parallel/)
3. Optionally install `fzf`, `jq` and `bat`, too.

### Python dependencies
`pip install -r requirements.txt`

## Download the data
- [English - Finnish](https://drive.google.com/file/d/1J7uX5TQ2ivMowLWFmZrYrtJ47DeEWG2Q/view?usp=sharing)
- [English - German](https://drive.google.com/file/d/1BxaHJGkJ4vRFuhPno3DMtcVWBI4aC8bh/view?usp=sharing)
- [English - Estonian](https://drive.google.com/file/d/1Z9azC-FGJABmxTo29P46BhwNsCbaRu9P/view?usp=sharing)

For Uzbek, use `scripts/download_til.py`.

After extracting, create a detokenized version of all of `{train,dev,test}.{eng,fin,deu,uzb}` using `sacremoses`:

```bash
sacremoses detokenize < input_file > output_file
```
## Run experiments

The general workflow to run an experiment is the same regardless of language/segmentation method.
Here is an example for English - Finnish translation using regular BPE and 32k merge operations.

0. Create an `experiments` and `eng_{fin,deu,est,uzb}_bin` directories in the root folder.

```bash
mkdir experiments eng_{fin,deu,est,uzb}_bin
```

1. Set the `randseg_experiment_name` and environment variable. 

```bash
export randseg_experiment_name=english2finnish_vanillabpe
```

2. Set variables for the experiment config file (`randsge_cfg_file`) and hyperparameter folder (`randseg_hparams_folder`)

```bash
export randseg_cfg_file=$(realpath config/english2finnish_sweep_vanillabpe_cfg.sh)
export randseg_hparams_folder=$(realpath config/sweep_confitions_32k_1worker)
```

You can also customize your config file if you're running a custom experiment. See `./config/english2*_cfg.sh` for inspiration.

3. Run the experiment using SLURM with 10 parallel jobs, one for each seed

```bash
sbatch -J your_job_name sweep_experiment.sh
```

4. Analyze results

After the experiments finish, the scores can be found in `test.eval.score.{bleu,chrf}` in each experiment folder.
