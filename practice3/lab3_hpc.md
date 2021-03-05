# Practice session 3: HPC

### 03.03.2021

In this practice session, we will learn how to use the University of Tartu's High Performance Computing Center servers. Later we will use GPUs there to train our big translation models.

## Login

To connect to the server, use SSH. If you only have a Windows system, you will have to use PuTTY instead.

Log in using your university username and password:

```
ssh your_username@rocket.hpc.ut.ee
```

## Finding your way around

If you are not comfortable with Linux commands, check out, for example, [this guide](https://maker.pro/linux/tutorial/basic-linux-commands-for-beginners).

Once you have logged in, you can create a directory where you will keep all the data for your experiments:

```
mkdir mtcourse
```

Move into the new directory:

```
cd mtcourse
```

Create directories in which to store your data and scripts:

```
mkdir data
mkdir scripts
```

## Fairseq

We will install Fairseq in a Conda virtual environment. By containing all the packages we need in a separate clean environment, we want to avoid version conflicts and [this situation](https://xkcd.com/1987/):

![](https://imgs.xkcd.com/comics/python_environment.png)

For additional information on managing Conda environments, see [this guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

First, load python:

```
module load python/3.6.3
```

Then create a clean environment:

```
conda create -n mtcourse python=3.8
```

Activate the environment:

```
source activate mtcourse
```

Install PyTorch:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
```

In the earlier practice sessions, we installed Fairseq from `pip`.
Now we will install the newest version of Fairseq from source. We will also use
the flag `--editable`. With it, you can change something in Fairseq's code and
continue using it as a package.

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

Install SentencePiece and TensorBoard as well:

```
pip install sentencepiece
pip install tensorboardX
```

After installing everything, reinstall NumPy:

```
pip uninstall numpy
pip install numpy
```

If you need to deactivate the environment:

```
conda deactivate
```

## SLURM

The Rocket cluster uses SLURM (a scheduling system for running jobs). We will cover all the basic things that you will need in this lab, but you can check out [HPC's guide on SLURM](https://hpc.ut.ee/en/guides/slurm/) as well. **DO NOT** execute commands that require considerable resources (preprocessing, model training, etc.) directly on the head node! **ALWAYS USE SLURM,** or your access to HPC may be suspended. You can do some small and quick stuff on the nead node, like managing your virtual environments, installing packages, copying or removing files, etc.

To submit your jobs to SLURM, you will need scripts like this one (you can find this script in `/gpfs/hpc/projects/nlpgroup/mt2021/scripts/01_example_script.sh`):

```
#!/bin/bash

#The name of the job is test_job
#SBATCH -J test_job

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is 5 minutes
#SBATCH -t 00:05:00

#SBATCH --mem=5G

#If you keep the next two lines, you will get an e-mail notification
#whenever something happens to your job (it starts running, completes or fails)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@here.com

#Keep this line if you need a GPU for your job
#SBATCH --partition=gpu

#Indicates that you need one GPU node
#SBATCH --gres=gpu:tesla:1

#Commands to execute go below

#Load Python
module load python/3.6.3/CUDA

#Activate your environment
source activate mtcourse

# Display fairseq's help message
fairseq-train --help
```

Save the scripts that you use for every step of your work. This way, you can always go back and check what you did. Having all your scripts makes it easier to find mistakes.

**Do not** ask for GPU nodes when you perform preprocessing steps (e.g. cleaning, SentencePiece). It will not make them faster and you will block valuable resources. Only use GPUs for training models.

To send your script to the queue:

```
sbatch path/to/your/script.sh
```

You will see output like:

```
Submitted batch job XXX
```

`XXX` will be the ID of your job. If you want to cancel it:

```
scancel XXX
```

Once your job starts running, a file named `slurm-XXX.out` will be created in your current working directory (where you executed `sbatch`). Your log and output will be written into this file.

## Queue

You can view your jobs that are pending or running:

```
squeue -u your_username
```

Or all the GPU jobs on the cluster:

```
squeue -p gpu
```

Or simply all the jobs on the cluster (the list will be very long):

```
squeue
```

## Run a script

**Task.** In `/gpfs/hpc/projects/nlpgroup/mt2021/data/sequence-copy-testing`, you will find files with data similar to the reversed copy task from our first lab.

1. Copy the files to your personal directory.
2. Using a SLURM script, binarize the data (command `fairseq-preprocess`; don't forget to activate the `mtcourse` environment beforehand).
3. Then train a small model on these data. If the GPU queue is busy, don't use a GPU for now. Use the following parameters (note that you need to change the paths to the binarized data and `save-dir` to the actual locations of your data, either absolute or relative):

```
fairseq-train path/to/binarized/data --arch transformer \
                                     --lr 0.005 \
                                     --encoder-attention-heads 2 \
                                     --encoder-embed-dim 8 \
                                     --encoder-layers 1 \
                                     --encoder-ffn-embed-dim 32 \
                                     --decoder-attention-heads 2 \
                                     --decoder-embed-dim 8 \
                                     --decoder-layers 1 \
                                     --decoder-ffn-embed-dim 32 \
                                     --max-epoch 3 \
                                     --optimizer adam \
                                     --max-tokens 5000 \
                                     --save-dir your/checkpoint/directory \
                                     --log-format json \
                                     2>&1 | tee log.out
```

Your goal is not to get a good model, but just to check that training on a GPU works.

4. Make it run and check the contents of your output file to see if everything works correctly.

## Fairseq + configuration files

One useful feature of the newer Fairseq version we've installed in our virtual environments is its [support of training with configuration files](https://github.com/pytorch/fairseq/blob/master/docs/hydra_integration.md). We can still use the command `fairseq-train` and specify all non-default hyperparameters through the command line, as we did before. Another option, though, is to store all the hyperparameters in a `.yaml` file and pass this file to the `fairseq-hydra-train` command (and possibly override some hyperparameters through the command line).

A basic configuration file with default parameters for translation with a Transformer model has been generated for you, you can find it in `/gpfs/hpc/projects/nlpgroup/mt2021/scripts/basic-transformer-config.yaml`.

Or you can get a basic configuration file for translation like this (and fill the model section yourself):

```
fairseq-hydra-train task=translation --cfg job > basic-translation-config.yaml
```

Here is how you can train a model using a config file:

```
fairseq-hydra-train --config-dir . --config-name basic-transformer-config \
                    task.data=/path/to/bin/data dataset.max_tokens=1000
```

