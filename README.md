# VLLM translation


This is code sample to run translation with VLLM. 

To run, you should log into your cluster:

### Running on a cluster

```bash
ssh leonardo

# install python environment
git clone https://github.com/OpenEuroLLM/multisynt-translate.git
cd multisynt-translate

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init .
uv add -r requirements.txt

# allocate a GPU
srun ...
uv run python vllm_translate_main.py --tgt_lg "French" --tgt_code "fr"
```

### Running on Slurmpilot

There is a script `launch.py` that will generate a slurm sbatch script to 
launch multiple languages in an array. It assumes that you have installed slurmpilot
and have access to Leonardo cluster, you can then just do `python launch.py` to launch the array.

