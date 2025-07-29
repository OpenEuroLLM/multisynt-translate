from pathlib import Path

from slurmpilot import JobCreationInfo, SlurmPilot, unify

model_name = "google/gemma-3-4b-it"
jobname = f"translate/{model_name}-nemotron-sample"
cluster = "leonardo"
partition = "boost_usr_prod"

languages = [
    # ("Bulgarian", "bg"),
    # ("Croatian", "hr"),
    # ("Czech", "cs"),
    # ("Danish", "da"),
    # ("Dutch", "nl"),
    # ("Estonian", "et"),
    # ("Finnish", "fi"),
    # ("French", "fr"),
    # ("German", "de"),
    # ("Greek", "el"),
    # ("Hungarian", "hu"),
    # ("Irish", "ga"),
    ("Italian", "it"),
    ("Latvian", "lv"),
    ("Lithuanian", "lt"),
    ("Maltese", "mt"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Romanian", "ro"),
    ("Slovak", "sk"),
    ("Slovenian", "sl"),
    ("Spanish", "es"),
    ("Swedish", "sv")
]

python_args = [
    {"tgt_lg": lg, "tgt_code": lg_code, "model_name": model_name}
    for lg, lg_code in languages
]

job = JobCreationInfo(
    cluster=cluster,
    partition=partition,
    jobname=unify(jobname),
    account="AIFAC_L01_028",
    entrypoint="vllm_translate_main.py",
    src_dir=str(Path(__file__).parent),
    python_binary="/leonardo/home/userexternal/dsalinas/.venv/bin/python",
    python_args=python_args,
    bash_setup_command="ml python; ml cuda",
    n_gpus=4,
    max_runtime_minutes=6 * 60 - 1,
    sbatch_arguments="--exclude lrdn0759",
    env={
        "HF_HUB_OFFLINE": "1",
    },
)
api = SlurmPilot(clusters=[cluster])
api.schedule_job(job_info=job)
