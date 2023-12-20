python -m sample_factory.launcher.run \
--run=swarm_rl.runs.hybrid.quads_hybrid_search_no_sol_rl_acc \
--backend=slurm --slurm_workdir=slurm_output \
--experiment_suffix=slurm --pause_between=1 \
--slurm_gpus_per_job=1 --slurm_cpus_per_gpu=30 \
--slurm_sbatch_template=/home/zhehui/search/slurm/swarm_rl_sbatch_timeout.sh \
--slurm_print_only=False --slurm_gpus_type=a6000