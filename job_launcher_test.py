import os
import datetime
import argparse
import yaml
import itertools

SLURMS = [ 
        "--account biped",
        "--time 479", # 479
        "--nodes 1",
        "--ntasks-per-node 1",
        "--gpus-per-node 1",
        "--partition gpu",
        "--cpus-per-task 10",
        "--mem 60GB",
        "--signal=SIGUSR1@90"
    ]

def create_slurm_file(exp_dir,exp_name, job_name, python_launcher, slurms=SLURMS):

    with open(os.path.join(exp_dir,'slurm_jobs', job_name), "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write(f"#SBATCH --job-name {exp_name}\n"),
        for param in slurms:
            f.write(f"#SBATCH {param}\n")
        f.write("\n")
        f.write("source /idiap/temp/pvuillecard/miniconda3/etc/profile.d/conda.sh\n")
        f.write("conda activate uniface\n")
        f.write(python_launcher)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config
    config_name = config_path.split('/')[-1].split('.')[0]

    root_dir = os.path.dirname(os.path.realpath(__file__))

    # load config 
    with open(os.path.join(root_dir,config_path)) as f:
        config = yaml.safe_load(f)
    
    print(f'Launching jobs {config_name} for the following config:')
    print(config)

    exp_name = config_name
    date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(root_dir,'logs','test', exp_name,f"run_{date_now}")
    slurm_jobs_dir = os.path.join(exp_dir,'slurm_jobs')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(slurm_jobs_dir, exist_ok=True)

    slurm_params = SLURMS
    slurm_params.append(f"--output {os.path.join(exp_dir,'slurm_logs','%j.out')}")

    print(f"Export code to: {exp_dir}")
   
    python_launcher_all = ''
    for k,values in config.items():
    
        # for auto reload with the pytorch ligtning feature we might use srun
        python_launcher_all += f'srun python gaze_module/eval.py ckpt_path={values} \n'
        
    create_slurm_file(exp_dir, exp_name, f"job_test.sh", python_launcher_all,slurm_params)
    job_file = os.path.join(slurm_jobs_dir, f"job_test.sh")
    os.system(f"sbatch {job_file}")

if __name__ == "__main__":
    main()
    
                






