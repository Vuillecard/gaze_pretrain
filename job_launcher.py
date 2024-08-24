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
    exp_dir = os.path.join(root_dir,'logs','experiments', exp_name,f"run_{date_now}")
    slurm_jobs_dir = os.path.join(exp_dir,'slurm_jobs')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(slurm_jobs_dir, exist_ok=True)

    slurm_params = SLURMS
    slurm_params.append(f"--output {os.path.join(exp_dir,'slurm_logs','%j.out')}")

    print(f"Export code to: {exp_dir}")
    # copy the necessary files
    # shutil.copytree(os.path.join(root_dir,'gaze_module'), os.path.join(exp_dir,'gaze_module'),dirs_exist_ok=True)
    # shutil.copytree(os.path.join(root_dir,'configs'), os.path.join(exp_dir,'configs'),dirs_exist_ok=True)
    # shutil.copy(os.path.join(root_dir,'.project-root'), os.path.join(exp_dir,'.project-root'))

    os.system(f"cp -r {os.path.join(root_dir,'gaze_module')} {os.path.join(exp_dir,'gaze_module')}")
    os.system(f"cp -r {os.path.join(root_dir,'configs')} {os.path.join(exp_dir,'configs')}")
    os.system(f"cp {os.path.join(root_dir,'.project-root')} {os.path.join(exp_dir,'.project-root')}")
    print(f'Done copying files')
    
    L = []
    for k,values in config.items():
        L.append([ f"{k}={v}" for v in values])

    combinations = list(itertools.product(*L))

    os.system(f"cd {exp_dir}")
    for i,combination in enumerate(combinations):
        print(f'task #{i} {" ".join(map(str, combination))}')
        #hydra run name
        hydra_run_dir = os.path.join(exp_dir,'logs','train','runs',f"run_{i}")
        if os.path.exists(hydra_run_dir):
            j = 0
            while os.path.exists(hydra_run_dir):
                j += 1
                hydra_run_dir = os.path.join(exp_dir,'logs','train','runs',f"run_{i}_{j}")
        # generate a random id for wandb
        
        # for auto reload with the pytorch ligtning feature we might use srun
        python_launcher = f'srun python {exp_dir}/gaze_module/train.py \
                                {" ".join(map(str, combination))} \
                                ++hydra.run.dir={hydra_run_dir}\
                                ++logger.wandb.id={os.urandom(8).hex()}'
        create_slurm_file(exp_dir, exp_name, f"job_{i}.sh", python_launcher,slurm_params)
        job_file = os.path.join(slurm_jobs_dir, f"job_{i}.sh")
        os.system(f"sbatch {job_file}")

if __name__ == "__main__":
    main()
    
                






