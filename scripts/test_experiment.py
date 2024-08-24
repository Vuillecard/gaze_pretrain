
import os 

path_omnivore = {
    'Gaze360V': {
        'path_dir': 'logs/experiments/omnivore/run_2024-07-03_14-25-02',
        'path_model': '/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/omnivore/run_2024-07-03_14-25-02/logs/train/runs/run_0/checkpoints/best_epoch_031.ckpt',
    },
    'Gaze360V GF': {
        'path_dir': 'logs/experiments/generalize/run_2024-07-04_17-42-47',
        'path_model': '/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/generalize/run_2024-07-04_17-42-47/logs/train/runs/run_0/checkpoints/best_epoch_033.ckpt',
    },
    'Gaze360V GF MPS': {
        'path_dir': 'logs/experiments/generalize/run_2024-07-04_17-43-42',
        'path_model': '/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/generalize/run_2024-07-04_17-43-42/logs/train/runs/run_0/checkpoints/best_epoch_018.ckpt',
    },
    'GFIEV': {
        'path_dir': 'logs/experiments/generalize/run_2024-07-04_17-41-31',
        'path_model': '/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/generalize/run_2024-07-04_17-41-31/logs/train/runs/run_0/checkpoints/best_epoch_017.ckpt',
    },
    'GFIEV GF': {
        'path_dir': 'logs/experiments/generalize/run_2024-07-04_23-42-08',
        'path_model': '/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/generalize/run_2024-07-04_23-42-08/logs/train/runs/run_0/checkpoints/last.ckpt',
    },
}

for key, value in path_omnivore.items():
    print(f"Running {key}")
    os.system(f"python gaze_module/eval.py ckpt_path={value['path_model']}")