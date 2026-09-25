import argparse
import subprocess
import sys

def run_cmd(cmd):
    print(f"==> Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Arrhythmia Classification Pipeline Orchestrator")
    
    parser.add_argument('--mode', type=str, choices=['all', 'preprocess', 'train', 'analyze'], default='all')
    parser.add_argument('--task', type=str, choices=['multi', 'binary'], default='multi')
    parser.add_argument('--augmentation', type=str, default='class_weight')
    parser.add_argument('--router', type=str, default=None)
    parser.add_argument('--tau', type=float, default=None)

    args = parser.parse_args()

    if args.router is None:
        args.router = 'rbf_svm' if args.task == 'multi' else 'logistic_regression'
    
    if args.tau is None:
        args.tau = 0.80 if args.task == 'multi' else 0.75

    py_exe = sys.executable

    if args.mode in ['all', 'preprocess']:
        run_cmd([py_exe, "src/backend/preprocessing/main.py", "--augmentation", args.augmentation, "--task", args.task])
        run_cmd([py_exe, "src/backend/pilae/feature_pipeline.py", "--task", args.task])
        
    if args.mode in ['all', 'train']:
        run_cmd([py_exe, "src/backend/classification/classical/train_classical.py", "--task", args.task])
        run_cmd([py_exe, "src/backend/classification/vqc/train_hybrid.py", 
                 "--router", args.router, "--tau", str(args.tau), "--task", args.task])
                 
    if args.mode in ['all', 'analyze']:
        run_cmd([py_exe, "src/backend/classification/feature_analysis.py"])

if __name__ == "__main__":
    main()
