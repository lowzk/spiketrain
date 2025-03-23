import json
import argparse
from pipeline import get_results

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run experiment with specified configuration files.")
    
    parser.add_argument('-hp', '--hyperparams', type=str, default='config/baseline_dynamic.json',
                        help="Path to the hyperparameters JSON file (default: config/baseline_dynamic.json).")
    parser.add_argument('-sp', '--sysparams', type=str, default='config/system_params.json',
                        help="Path to the system parameters JSON file (default: config/system_params.json).")

    args = parser.parse_args()

    # Load parameters from JSON files
    with open(args.hyperparams) as f:
        baseline_hyperparameters = json.load(f)
    with open(args.sysparams) as f:
        system_params = json.load(f)

    # Run the experiment
    acc, time_taken = get_results(baseline_hyperparameters, system_params)
    print(f"Final accuracy: {acc:.2f}%, Time taken: {time_taken:.2f} seconds")

if __name__ == '__main__':
    main()
