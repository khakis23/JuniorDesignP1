import argparse
from pathlib import Path
from util.SavePaths import SavePaths


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sp", "--save_path", type=Path, help="save path for plots and results")  # save path
    parser.add_argument("-sr", "--save_results", action="store_true", help="boolean flag for saving results.txt")  # save results
    return parser.parse_args()


def parse_args_save_paths():
    args = _parse_args()

    if args.save_path:
        args.save_path.mkdir(parents=True, exist_ok=True)
        SavePaths.save_path = args.save_path
    else:
        if args.save_results:
            print("Please provide a save path for saving results.")
        return

    sr_text = ""
    if args.save_results:
        SavePaths.save_results = args.save_results
        sr_text = "and model evals "

    print(f"Saving plots {sr_text}to {SavePaths.save_path}")
