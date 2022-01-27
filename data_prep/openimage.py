import fiftyone
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='datasets/OpenImage',
                    help="directory to save openimage dataset")
args = parser.parse_args()

dataset = fiftyone.zoo.datasets.download_zoo_dataset(
              name="open-images-v6",
              split="validation",
              dataset_dir=args.save_dir,
              overwrite=False,
              max_samples=100
          )