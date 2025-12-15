"""Entry point for training the NewsRec model.

This script loads a YAML configuration file and runs the training
pipeline defined in ``src.trainer``. It can be executed from the
command line as follows:

```
python train.py --config configs/train.yaml
```
"""

import argparse
import yaml

from src.trainer import train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NewsRec model")
    parser.add_argument(
        '--config', type=str, required=True, help='Path to a YAML configuration file',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train_pipeline(config)


if __name__ == '__main__':
    main()
