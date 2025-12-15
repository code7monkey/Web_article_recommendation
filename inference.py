"""Entry point for generating predictions using the NewsRec model.

This script loads a YAML configuration file and runs the inference
pipeline defined in ``src.trainer``. It produces a submission file
compatible with the format expected by the target competition.

Usage:

```
python inference.py --config configs/submit.yaml
```
"""

import argparse
import yaml

from src.trainer import inference_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the NewsRec model")
    parser.add_argument(
        '--config', type=str, required=True, help='Path to a YAML configuration file',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    inference_pipeline(config)


if __name__ == '__main__':
    main()
