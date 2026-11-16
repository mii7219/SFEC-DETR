"""Run detection on images or video with an M7 checkpoint."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default=None)
    parser.add_argument('--project', default='runs/predict')
    parser.add_argument('--name', default='M7')
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        parser.error(f'File does not exist: {args.weights}')

    from ultralytics import RTDETR

    results = RTDETR(args.weights).predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        half=False,
        save=True,
        stream=True,
        project=args.project,
        name=args.name,
    )
    for _ in results:
        pass


if __name__ == '__main__':
    main()
