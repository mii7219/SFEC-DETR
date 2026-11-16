"""Evaluate an M7 checkpoint."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default=None)
    parser.add_argument('--project', default='runs/val')
    parser.add_argument('--name', default='M7')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    for filename in (args.weights, args.data):
        if not Path(filename).is_file():
            parser.error(f'File does not exist: {filename}')

    from ultralytics import RTDETR

    RTDETR(args.weights).val(
        data=str(Path(args.data).resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        half=False,
        project=args.project,
        name=args.name,
        plots=args.plots,
    )


if __name__ == '__main__':
    main()
