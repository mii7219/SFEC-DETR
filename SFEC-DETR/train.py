"""Train M7 with a dataset in YOLO detection format."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Dataset YAML')
    parser.add_argument('--model', default=str(Path(__file__).with_name('M7.yaml')))
    parser.add_argument('--weights', help='Optional local initialization checkpoint')
    parser.add_argument('--resume', help='Resume from a local last.pt checkpoint')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default=None, help='e.g. 0, 0,1, or cpu')
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--lr0', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project', default='runs/train')
    parser.add_argument('--name', default='M7')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    if args.resume and args.weights:
        parser.error('--resume and --weights cannot be combined')
    for filename in (args.data, args.model if not args.resume else args.resume, args.weights):
        if filename and not Path(filename).is_file():
            parser.error(f'File does not exist: {filename}')

    from ultralytics import RTDETR

    model = RTDETR(args.resume or args.model)
    if args.weights:
        model.load(args.weights)
    model.train(
        data=str(Path(args.data).resolve()),
        resume=bool(args.resume),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        cos_lr=True,
        patience=0,
        amp=False,
        seed=args.seed,
        cache=False,
        project=args.project,
        name=args.name,
        plots=args.plots,
    )


if __name__ == '__main__':
    main()
