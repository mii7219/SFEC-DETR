"""Check model construction, inference, loss, and backward without a dataset."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=128)
    args = parser.parse_args()
    if args.imgsz < 128 or args.imgsz % 32:
        parser.error('--imgsz must be a multiple of 32 and at least 128')

    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel

    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(0)
    model = RTDETRDetectionModel(str(Path(__file__).with_name('M7.yaml')), verbose=False).to(args.device)
    model.nc = model.yaml['nc']
    images = torch.rand(2, 3, args.imgsz, args.imgsz, device=args.device)
    model.eval()
    with torch.no_grad():
        predictions = model(images)[0]
    assert predictions.shape == (2, 300, 14), predictions.shape
    assert torch.isfinite(predictions).all()
    model.train()
    loss, components = model({
        'img': images,
        'cls': torch.tensor([[0.], [1.]], device=args.device),
        'bboxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.3, 0.3]], device=args.device),
        'batch_idx': torch.tensor([0, 1], device=args.device),
    })
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    for index in (2, 20, 28, 29):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model[index].parameters())
    print(f'PASS: inference {tuple(predictions.shape)}, loss {loss.item():.6f}, finite gradients')


if __name__ == '__main__':
    main()
