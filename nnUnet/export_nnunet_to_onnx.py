"""
Export a trained nnUNet model (PlainConvUNet) to ONNX format.

Only the neural network is exported — preprocessing (resampling, normalisation) and
sliding window inference are handled separately by run_inference_onnx.py.

Requires the contrast_agnostic conda environment (nnunetv2 + torch).

Usage:
    conda activate contrast_agnostic
    python export_nnunet_to_onnx.py \
        --model-folder /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --output nnunet_seg.onnx
"""

import argparse

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-folder', required=True,
                        help='Path to nnUNetTrainer__nnUNetPlans__3d_fullres folder')
    parser.add_argument('--output', default='nnunet_seg.onnx',
                        help='Output ONNX file path (default: nnunet_seg.onnx)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold to export (default: 0)')
    return parser.parse_args()


def main():
    args = parse_args()

    predictor = nnUNetPredictor(
        device=torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        use_folds=[args.fold],
        checkpoint_name='checkpoint_final.pth',
    )

    net = predictor.network
    net.eval()

    patch_size = predictor.configuration_manager.patch_size
    n_channels = 1  # inferred from first conv weight
    dummy = torch.randn(1, n_channels, *patch_size)

    with torch.no_grad():
        out = net(dummy)
    print(f'Network: {type(net).__name__}')
    print(f'Input  : (1, {n_channels}, {patch_size[0]}, {patch_size[1]}, {patch_size[2]})')
    print(f'Output : {tuple(out.shape)}')

    torch.onnx.export(
        net,
        dummy,
        args.output,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    )

    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'Exported → {args.output}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
