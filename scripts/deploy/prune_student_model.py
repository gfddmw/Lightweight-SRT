#!/usr/bin/env python
import argparse
from collections import OrderedDict
import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.student_model.architecture.st_gcn import Model
from src.student_model.architecture.utils.tgcn import ConvTemporalGraphical


def _clean_state_dict(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith('module.'):
        return OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
    return state_dict


def _pick_output_bn(block):
    if hasattr(block, 'use_bottleneck') and block.use_bottleneck:
        return block.bn_expand
    return block.tcn[3]


def _compute_keep_indices(model, prune_ratio, min_channels):
    all_gamma = []
    block_gamma = []
    for block in model.st_gcn_networks:
        bn = _pick_output_bn(block)
        gamma = bn.weight.detach().abs().cpu()
        block_gamma.append(gamma)
        all_gamma.append(gamma)

    if not all_gamma:
        raise RuntimeError('No prunable BN layers found.')

    merged = torch.cat(all_gamma)
    threshold = torch.quantile(merged, prune_ratio)

    keep_indices = []
    new_channels = []
    for gamma in block_gamma:
        idx = torch.where(gamma > threshold)[0]
        if idx.numel() < min_channels:
            idx = torch.argsort(gamma, descending=True)[:min_channels]
        idx = idx.sort()[0]
        keep_indices.append(idx)
        new_channels.append(int(idx.numel()))

    return keep_indices, new_channels, float(threshold.item())


def _copy_bn(dst_bn, src_bn, idx=None):
    if idx is None:
        c = min(dst_bn.weight.numel(), src_bn.weight.numel())
        idx = torch.arange(c)
    dst_bn.weight.data.copy_(src_bn.weight.data[idx].clone())
    dst_bn.bias.data.copy_(src_bn.bias.data[idx].clone())
    dst_bn.running_mean.data.copy_(src_bn.running_mean.data[idx].clone())
    dst_bn.running_var.data.copy_(src_bn.running_var.data[idx].clone())


def _copy_conv2d(dst_conv, src_conv, out_idx=None, in_idx=None):
    src_w = src_conv.weight.data

    if out_idx is None:
        out_idx = torch.arange(min(dst_conv.weight.size(0), src_w.size(0)))
    if in_idx is None:
        in_idx = torch.arange(min(dst_conv.weight.size(1), src_w.size(1)))

    w = src_w[out_idx][:, in_idx].clone()
    dst_conv.weight.data.copy_(w)

    if dst_conv.bias is not None and src_conv.bias is not None:
        dst_conv.bias.data.copy_(src_conv.bias.data[out_idx].clone())


def _expand_gcn_out_idx(base_idx, old_out_channels, kernel_size):
    parts = []
    for k in range(kernel_size):
        parts.append(base_idx + k * old_out_channels)
    return torch.cat(parts, dim=0)


def _expand_shift_in_idx(base_idx, old_in_channels, kernel_size):
    parts = []
    for k in range(kernel_size):
        parts.append(base_idx + k * old_in_channels)
    return torch.cat(parts, dim=0)


def _transfer_block_weights(dst_block, src_block, in_idx, out_idx):
    # GCN branch
    if isinstance(src_block.gcn, ConvTemporalGraphical) and isinstance(dst_block.gcn, ConvTemporalGraphical):
        old_out = src_block.gcn.conv.weight.size(0) // src_block.gcn.kernel_size
        src_out_idx = _expand_gcn_out_idx(out_idx, old_out, src_block.gcn.kernel_size)
        _copy_conv2d(dst_block.gcn.conv, src_block.gcn.conv, out_idx=src_out_idx, in_idx=in_idx)
    else:
        old_in = src_block.gcn.proj.weight.size(1) // src_block.gcn.kernel_size
        expanded_in_idx = _expand_shift_in_idx(in_idx, old_in, src_block.gcn.kernel_size)
        _copy_conv2d(dst_block.gcn.proj, src_block.gcn.proj, out_idx=out_idx, in_idx=expanded_in_idx)

    # TCN/Bottleneck branch
    if src_block.use_bottleneck and dst_block.use_bottleneck:
        _copy_conv2d(dst_block.bottleneck_reduce, src_block.bottleneck_reduce, in_idx=out_idx)
        _copy_bn(dst_block.bn_reduce, src_block.bn_reduce)
        _copy_conv2d(dst_block.bottleneck_t, src_block.bottleneck_t)
        _copy_bn(dst_block.bn_t, src_block.bn_t)
        _copy_conv2d(dst_block.bottleneck_v, src_block.bottleneck_v)
        _copy_bn(dst_block.bn_v, src_block.bn_v)
        _copy_conv2d(dst_block.bottleneck_expand, src_block.bottleneck_expand, out_idx=out_idx)
        _copy_bn(dst_block.bn_expand, src_block.bn_expand, idx=out_idx)
    else:
        _copy_bn(dst_block.tcn[0], src_block.tcn[0], idx=out_idx)
        _copy_conv2d(dst_block.tcn[2], src_block.tcn[2], out_idx=out_idx, in_idx=out_idx)
        _copy_bn(dst_block.tcn[3], src_block.tcn[3], idx=out_idx)

    # Residual branch
    if isinstance(dst_block.residual, nn.Sequential):
        if isinstance(src_block.residual, nn.Sequential):
            _copy_conv2d(dst_block.residual[0], src_block.residual[0], out_idx=out_idx, in_idx=in_idx)
            _copy_bn(dst_block.residual[1], src_block.residual[1], idx=out_idx)
        else:
            # Source was nn.Identity, but destination is Conv because of channel mismatch.
            # We must initialize it as a "pruned identity" matrix to avoid random noise.
            dst_block.residual[0].weight.data.zero_()
            if dst_block.residual[0].bias is not None:
                dst_block.residual[0].bias.data.zero_()
            
            # Map kept indices between layers
            in_idx_list = in_idx.tolist()
            out_idx_list = out_idx.tolist()
            in_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(in_idx_list)}
            
            count = 0
            for new_out_idx, orig_idx in enumerate(out_idx_list):
                if orig_idx in in_map:
                    new_in_idx = in_map[orig_idx]
                    dst_block.residual[0].weight.data[new_out_idx, new_in_idx, 0, 0] = 1.0
                    count += 1
            
            # Initialize BN as identity
            dst_block.residual[1].weight.data.fill_(1.0)
            dst_block.residual[1].bias.data.zero_()
            dst_block.residual[1].running_mean.data.zero_()
            dst_block.residual[1].running_var.data.fill_(1.0)


def build_pruned_model(
    base_model,
    prune_ratio=0.15,
    min_channels=16,
    in_channels=3,
    num_class=2000,
    layout='openpose',
    strategy='spatial',
):
    keep_indices, channel_cfg, threshold = _compute_keep_indices(base_model, prune_ratio, min_channels)

    model_args = {
        'in_channels': in_channels,
        'num_class': num_class,
        'graph_args': {
            'layout': layout,
            'strategy': strategy,
        },
        'edge_importance_weighting': isinstance(base_model.edge_importance, nn.ParameterList),
        'channel_cfg': channel_cfg,
    }

    first_block = base_model.st_gcn_networks[0]
    model_args['use_shift_gcn'] = hasattr(first_block.gcn, 'proj')
    model_args['use_bottleneck'] = any(b.use_bottleneck for b in base_model.st_gcn_networks)
    model_args['bottleneck_layers'] = [i + 1 for i, b in enumerate(base_model.st_gcn_networks) if b.use_bottleneck]

    pruned_model = Model(**model_args)

    # Copy global layers
    _copy_bn(pruned_model.data_bn, base_model.data_bn)
    
    # Copy edge importance
    if isinstance(base_model.edge_importance, nn.ParameterList):
        for dst_ei, src_ei in zip(pruned_model.edge_importance, base_model.edge_importance):
            dst_ei.data.copy_(src_ei.data)

    prev_idx = torch.arange(in_channels)
    for i, (dst_block, src_block) in enumerate(zip(pruned_model.st_gcn_networks, base_model.st_gcn_networks)):
        out_idx = keep_indices[i]
        _transfer_block_weights(dst_block, src_block, prev_idx, out_idx)
        prev_idx = out_idx

    _copy_conv2d(pruned_model.fcn, base_model.fcn, in_idx=prev_idx)

    return pruned_model, channel_cfg, threshold


def parse_args():
    parser = argparse.ArgumentParser(description='Structured channel pruning for student ST-GCN')
    parser.add_argument('--weights', required=True, help='Path to trained student checkpoint (.pt)')
    parser.add_argument('--output', required=True, help='Path to save pruned checkpoint (.pt)')
    parser.add_argument('--prune_ratio', type=float, default=0.15, help='Global BN gamma prune ratio')
    parser.add_argument('--min_channels', type=int, default=16, help='Minimum channels per block after pruning')
    parser.add_argument('--num_class', type=int, default=2000)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--layout', default='openpose')
    parser.add_argument('--strategy', default='spatial')
    parser.add_argument('--use_shift_gcn', action='store_true')
    parser.add_argument('--use_bottleneck', action='store_true')
    parser.add_argument('--bottleneck_layers', nargs='*', type=int, default=[8, 9, 10])
    return parser.parse_args()


def main():
    args = parse_args()

    base_model = Model(
        in_channels=3,
        num_class=args.num_class,
        graph_args={'layout': args.layout, 'strategy': args.strategy},
        edge_importance_weighting=True,
        dropout=args.dropout,
        use_shift_gcn=args.use_shift_gcn,
        use_bottleneck=args.use_bottleneck,
        bottleneck_layers=args.bottleneck_layers,
    )

    ckpt = torch.load(args.weights, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = _clean_state_dict(state_dict)
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()

    pruned_model, channel_cfg, threshold = build_pruned_model(
        base_model,
        prune_ratio=args.prune_ratio,
        min_channels=args.min_channels,
        in_channels=3,
        num_class=args.num_class,
        layout=args.layout,
        strategy=args.strategy,
    )

    payload = {
        'state_dict': pruned_model.state_dict(),
        'channel_cfg': channel_cfg,
        'prune_ratio': args.prune_ratio,
        'threshold': threshold,
        'note': 'Load with Model(..., channel_cfg=channel_cfg) before finetune.',
    }
    torch.save(payload, args.output)

    print('Pruning done.')
    print('channel_cfg = {}'.format(channel_cfg))
    print('threshold = {:.6f}'.format(threshold))
    print('saved to {}'.format(args.output))


if __name__ == '__main__':
    main()
