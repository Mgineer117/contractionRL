"""Compare the two actual reward objectives at identical policy parameters."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = ['PPO reward: Euclidean energy decrease', 'C2RL reward: metric energy decrease']


def centered_surface(returns, center_returns, scales):
    return (returns.mean(axis=2)[..., :2] - center_returns.mean(axis=0)[:2]) / scales[:2]


def render(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source / 'summary.json').read_text())
    scales = np.array([summary['fixed_initial_return_rms'][name] for name in summary['objectives']])
    records, surfaces, hashes = [], {}, {}
    for row in summary['rows']:
        folder = source / row['path']
        path = folder / 'curvature.npz'
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            hs = data['h']
            expected = np.stack([(data['returns'][i, :, 0] - 2*data['center_returns']
                                  + data['returns'][i, :, 1])/h**2 for i, h in enumerate(hs)])
            np.testing.assert_array_equal(expected, data['curvature'])
            absolute = np.abs(expected.mean(axis=2))
            for i, h in enumerate(hs):
                raw = np.median(absolute[i, :, :2], axis=0)
                scaled = raw / scales[:2]
                records.append(dict(step=row['step'], policy_center=row['method'], h=float(h),
                    directions=int(expected.shape[1]), ppo_raw=float(raw[0]), c2rl_raw=float(raw[1]),
                    ppo_normalized=float(scaled[0]), c2rl_normalized=float(scaled[1]),
                    c2rl_over_ppo=float(scaled[1]/scaled[0]) if scaled[0] else None))
            if row['surface']:
                surface_path = folder / 'surface.npz'
                hashes[str(surface_path.resolve())] = hashlib.sha256(surface_path.read_bytes()).hexdigest()
                with np.load(surface_path) as grid:
                    z = centered_surface(grid['returns'], data['center_returns'], scales)
                    axis = grid['axis']
                    np.testing.assert_allclose(z[len(axis)//2, len(axis)//2], 0., atol=1e-7)
                    surfaces[row['step'], row['method']] = (axis.copy(), z)

    for step in sorted({key[0] for key in surfaces}):
        centers = [method for method in ['ppo', 'c2rl_ppo'] if (step, method) in surfaces]
        if step == 0:
            # Shared initialization: show this parameter center once.
            if len(centers) == 2:
                np.testing.assert_allclose(surfaces[step, centers[0]][1], surfaces[step, centers[1]][1], atol=1e-7)
            centers = centers[:1]
        limit = max(float(np.abs(surfaces[step, method][1]).max()) for method in centers)
        limit = max(limit, 1e-12)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit)
        fig = plt.figure(figsize=(11, 4.8*len(centers)), layout='constrained')
        for index, method in enumerate(centers):
            axis, z = surfaces[step, method]
            a, b = np.meshgrid(axis, axis, indexing='ij')
            for objective in range(2):
                ax = fig.add_subplot(len(centers), 2, 2*index+objective+1, projection='3d')
                ax.plot_surface(a, b, z[..., objective], cmap='coolwarm', norm=norm, linewidth=.2)
                center_label = 'Shared initial policy' if step == 0 else f'Same {method} policy center'
                ax.set(title=f'{LABELS[objective]}\n{center_label}', xlabel='a (parameter L2)',
                       ylabel='b (parameter L2)', zlabel='Change in return / initial RMS', zlim=(-limit, limit))
        fig.suptitle(f'Reward objectives at identical policy parameters — step {step:,}\n'
                    f'{len(axis)}×{len(axis)} grid; common vertical/color scale within this figure', fontsize=12)
        for extension in ['png', 'pdf']:
                fig.savefig(output / f'reward_comparison_{step:06d}.{extension}', dpi=170,
                            bbox_inches='tight', pad_inches=.3)
        plt.close(fig)

    centers = [method for method in ['ppo', 'c2rl_ppo'] if any(r['policy_center'] == method for r in records)]
    fig, axes = plt.subplots(len(centers), len(hs), figsize=(12, 3.7*len(centers)),
                             squeeze=False, layout='constrained')
    for index, method in enumerate(centers):
        for column, h in enumerate(hs):
            selected = sorted([r for r in records if r['policy_center'] == method and r['h'] == h], key=lambda r:r['step'])
            for key, label, color in [('ppo_normalized', 'PPO reward', '#3c6eb4'),
                                      ('c2rl_normalized', 'C2RL reward', '#ce633b')]:
                axes[index, column].plot([r['step']/1000 for r in selected], [r[key] for r in selected], 'o-', color=color, label=label)
            axes[index, column].set(title=f'Same {method} policy centers; h={h:g}', xlabel='Training steps (thousands)',
                                    ylabel='Median |curvature| / initial RMS', yscale='log')
            axes[index, column].grid(alpha=.2)
            axes[index, column].legend(fontsize=8)
    fig.suptitle('Change only the reward definition at each fixed policy center', fontsize=13)
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'reward_curvature_comparison.{extension}', dpi=170)
    plt.close(fig)
    with (output/'curvature_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    metadata = dict(source=str(source.resolve()), objectives=LABELS, rows=records,
        scales=summary['fixed_initial_return_rms'], preview=summary.get('preview', False),
        source_sha256=hashes, verification='Finite differences recomputed exactly from saved paired returns.',
        interpretation='Compare columns/curves at identical policy parameters, scenarios, actions and perturbations; only the reward changes.',
        surface_centering='Subtract each objective return at the central policy, then divide by its fixed shared-initialization RMS.',
        figure_scaling='Common vertical/color scale within each checkpoint figure; scales can differ between checkpoint figures.',
        limitations=summary.get('limitations', []))
    (output/'comparison.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.input.resolve(), args.output.resolve())
