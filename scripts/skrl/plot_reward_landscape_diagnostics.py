"""Render paired reward geometry with explicit scale and checkpoint controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from plot_reward_landscape_comparison import centered_surface

METHODS = ['ppo', 'c2rl_ppo']
REWARDS = ['PPO reward (I)', 'C2RL reward (M)']
COLORS = ['#2463a5', '#d56c24']


def geometry_statistics(z):
    first, second = z[..., 0].ravel(), z[..., 1].ravel()
    difference = second - first
    return dict(ppo_rms=float(np.sqrt(np.mean(first**2))),
        c2rl_rms=float(np.sqrt(np.mean(second**2))),
        difference_rms=float(np.sqrt(np.mean(difference**2))),
        correlation=float(np.corrcoef(first, second)[0,1]) if min(first.std(),second.std()) > 1e-14 else None)


def plane_ascent(cut, scales):
    axis=cut['axis']
    hi=np.flatnonzero(axis>1e-10)[0]
    lo=int(np.argmin(np.abs(axis+axis[hi])))
    np.testing.assert_allclose(axis[lo],-axis[hi],atol=1e-10)
    return (cut['returns'][hi,...,:2]-cut['returns'][lo,...,:2]).mean(1)/(2*axis[hi]*scales[:2]),float(axis[hi])


def render(source, curvature_source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source / 'summary.json').read_text())
    scales = np.array([summary['fixed_initial_return_rms'][k] for k in summary['objectives']])
    surfaces, cuts, completion, hashes, statistics = {}, {}, {}, {}, []
    def archive(path):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            return {k:data[k].copy() for k in data.files}
    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(output / f'{name}.{ext}', dpi=150, bbox_inches='tight')
        plt.close(fig)
    for row in summary['rows']:
        if not row.get('surface'):
            continue
        folder = source / row['path']
        grid = archive(folder / 'surface.npz')
        if (folder / 'cuts.npz').exists():
            cut = archive(folder / 'cuts.npz')
        else:
            curvature = archive(folder / 'curvature.npz')
            mid = len(grid['axis'])//2
            cut = dict(axis=grid['axis'],center_returns=curvature['center_returns'],
                returns=np.stack([grid['returns'][:,mid],grid['returns'][mid,:]],axis=1))
        z = centered_surface(grid['returns'], cut['center_returns'], scales)
        np.testing.assert_allclose(z[len(z)//2,len(z)//2],0.,atol=1e-7)
        surfaces[row['step'],row['method']] = (grid['axis'],z)
        cuts[row['step'],row['method']] = cut
        completion[row['step'],row['method']] = (grid['axis'],grid['completed'].mean(-1))
    # The initial center is identical; mirror its data for the second training path.
    for mapping in [surfaces,cuts,completion]:
        if (0,'ppo') in mapping and (0,'c2rl_ppo') not in mapping:
            mapping[0,'c2rl_ppo'] = mapping[0,'ppo']
    steps = sorted({step for step,_ in surfaces})
    radii = summary.get('radii',[float(next(iter(surfaces.values()))[0].max())])
    plane = 'Training-displacement plane' if 'plane' in summary else 'Fixed random-direction plane'
    if len(radii)>1:
        fig,axes=plt.subplots(2,len(radii),figsize=(11.5,6.5),squeeze=False,layout='constrained')
        all_axis,all_z=surfaces[0,'ppo']
        for col,radius in enumerate(radii):
            keep=np.abs(all_axis)<=radius+1e-10
            axis,z=all_axis[keep],all_z[keep][:,keep]
            limit=max(float(np.abs(z).max()),1e-12)
            for objective in range(2):
                ax=axes[objective,col]
                image=ax.pcolormesh(axis,axis,z[...,objective].T,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest')
                ax.contour(axis,axis,z[...,objective].T,levels=np.linspace(-limit,limit,9),colors='k',linewidths=.45,alpha=.45)
                ax.plot(0,0,'k+',ms=7)
                ax.set(title=f'Radius {radius:g}; {len(axis)}×{len(axis)}',xlabel='a',aspect='equal',xlim=(-radius,radius),ylim=(-radius,radius))
                if col==0:ax.set_ylabel(REWARDS[objective]+'\nb')
            fig.colorbar(image,ax=axes[:,col],shrink=.7,format='%.2g',label='Centered return / initial RMS')
        fig.suptitle('Increase the domain at the shared initial policy — same training-displacement plane\nBoth rewards evaluated at every identical point. Spacing stays fixed; color scale shared within each column.',fontsize=12)
        save(fig,'domain_comparison')
    for radius in radii:
        selected = {}
        for key,(axis,z) in surfaces.items():
            keep = np.abs(axis) <= radius+1e-10
            selected[key] = (axis[keep],z[keep][:,keep])
            statistics.append(dict(step=key[0],method=key[1],radius=radius,**geometry_statistics(z[keep][:,keep])))
        for method in METHODS:
            fig,axes = plt.subplots(2,len(steps),figsize=(14,6.3),squeeze=False,layout='constrained')
            for col,step in enumerate(steps):
                if (step,method) not in selected:
                    continue
                axis,z = selected[step,method]
                limit = max(float(np.abs(z).max()),1e-12)
                norm = TwoSlopeNorm(vmin=-limit,vcenter=0.,vmax=limit)
                ascent,_=plane_ascent(cuts[step,method],scales)
                for objective in range(2):
                    ax=axes[objective,col]
                    image=ax.pcolormesh(axis,axis,z[...,objective].T,cmap='RdBu_r',norm=norm,shading='nearest')
                    if np.ptp(z[...,objective]) > 1e-12:
                        ax.contour(axis,axis,z[...,objective].T,levels=np.linspace(-limit,limit,9),colors='k',linewidths=.45,alpha=.45)
                    ax.plot(0,0,'k+',ms=7)
                    vector=ascent[:,objective]
                    if np.linalg.norm(vector)>1e-14:
                        vector=vector/np.linalg.norm(vector)*radius*.28
                        ax.annotate('',xy=vector,xytext=(0,0),arrowprops=dict(arrowstyle='->',color='#1a9d67',lw=2))
                    ax.set(title=f'{step//1000}k steps; {len(axis)}×{len(axis)}',xlabel='a (parameter L2)',aspect='equal',xlim=(-radius,radius),ylim=(-radius,radius))
                    if col==0: ax.set_ylabel(REWARDS[objective]+'\nb (parameter L2)')
                fig.colorbar(image,ax=axes[:,col],shrink=.72,format='%.2g',label='Centered return / initial RMS')
            fig.suptitle(f'{plane} — {method} checkpoint centers; radius {radius:g}\n'
                         'Only the reward changes within each column. Shared column color scale; green arrows show finite-difference ascent direction.',fontsize=12)
            save(fig,f'contours_{method}_r{radius:02g}')
        fig,axes=plt.subplots(2,len(steps),figsize=(14,7.2),squeeze=False,layout='constrained')
        fig.get_layout_engine().set(hspace=.14)
        for row,method in enumerate(METHODS):
            for col,step in enumerate(steps):
                ax=axes[row,col]
                if (step,method) not in selected: continue
                axis,z=selected[step,method]
                difference=z[...,1]-z[...,0]
                limit=max(float(np.abs(difference).max()),1e-12)
                image=ax.pcolormesh(axis,axis,difference.T,cmap='PuOr_r',vmin=-limit,vmax=limit,shading='nearest')
                ax.plot(0,0,'k+',ms=7)
                stats=geometry_statistics(z)
                ax.set(title=f'{step//1000}k; difference RMS {stats["difference_rms"]:.2g}',xlabel='a',aspect='equal',xlim=(-radius,radius),ylim=(-radius,radius))
                if col==0: ax.set_ylabel(f'{method} centers\nb')
                fig.colorbar(image,ax=ax,shrink=.75,format='%.2g')
        fig.suptitle(f'Reward-shape difference: ΔJ(M) / RMS(M) − ΔJ(I) / RMS(I); radius {radius:g}\n'
                     'Zero means agreement after fixed normalization. Each panel has its own labeled color range.',fontsize=12)
        save(fig,f'difference_r{radius:02g}')
        if radius==5:
            fig,axes=plt.subplots(4,len(steps),figsize=(14,11.5),squeeze=False,layout='constrained')
            for index,method in enumerate(METHODS):
                for col,step in enumerate(steps):
                    if (step,method) not in selected:continue
                    axis,z=selected[step,method]
                    limit=max(float(np.abs(z).max()),1e-12)
                    ascent,_=plane_ascent(cuts[step,method],scales)
                    for objective in range(2):
                        ax=axes[2*index+objective,col]
                        image=ax.pcolormesh(axis,axis,z[...,objective].T,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest')
                        ax.contour(axis,axis,z[...,objective].T,levels=np.linspace(-limit,limit,9),colors='k',linewidths=.45,alpha=.45)
                        ax.plot(0,0,'k+',ms=7)
                        vector=ascent[:,objective]
                        if np.linalg.norm(vector)>1e-14:
                            vector=vector/np.linalg.norm(vector)*radius*.28
                            ax.annotate('',xy=vector,xytext=(0,0),arrowprops=dict(arrowstyle='->',color='#1a9d67',lw=2))
                        ax.set(title=f'{step//1000}k steps',xlabel='a',aspect='equal',xlim=(-radius,radius),ylim=(-radius,radius))
                        if col==0:ax.set_ylabel(f'{method} center\n{REWARDS[objective]}\nb')
                    fig.colorbar(image,ax=axes[2*index:2*index+2,col],shrink=.7,format='%.2g')
            fig.suptitle('Both reward objectives along both training paths — radius 5\n'
                'Compare adjacent I/M rows at the same checkpoint. Each pair shares a color range; ranges differ between centers and times.',fontsize=12)
            save(fig,'contours_both_r05')

    for method in METHODS:
        for radius in [1.,20.]:
            if any(np.count_nonzero(np.abs(data['axis'])<=radius+1e-10)<3 for key,data in cuts.items() if key[1]==method):
                continue
            fig,axes=plt.subplots(2,len(steps),figsize=(14,6.3),squeeze=False,layout='constrained')
            for col,step in enumerate(steps):
                if (step,method) not in cuts: continue
                data=cuts[step,method]
                keep=np.abs(data['axis'])<=radius+1e-10
                for direction in range(2):
                    ax=axes[direction,col]
                    changes=(data['returns'][:,direction,...,:2]-data['center_returns'][None,:,:2])/scales[:2]
                    for objective in range(2):
                        means=changes[...,objective].mean(-1)
                        se=changes[...,objective].std(-1,ddof=1)/np.sqrt(changes.shape[1])
                        ax.plot(data['axis'][keep],means[keep],color=COLORS[objective],label=REWARDS[objective])
                        ax.fill_between(data['axis'][keep],(means-se)[keep],(means+se)[keep],color=COLORS[objective],alpha=.12)
                    ax.axhline(0,color='.6',lw=.6);ax.axvline(0,color='.6',lw=.6)
                    ax.set(title=f'{step//1000}k; direction {direction+1}',xlabel='Parameter displacement',ylabel='Centered return / initial RMS')
                    ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.grid(alpha=.15)
            axes[0,0].legend(fontsize=8)
            fig.suptitle(f'Overlaid reward slices at identical {method} policy centers — radius {radius:g}\n'
                         'Same fixed reward normalization throughout; y-ranges vary by panel. Bands: ±1 scenario standard error.',fontsize=12)
            save(fig,f'cuts_{method}_r{radius:02g}')

    fig,axes=plt.subplots(2,len(steps),figsize=(14,6.2),squeeze=False,layout='constrained')
    for row,method in enumerate(METHODS):
        for col,step in enumerate(steps):
            if (step,method) not in completion:continue
            axis,fraction=completion[step,method]
            ax=axes[row,col]
            image=ax.pcolormesh(axis,axis,fraction.T,vmin=0,vmax=1,cmap='viridis',shading='nearest')
            ax.set(title=f'{step//1000}k; minimum {fraction.min():.0%}',xlabel='a',aspect='equal',xlim=(axis[0],axis[-1]),ylim=(axis[0],axis[-1]))
            if col==0:ax.set_ylabel(f'{method} centers\nb')
    fig.colorbar(image,ax=axes,shrink=.7,label='Fraction completing the full 500-step episode')
    fig.suptitle('Episode-completion control — compare with cliffs in the return surfaces\nFailed episodes are retained; the reward return stops at first episode end.',fontsize=12)
    save(fig,'episode_completion')

    # Raw and normalized profiles at the shared initial center make scaling visible.
    initial=cuts[0,'ppo']; keep=np.abs(initial['axis'])<=5+1e-10
    fig,axes=plt.subplots(2,2,figsize=(11,6.8),layout='constrained')
    for direction in range(2):
        changes=(initial['returns'][:,direction,...,:2]-initial['center_returns'][None,:,:2]).mean(1)
        for objective in range(2):
            axes[direction,0].plot(initial['axis'][keep],changes[keep,objective],color=COLORS[objective],label=REWARDS[objective])
            axes[direction,1].plot(initial['axis'][keep],changes[keep,objective]/scales[objective],color=COLORS[objective],label=REWARDS[objective])
        for ax,title in zip(axes[direction],['Raw reward-return changes','After fixed initial-RMS normalization']):
            ax.set(title=f'Direction {direction+1}: {title}',xlabel='Parameter displacement',ylabel='Change in return');ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f'Same initial policy and same perturbations; initial RMS ratio M/I = {scales[1]/scales[0]:.1f}')
    save(fig,'raw_vs_normalized')

    curvature_summary=json.loads((curvature_source/'summary.json').read_text())
    np.testing.assert_allclose(scales,[curvature_summary['fixed_initial_return_rms'][k] for k in summary['objectives']],rtol=1e-12)
    records=[]
    for row in curvature_summary['rows']:
        data=archive(curvature_source/row['path']/'curvature.npz')
        h=data['h']
        expected=np.stack([(data['returns'][i,:,0]-2*data['center_returns']+data['returns'][i,:,1])/float(v)**2 for i,v in enumerate(h)])
        np.testing.assert_array_equal(expected,data['curvature'])
        for i,value in enumerate(h):
            signed=expected[i,...,:2].mean(axis=1)/scales[:2]
            records.append(dict(step=row['step'],method=row['method'],h=float(value),values=np.abs(signed)))
    hs=sorted({r['h'] for r in records})
    for value in hs:
        times=sorted({r['step'] for r in records})
        fig,axes=plt.subplots(2,len(times),figsize=(max(12,2.3*len(times)),6.2),squeeze=False,layout='constrained')
        for row,method in enumerate(METHODS):
            for col,step in enumerate(times):
                ax=axes[row,col]
                selected=[r for r in records if r['method']==method and r['step']==step and r['h']==value]
                if not selected: ax.axis('off');continue
                values=selected[0]['values']
                box=ax.boxplot([values[:,0],values[:,1]],tick_labels=['I','M'],patch_artist=True,
                    showfliers=False,widths=.5,medianprops={'color':'black'})
                for patch,color in zip(box['boxes'],COLORS):patch.set_facecolor(color);patch.set_alpha(.4)
                for point in values:
                    ax.plot([1,2],np.maximum(point,1e-16),color='.7',lw=.4,alpha=.5)
                ratio=float(np.median(values[:,1])/np.median(values[:,0]))
                ax.set(title=f'{step//1000}k; M/I={ratio:.2g}',yscale='log');ax.grid(axis='y',alpha=.15)
                if col==0:ax.set_ylabel(f'{method} centers\n|directional curvature| / initial RMS')
        fig.suptitle(f'Change only the reward at each fixed policy center — h={value:g}\n'
                     '32 common random directions per checkpoint; lines pair the same direction. Lower means less curvature at this h.',fontsize=12)
        save(fig,f'curvature_h{value:g}')
    with (output/'geometry_statistics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(statistics[0]));writer.writeheader();writer.writerows(statistics)
    serial=[dict(step=r['step'],method=r['method'],h=r['h'],directions=len(r['values']),
        median_ppo=float(np.median(r['values'][:,0])),median_c2rl=float(np.median(r['values'][:,1])),
        ratio=float(np.median(r['values'][:,1])/np.median(r['values'][:,0]))) for r in records]
    times=sorted({r['step'] for r in serial})
    fig,axes=plt.subplots(2,1,figsize=(12,5.6),layout='constrained')
    largest=max(abs(np.log2(r['ratio'])) for r in serial)
    for ax,method in zip(axes,METHODS):
        ratios=np.full((len(hs),len(times)),np.nan)
        for r in serial:
            if r['method']==method:ratios[hs.index(r['h']),times.index(r['step'])]=r['ratio']
        image=ax.imshow(np.log2(ratios),cmap='RdBu_r',vmin=-largest,vmax=largest,aspect='auto')
        for (i,j),value in np.ndenumerate(ratios):
            if np.isfinite(value):ax.text(j,i,f'{value:.2f}×',ha='center',va='center',fontsize=11,
                color='white' if abs(np.log2(value))>largest*.65 else 'black')
        ax.set_xticks(range(len(times)),[f'{t//1000}k' for t in times])
        ax.set_yticks(range(len(hs)),[f'h={h:g}' for h in hs]);ax.set_title(f'Identical {method} checkpoint centers')
    fig.colorbar(image,ax=axes,shrink=.8,label='log2(median |curvature M| / median |curvature I|)')
    fig.suptitle('Reward-only curvature ratio after fixed normalization\nBlue: metric reward less curved; red: more curved; 1×: equal. Same 32 directions in each comparison.',fontsize=12)
    save(fig,'curvature_ratios')
    ascents=[]
    for key,cut in cuts.items():
        ascent,h=plane_ascent(cut,scales)
        hi=int(np.argmin(np.abs(cut['axis']-h)));lo=int(np.argmin(np.abs(cut['axis']+h)))
        local_curvature=(cut['returns'][hi,...,:2]+cut['returns'][lo,...,:2]-2*cut['center_returns'][None,:,:2]).mean(1)/(h*h*scales[:2])
        norms=np.linalg.norm(ascent,axis=0)
        cosine=float(np.dot(ascent[:,0],ascent[:,1])/np.prod(norms)) if np.prod(norms)>1e-24 else None
        ascents.append(dict(step=key[0],method=key[1],h=h,normalized_ascent=ascent.tolist(),
            normalized_curvature=local_curvature.tolist(),
            angle_degrees=float(np.degrees(np.arccos(np.clip(cosine,-1.,1.)))) if cosine is not None else None))
    metadata=dict(source=str(source.resolve()),curvature_source=str(curvature_source.resolve()),
        plane=summary.get('plane',plane),statistics=statistics,curvature=serial,plane_ascent=ascents,
        completion=[dict(step=k[0],method=k[1],minimum=float(v[1].min())) for k,v in completion.items()],
        reward_rms=scales.tolist(),source_sha256=hashes,source_status=summary['status'],curvature_source_status=curvature_summary['status'],
        verification='Finite differences recomputed exactly; centers verified; same parameter offsets and paired scenarios for both rewards.',
        limitations=summary.get('limitations',[]))
    metadata['renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'verification.json').write_text(json.dumps(metadata,indent=2)+'\n')
    baseline=[r for r in statistics if r['step']==0 and r['method']=='ppo']
    ratio_min=min(r['ratio'] for r in serial);ratio_max=max(r['ratio'] for r in serial)
    lines=['# Reward landscape figures and interpretation','',
        'Both objectives are evaluated on identical trajectories at each parameter point. The state is (x, x*, xi), without future or history windows. Each objective is centered at its checkpoint and divided by its own fixed shared-initialization return RMS.','',
        '| Figure | What to look for | Takeaway |','| --- | --- | --- |',
        '| domain_comparison | Compare the two rewards within each radius column; then inspect the broader structure at larger radii. | At shared initialization, paired grid-return correlations are '+', '.join('{correlation:.3f} at radius {radius:g}'.format(**r) for r in baseline if r['correlation'] is not None)+'. Correlation measures agreement of sampled grid values, not convexity. |',
        '| contours_ppo / contours_c2rl_ppo | Compare I and M vertically within one column. Red improves on the central return; blue worsens it. Green arrows show finite-difference ascent orientation, with fixed display length. | The two rewards can differ in slopes and level sets. Comparing different checkpoint centers also includes the effect of training; it does not isolate reward shaping. Read color-bar magnitudes across time. |',
        '| difference | Locate departures from zero and read the panel difference RMS. | Nonzero regions identify changes beyond a fixed scalar rescaling. Orange means a larger normalized return change under M; it does not mean better tracking performance. Each panel has its own labeled range. |',
        '| cuts | Compare slopes, turning points, and bending of I and M on exactly the same one-dimensional path. | Radius-1 cuts resolve local differences; radius-20 cuts expose broad plateaus and transitions. Bands are scenario standard errors, not uncertainty across training seeds. |',
        f'| curvature | Compare paired I/M distributions within one checkpoint and inspect all three h values. | Across the available comparisons, the median normalized M/I ratio ranges from {ratio_min:.3g} to {ratio_max:.3g}. A ratio below one means less finite-difference curvature at that checkpoint and h. |',
        f'| raw_vs_normalized | Compare the raw profiles with the normalized profiles. | The initial return RMS ratio M/I is {scales[1]/scales[0]:.2f}; raw curvature alone includes this reward-scale difference. |',
        '| episode_completion | Compare low-completion locations with cliffs in the return maps. | Episode endings can contribute to landscape structure. Failed scenarios remain in the data; reward returns stop at first episode end. |','',
        '## Curvature ratios at identical parameters','',
        '| Steps | Policy center | h=0.1 | h=0.3 | h=1 |','| --- | --- | --- | --- | --- |']
    for step in times:
        for method in METHODS:
            values=[next((r['ratio'] for r in serial if r['step']==step and r['method']==method and np.isclose(r['h'],h)),None) for h in [.1,.3,1.]]
            if any(v is not None for v in values):
                lines.append(f'| {step:,} | {method} | '+' | '.join(f'{v:.3f}' if v is not None else 'pending' for v in values)+' |')
    lines+=['','## Local ascent directions in the plotted plane','',
        '| Steps | Policy center | Finite-difference h | Angle between I and M ascent directions |','| --- | --- | --- | --- |']
    for row in ascents:
        angle=f"{row['angle_degrees']:.2f} degrees" if row['angle_degrees'] is not None else 'undefined at zero slope'
        lines.append(f"| {row['step']:,} | {row['method']} | {row['h']:.3g} | {angle} |")
    minimum=min(float(v[1].min()) for v in completion.values())
    lines+=['',f'Minimum full-episode completion over the saved grid points: {minimum:.2%}.',
        '',f'Surface source status: {summary["status"]}. Curvature source status: {curvature_summary["status"]}.',
        '', 'With fixed rewards and a frozen metric, training visits different neighborhoods of two fixed landscapes. These slices do not characterize the entire parameter space or establish that a geometry difference causes faster learning. PPO optimizes a clipped surrogate, whereas these figures show rollout-return objectives.','',
        'Raw paired returns, completion masks, input hashes, normalization constants and exact finite-difference verification are retained in verification.json and the source directories.']
    lines+=['','## Performance of the central policies','',
        'Every surface is centered at zero by construction. Zero color does not identify whether the central policy performs well. The common tracking AUC below supplies that context; lower is better.','',
        '| Steps | Policy center | Common tracking AUC |','| --- | --- | --- |']
    for row in summary['rows']:
        if row.get('surface'):
            lines.append(f"| {row['step']:,} | {row['method']} | {-row['center_mean'][2]:.6g} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--curvature-input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.input.resolve(),args.curvature_input.resolve(),args.output.resolve())
