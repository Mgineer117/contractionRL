"""One idle GPU: integration checks followed by a finite official W&B agent."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,resource
ROOT=Path(__file__).resolve().parent

def save(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main(name):
 plan=json.loads((ROOT/'plan.json').read_text());profile=next(p for p in plan['agents'] if p['name']==name)
 folder=ROOT/'agents'/name;entry=json.loads((folder/'manifest.json').read_text());assert entry['recorded_status'] in {'PLANNED','REQUESTING_IMMEDIATE_ALLOCATION'}
 slurm=os.environ.get('SLURM_JOB_ID')
 revision=subprocess.check_output(['git','-C',str(ROOT/'code'),'rev-parse','HEAD'],text=True).strip()
 assert revision==plan['git_revision'],'Checkout does not match the recorded experiment commit'
 subprocess.run(['git','-C',str(ROOT/'code'),'diff','--exit-code','HEAD','--'],check=True,stdout=subprocess.DEVNULL)
 key=slurm or 'gpu'+str(profile['gpu_index'])
 with (Path('/tmp')/('contractionrl-'+str(os.getuid())+'-'+key+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if not slurm:
   apps=subprocess.check_output(['nvidia-smi','--id='+str(profile['gpu_index']),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   if apps:raise RuntimeError('GPU occupied; existing jobs preserved: '+apps)
  env=dict(os.environ,CRL_SWEEP_MACHINE=profile['machine'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_MODE='online',WANDB_ENTITY=plan['entity'],WANDB_PROJECT=plan['project'],WANDB_DIR=str(folder/'wandb'),WANDB_RESUME='never',PATH=str(Path(profile['python']).parent)+':'+os.environ['PATH'])
  if not slurm:env['CUDA_VISIBLE_DEVICES']=str(profile['gpu_index'])
  env['PYTHONPATH']=str(ROOT/'runtime-packages')+os.pathsep+env.get('PYTHONPATH','')
  for key in list(env):
   if key in ('WANDB_SWEEP_ID','WANDB_RUN_ID') or ('WANDB' in key and 'SERVICE' in key):env.pop(key)
  (folder/'wandb').mkdir(exist_ok=True)
  entry.update(repository=str(ROOT/'code'),configuration=str(ROOT/'plan.json'),paths=[str(folder)],code_revision=revision,recorded_status='PREFLIGHT',job_ids=[slurm or str(os.getpid())],node=os.uname().nodename,cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),started_at=datetime.now(timezone.utc).isoformat())
  save(folder/'manifest.json',entry)
  results={}
  for variant in ['heavy','compact']:
   checkenv=dict(env,WANDB_MODE='disabled',WANDB_SWEEP_ID='smoke',CRL_SMOKE_NAME=name+'-'+variant)
   command=[profile['python'],'-u',str(ROOT/'code/search/train_bayes.py'),'--root',str(ROOT),'--smoke-config',str(ROOT/'preflight'/('proposal-'+variant+'.yaml')),'--smoke-steps','256']
   if variant=='heavy':command+=['--checkpoint',str(ROOT/'metric/checkpoints/metric.pt')]
   with (folder/('preflight-'+variant+'.log')).open('x') as out:
    try:result=subprocess.run(command,cwd=ROOT/'code',env=checkenv,stdout=out,stderr=subprocess.STDOUT,timeout=900)
    except subprocess.TimeoutExpired:raise RuntimeError('Finite GPU check exceeded900seconds; no automatic retry')
   results[variant]=dict(exit_code=result.returncode,command=command,status='PASSED' if result.returncode==0 else 'FAILED')
   save(folder/'preflight.json',results)
   if result.returncode:
    entry.update(recorded_status='FAILED_PREFLIGHT',results=results);save(folder/'manifest.json',entry);return result.returncode
  sweep=json.loads((ROOT/'sweep-receipt.json').read_text())['path']
  command=['timeout','--signal=TERM','--kill-after=90s','14d',profile['python'],'-m','wandb','agent','--forward-signals','--count',str(profile['run_limit']),sweep]
  with (folder/'agent.log').open('x') as out:
   process=subprocess.Popen(command,cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
   state=dict(status='RUNNING',wrapper_pid=os.getpid(),agent_pid=process.pid,slurm_job_id=slurm,command=command,machine=profile['machine'],gpu_index=profile['gpu_index'],node=os.uname().nodename,started_at=datetime.now(timezone.utc).isoformat())
   save(folder/'state.json',state)
   entry.update(recorded_status='RUNNING',agent_pid=process.pid,process_pattern=sweep,log_path=str(folder/'agent.log'),results={'preflight':results,'preflight_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})
   save(folder/'manifest.json',entry);code=process.wait()
  state.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat());save(folder/'state.json',state)
  entry.update(recorded_status=state['status'],results=state);save(folder/'manifest.json',entry);return code

if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
