"""Monitor artifacts and fill free workstation slots without SSH retries."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl,json,shlex,subprocess,time
from publish import publish
ROOT=Path(__file__).resolve().parent

def save(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def main():
 lock=(ROOT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=json.loads((ROOT/'plan.json').read_text());disabled={};last_publish=0
 for iteration in range(14*24*60):
  changed=False
  for host in ['uiuc-cc','a4000','delta','europa','titan']:
   if host in disabled:continue
   script="""from pathlib import Path
import json,subprocess
root=Path(REMOTE);host=HOST
plan=json.loads((root/'plan.json').read_text());started=[]
if host in ['a4000','europa','titan']:
 for profile in plan['agents']:
  if profile['machine']!=host:continue
  folder=root/'agents'/profile['name'];entry=json.loads((folder/'manifest.json').read_text())
  if entry['recorded_status']!='PLANNED' or (folder/'launcher.log').exists():continue
  apps=subprocess.check_output(['nvidia-smi','--id='+str(profile['gpu_index']),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if apps:continue
  with (folder/'launcher.log').open('x') as out:
   p=subprocess.Popen([profile['python'],'-u',str(root/'worker.py'),profile['name']],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  started.append(dict(name=profile['name'],pid=p.pid))
out={'started':started,'agents':{},'trials':{},'allocations':{},'preflight':{}}
for category in ['agents','trials','preflight']:
 for folder in (root/category).iterdir():
  if not folder.is_dir() or not (folder/'manifest.json').exists():continue
  entry=json.loads((folder/'manifest.json').read_text())
  if entry.get('machines')!=[host]:continue
  files={p.name:json.loads(p.read_text()) for p in folder.glob('*.json')}
  if (folder/'config.yaml').exists():files['config.yaml']=(folder/'config.yaml').read_text()
  out[category][folder.name]=files
for p in (root/'allocations').glob('*.json'):out['allocations'][p.name]=json.loads(p.read_text())
print(json.dumps(out))
""".replace('REMOTE',repr(plan['machine_paths'][host])).replace('HOST',repr(host))
   try:
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=45)
    if r.returncode:raise RuntimeError(r.stderr[-2000:])
    data=json.loads(r.stdout)
   except (subprocess.TimeoutExpired,RuntimeError,ValueError) as error:
    disabled[host]=dict(error=str(error),at=datetime.now(timezone.utc).isoformat(),automatic_retries=False);save(ROOT/'disabled-hosts.json',disabled);print(host,'disabled; no SSH retry',flush=True);changed=True;continue
   save(ROOT/('last-'+host+'.json'),dict(observed_at=datetime.now(timezone.utc).isoformat(),**data))
   for category in ['agents','trials','preflight']:
    for name,files in data[category].items():
     folder=ROOT/category/name;folder.mkdir(exist_ok=True)
     before=json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else {}
     for filename,value in files.items():
      if filename.endswith('.yaml'):(folder/filename).write_text(value)
      else:save(folder/filename,value)
     after=files.get('manifest.json',{})
     if before.get('recorded_status')!=after.get('recorded_status') or before.get('current_steps',0)//100000<after.get('current_steps',0)//100000:changed=True
   for name,data in data['allocations'].items():save(ROOT/'allocations'/name,data)
  trials=[json.loads(p.read_text()) for p in (ROOT/'trials').glob('*/manifest.json')]
  complete=[v for v in trials if v['recorded_status']=='COMPLETED'];failed=[v for v in trials if v['recorded_status']=='FAILED']
  entry=json.loads((ROOT/'manifest.json').read_text());entry.update(recorded_status='RUNNING',observed_trials=len(trials),completed_trials=len(complete),failed_trials=len(failed),disabled_hosts=disabled,observations_stale=bool(disabled))
  if complete:
   ranked=sorted(complete,key=lambda v:v['results']['objective']);save(ROOT/'ranking.json',dict(results=[dict(id=v['id'],auc=v['results']['objective'],parameters=v['parameters'],wandb_url=v.get('wandb_url')) for v in ranked]));entry['best_validation_auc']=ranked[0]['results']['objective']
  finished=len(complete)+len(failed)>=plan['run_cap']
  if finished:entry['recorded_status']='COMPLETED' if len(complete)==plan['run_cap'] else 'PARTIAL_FAILURE'
  save(ROOT/'manifest.json',entry)
  if changed or time.monotonic()-last_publish>600 or finished:
   publish();last_publish=time.monotonic()
  if finished:return
  time.sleep(60)

if __name__=='__main__':main()
