"""Submit real immediate-start GPU workers, keeping no pending experiments."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,subprocess,sys,time
ROOT=Path(__file__).resolve().parent

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main(host):
 plan=json.loads((ROOT/'plan.json').read_text());workers=[]
 for p in plan['agents']:
  if p['machine']!=host or not p.get('slurm'):continue
  name=p['name'];node=p['node'];folder=ROOT/'agents'/name
  if json.loads((folder/'manifest.json').read_text())['recorded_status']!='PLANNED':continue
  part,account,gres=p['partition'],p['account'],p['gres']
  cmd=['srun','--immediate=30','--job-name=crl-bayes192','--account='+account,'--partition='+part,'--nodelist='+node,'--gres='+gres,'--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=8G','--time='+p['walltime'],'--output='+str(folder/'slurm-%j.log'),p['python'],'-u',str(ROOT/'worker.py'),name]
  entry=json.loads((folder/'manifest.json').read_text());entry.update(recorded_status='REQUESTING_IMMEDIATE_ALLOCATION',submission_command=cmd,account=account,partition=part)
  save(folder/'manifest.json',entry)
  log=(folder/'srun-client.log').open('x');proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  receipt=dict(purpose='Real immediately-startable GPU allocation; no pending queue retained',machine=host,name=name,node=node,command=cmd,client_pid=proc.pid,seed=42,configuration=str(ROOT/'sweep.yaml'),dependencies=[],status='REQUESTED',requested_at=datetime.now(timezone.utc).isoformat())
  save(ROOT/'allocations'/(name+'.json'),receipt);workers.append((p,proc,log,receipt))
 while workers:
  for p,proc,log,receipt in list(workers):
   code=proc.poll()
   folder=ROOT/'agents'/p['name'];entry=json.loads((folder/'manifest.json').read_text())
   if entry.get('job_ids'):receipt.update(job_ids=entry['job_ids'],status=entry['recorded_status'])
   if code is None:
    save(ROOT/'allocations'/(p['name']+'.json'),receipt);continue
   log.close();receipt.update(exit_code=code,status=entry['recorded_status'],finished_at=datetime.now(timezone.utc).isoformat())
   client_log=(folder/'srun-client.log').read_text()
   if not entry.get('job_ids'):entry['job_ids']=list(dict.fromkeys(re.findall(r'job (\d+)',client_log)))
   if entry['recorded_status']=='REQUESTING_IMMEDIATE_ALLOCATION':
    entry.update(recorded_status='FAILED_STARTUP' if 'has been allocated resources' in client_log else 'IMMEDIATE_ALLOCATION_UNAVAILABLE',exit_code=code,reason=(folder/'srun-client.log').read_text()[-3000:]);save(folder/'manifest.json',entry);receipt.update(status=entry['recorded_status'],reason=entry['reason'])
   save(ROOT/'allocations'/(p['name']+'.json'),receipt);workers.remove((p,proc,log,receipt))
  if workers:time.sleep(5)

if __name__=='__main__':main(sys.argv[1])
