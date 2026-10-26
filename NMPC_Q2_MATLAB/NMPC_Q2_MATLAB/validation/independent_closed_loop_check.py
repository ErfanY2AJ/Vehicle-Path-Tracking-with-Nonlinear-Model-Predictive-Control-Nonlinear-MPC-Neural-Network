"""Optional 15 s independent NumPy closed-loop checks. NOT MATLAB execution."""
import sys,json,time
from pathlib import Path
import numpy as np
import independent_q2_check as nn
import independent_core as core
runs=nn.generate();parts,prep,groups=nn.prepare(runs);net,training=nn.train(parts,prep)
results=[]
for factor in [1.,1.1]:
 for method in ['math','neural']:
  dt=.1;N=20;K=150;ref,_=core.reference(Tsim=15,N=N)
  x=np.array([.4,0,14.5,0,core.torque(14.5)]);prev=np.array([core.torque(14.5),0])/core.SCALE
  warm=np.tile(prev,(N,1)).T;actual=core.P.copy();actual['m']*=factor
  history=[x.copy()];statuses=[];clock=time.perf_counter()
  for k in range(K):
   yr=ref[:3,k:k+N+1];zr=np.vstack((core.torque(yr[2,1:])/core.SCALE[0],np.zeros(N)))
   if method=='math':fun=lambda z:core.objective(z,x,prev,yr,zr,N=N)
   else:fun=lambda z:nn.objective(z,x,prev,yr,zr,net,N=N)
   jw=fun(warm.ravel(order='F'))[0]
   z,j,flag,it=core.solve(fun,warm.ravel(order='F'))
   seq=z.reshape(2,N,order='F');status=1 if flag>0 else 2
   if not np.isfinite(z).all() or not np.isfinite(j) or (abs(z)>1+1e-9).any() or (flag<=0 and j>jw):
    seq=warm.copy();status=3
   x=core.step(x,core.SCALE*seq[:,0],p=actual)[0]
   assert np.isfinite(x).all() and x[2]>0 and (abs(seq)<=1+1e-9).all()
   history.append(x.copy());statuses.append(status)
   prev=seq[:,0].copy();warm=np.column_stack((seq[:,1:],seq[:,-1]))
   if (k+1)%50==0:print(f'{method} mass={factor}: {k+1}/{K}',flush=True)
  X=np.array(history).T;e=X[:3]-ref[:3,:K+1];ep=np.linalg.norm(e[:2],axis=0);post=np.arange(K+1)*dt>=5-1e-10
  results.append(dict(method=method,massFactor=factor,completed=True,
   positionRMSE_after5_m=float(np.sqrt(np.mean(ep[post]**2))),
   speedRMSE_after5_mps=float(np.sqrt(np.mean(e[2,post]**2))),
   fallback=statuses.count(3),elapsedPython_s=time.perf_counter()-clock))
print(json.dumps(results,indent=2),flush=True)
Path(__file__).with_name('independent_closed_loop_results.json').write_text(json.dumps({'execution':'Independent NumPy port, not native MATLAB results','duration_s':15,'Np':20,'dt':0.1,'results':results},indent=2))
