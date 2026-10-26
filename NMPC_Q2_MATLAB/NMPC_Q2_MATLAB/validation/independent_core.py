"""Independent executable port of the MATLAB core. NOT MATLAB execution."""
import numpy as np, json, time
from pathlib import Path

P=dict(m=1270.,R=.3036,CA=.9871,tau=.5107,eta=.96,fr=.01,g=9.8,L=2.)
SCALE=np.array([1965.1,.5]);Q=np.diag([2.,2.,3.]);PTERM=np.diag([8.,8.,6.]);RR=np.diag([.2,.3]);SS=np.diag([.3,3.])
def torque(v,p=P):return p['R']/p['eta']*(p['CA']*np.asarray(v)**2+p['m']*p['g']*p['fr'])
def step(x,u,p=P,dt=.1):
    v,phi,T=x[2:];delta=u[1]
    out=np.array([x[0]+dt*v*np.cos(phi),x[1]+dt*v*np.sin(phi),v+p['eta']*dt/(p['m']*p['R'])*(T-p['R']/p['eta']*(p['CA']*v*v+p['m']*p['g']*p['fr'])),phi+dt*v/p['L']*np.tan(delta),T+dt/p['tau']*(u[0]-T)])
    A=np.eye(5);A[0,2]=dt*np.cos(phi);A[0,3]=-dt*v*np.sin(phi);A[1,2]=dt*np.sin(phi);A[1,3]=dt*v*np.cos(phi);A[2,2]=1-2*dt*p['CA']*v/p['m'];A[2,4]=p['eta']*dt/(p['m']*p['R']);A[3,2]=dt/p['L']*np.tan(delta);A[4,4]=1-dt/p['tau']
    B=np.zeros((5,2));B[3,1]=dt*v/(p['L']*np.cos(delta)**2);B[4,0]=dt/p['tau']
    return out,A,B
def reference(Tsim=100,dt=.1,N=20,alpha=.3):
    t=np.arange(round(Tsim/dt)+N+1)*dt;yr=np.zeros_like(t)
    m=(t>=10)&(t<70);yr[m]=.24*np.cos(2*np.pi*(t[m]-10)/30)
    m=(t>=70)&(t<92);yr[m]=-.24*np.cos(2*np.pi*(t[m]-70)/22);yr*=alpha
    a=((t>=1)&(t<2)).astype(float);x=np.zeros((4,len(t)));x[2,0]=14
    for k in range(len(t)-1):
        v,phi=x[2:,k];x[:,k+1]=x[:,k]+dt*np.array([v*np.cos(phi),v*np.sin(phi),a[k],yr[k]])
    return x,yr
def objective(z,x0,prev,yr,zr,dt=.1,N=20,S=SS):
    Z=z.reshape((2,N),order='F');X=np.zeros((5,N+1));X[:,0]=x0
    AA=[];BB=[];J=0.
    for j in range(N):
        X[:,j+1],A,B=step(X[:,j],SCALE*Z[:,j],dt=dt);AA.append(A);BB.append(B@np.diag(SCALE))
        e=X[:3,j+1]-yr[:,j+1];J+=e@Q@e
    e=X[:3,-1]-yr[:,-1];J+=e@PTERM@e
    E=Z-zr;D=Z-np.column_stack((prev,Z[:,:-1]));J+=np.sum(E*(RR@E))+np.sum(D*(S@D))
    G=2*RR@E+2*S@D;G[:,:-1]-=2*S@D[:,1:]
    lam=np.r_[2*(Q+PTERM)@e,0.,0.]
    for j in range(N-1,-1,-1):
        G[:,j]+=BB[j].T@lam
        if j>0:lam=AA[j].T@lam+np.r_[2*Q@(X[:3,j]-yr[:,j]),0.,0.]
    return J,G.ravel(order='F')
def solve(fun,z,maxiter=200):
    z=np.clip(z,-1,1);J,g=fun(z);n=len(z);H=np.eye(n);flag=0
    for it in range(maxiter):
        pg=z-np.clip(z-g,-1,1)
        if np.max(np.abs(pg))<=1e-6:flag=1;break
        active=((z<=-1+1e-12)&(g>0))|((z>=1-1e-12)&(g<0));free=~active;di=np.zeros(n);di[free]=-H[np.ix_(free,free)]@g[free]
        if not np.all(np.isfinite(di)) or g@di>=-1e-14*np.linalg.norm(g)*np.linalg.norm(di):di=-g.copy();di[active]=0;H=np.eye(n)
        accepted=False;a=1.
        for ls in range(40):
            trial=np.clip(z+a*di,-1,1);s=trial-z;Jt,gt=fun(trial)
            if np.isfinite(Jt) and np.all(np.isfinite(gt)) and g@s<0 and Jt<=J+1e-4*(g@s):accepted=True;break
            a*=.5
        if not accepted:break
        y=gt-g;ys=y@s
        if ys>1e-10*np.linalg.norm(y)*np.linalg.norm(s) and ys>0:
            Hy=H@y;H+=((ys+y@Hy)/ys**2)*np.outer(s,s)-(np.outer(Hy,s)+np.outer(s,Hy))/ys;H=.5*(H+H.T)
        old=J;z,J,g=trial,Jt,gt
        if np.max(np.abs(z-np.clip(z-g,-1,1)))<=1e-6:flag=1;break
        if np.max(np.abs(s))<=1e-10:break
    return z,J,flag,it+1
