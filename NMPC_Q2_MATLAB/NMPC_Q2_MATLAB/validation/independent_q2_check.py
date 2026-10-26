"""Independent NumPy verification. This DOES NOT execute the MATLAB sources."""
import json
import time
from pathlib import Path
import numpy as np
from independent_core import P, SCALE, Q, PTERM, RR, SS, step, torque, reference, solve


def features(x, u):
    return np.vstack((x[2], x[4]/SCALE[0], u[0]/SCALE[0], u[1]/SCALE[1]))


def body_targets(x, xn):
    d=xn-x; c=np.cos(x[3]); s=np.sin(x[3])
    return np.vstack((c*d[0]+s*d[1],-s*d[0]+c*d[1],d[2:]))


def generate():
    rng=np.random.default_rng(42); runs=[]
    for run in range(40):
        for attempt in range(100):
            v=5+20*(run+rng.random())/40
            phi=-np.pi+2*np.pi*rng.random()
            t=torque(v)+.6*SCALE[0]*(2*rng.random()-1)
            x=np.zeros((5,201));u=np.zeros((2,200))
            x[:,0]=np.r_[400*rng.random(2)-200,v,phi,t]
            for k in range(200):
                if k%8==0:
                    block=k//8+1; command=2*rng.random(2)-1
                    if block==1: command=np.ones(2)
                    if block==2: command=-np.ones(2)
                    if block>2:
                        for d in range(2):
                            if rng.random()<.25:command[d]=2*(rng.random()>.5)-1
                applied=command.copy()
                if x[2,k]<6:applied[0]=1
                if x[2,k]>25:applied[0]=-1
                u[:,k]=SCALE*applied
                candidate=step(x[:,k],u[:,k])[0]
                if not np.isfinite(candidate).all() or candidate[2]<=0:break
                x[:,k+1]=candidate
            else:
                runs.append((x,u));break
        else:raise RuntimeError("Could not generate a valid independent run")
    return runs


def prepare(runs):
    ids=np.random.default_rng(43).permutation(40)
    groups=[ids[:28],ids[28:34],ids[34:]]
    assert not set(groups[0])&set(groups[1]) and not set(groups[0])&set(groups[2])
    parts=[]
    for group in groups:
        x=np.hstack([runs[i][0][:,:-1] for i in group])
        u=np.hstack([runs[i][1] for i in group])
        xn=np.hstack([runs[i][0][:,1:] for i in group])
        parts.append(dict(x=x,u=u,xn=xn,f=features(x,u),d=body_targets(x,xn)))
    t=parts[0]; mu=t["f"].mean(1,keepdims=True); sd=t["f"].std(1,ddof=1,keepdims=True)
    q=(t["f"]-mu)/sd; aug=np.vstack((np.ones(q.shape[1]),q))
    affine=np.linalg.lstsq(aug.T,t["d"].T,rcond=None)[0].T
    residual=t["d"]-affine@aug
    rm=residual.mean(1,keepdims=True); rs=residual.std(1,ddof=1,keepdims=True)
    active=rs>1e-10; rs=np.where(active,rs,0)
    for p in parts:
        p["q"]=(p["f"]-mu)/sd
        p["y"]=(p["d"]-affine@np.vstack((np.ones(p["q"].shape[1]),p["q"]))-rm)/np.where(active,rs,1)
        p["y"]=np.where(active,p["y"],0)
    return parts,dict(mu=mu,sd=sd,affine=affine,rm=rm,rs=rs,active=active),groups


def init(hidden,active,seed=44):
    rng=np.random.default_rng(seed);h1,h2=hidden
    p={"W1":np.sqrt(6/(4+h1))*(2*rng.random((h1,4))-1),"b1":np.zeros((h1,1)),
       "W2":np.sqrt(6/(h1+h2))*(2*rng.random((h2,h1))-1),"b2":np.zeros((h2,1)),
       "W3":.1*np.sqrt(6/(h2+5))*(2*rng.random((5,h2))-1),"b3":np.zeros((5,1))}
    p["W3"][~active[:,0]]=0
    return p


def forward(p,q):
    a1=np.tanh(p["W1"]@q+p["b1"])
    a2=np.tanh(p["W2"]@a1+p["b2"])
    return p["W3"]@a2+p["b3"],a1,a2


def loss(p,q,y,active,wd=0):
    pred,a1,a2=forward(p,q);e=(pred-y)*active;n=max(1,active.sum())*q.shape[1]
    value=(e*e).sum()/n+.5*wd*sum((p[k]**2).sum() for k in ["W1","W2","W3"])
    d3=2*e/n;d2=(p["W3"].T@d3)*(1-a2*a2);d1=(p["W2"].T@d2)*(1-a1*a1)
    g={"W3":d3@a2.T+wd*p["W3"],"b3":d3.sum(1,keepdims=True),
       "W2":d2@a1.T+wd*p["W2"],"b2":d2.sum(1,keepdims=True),
       "W1":d1@q.T+wd*p["W1"],"b1":d1.sum(1,keepdims=True)}
    return value,g


def nn_step(x,u,net,jac=False):
    single=x.ndim==1
    if single:x=x[:,None];u=u[:,None]
    p,prep=net
    q=(features(x,u)-prep["mu"])/prep["sd"]
    residual,a1,a2=forward(p,q)
    d=prep["affine"]@np.vstack((np.ones(q.shape[1]),q))+prep["rm"]+prep["rs"]*residual
    c=np.cos(x[3]);s=np.sin(x[3])
    xn=x+np.vstack((c*d[0]-s*d[1],s*d[0]+c*d[1],d[2:]))
    if not jac:return xn[:,0] if single else xn
    hidden=(p["W3"]*(1-a2[:,0]**2))@p["W2"]
    hidden=hidden*(1-a1[:,0]**2)
    j=prep["affine"][:,1:]+prep["rs"]*(hidden@p["W1"])
    j=j/prep["sd"][:,0]
    rot=np.eye(5);rot[:2,:2]=[[c[0],-s[0]],[s[0],c[0]]];j=rot@j
    a=np.eye(5);a[:,2]+=j[:,0];a[:,4]+=j[:,1]/SCALE[0]
    a[0,3]=-s[0]*d[0,0]-c[0]*d[1,0];a[1,3]=c[0]*d[0,0]-s[0]*d[1,0]
    b=j[:,2:]/SCALE
    return xn[:,0],a,b


def objective(z,x,prev,yr,zr,net,N=20):
    Z=z.reshape(2,N,order="F");X=np.zeros((5,N+1));X[:,0]=x
    aa=[];bb=[];value=0.
    for j in range(N):
        X[:,j+1],a,b=nn_step(X[:,j],SCALE*Z[:,j],net,True)
        aa.append(a);bb.append(b*SCALE)
        e=X[:3,j+1]-yr[:,j+1];value+=e@Q@e
    value+=e@PTERM@e
    E=Z-zr;D=Z-np.column_stack((prev,Z[:,:-1]))
    value+=(E*(RR@E)).sum()+(D*(SS@D)).sum()
    G=2*RR@E+2*SS@D;G[:,:-1]-=2*SS@D[:,1:]
    lam=np.r_[2*(Q+PTERM)@e,0,0]
    for j in range(N-1,-1,-1):
        G[:,j]+=bb[j].T@lam
        if j>0:lam=aa[j].T@lam+np.r_[2*Q@(X[:3,j]-yr[:,j]),0,0]
    return value,G.ravel(order="F")


def derivative_checks(net):
    rng=np.random.default_rng(71);active=np.ones((5,1),bool);p=init([3,2],active)
    q=rng.normal(size=(4,7));y=rng.normal(size=(5,7));_,g=loss(p,q,y,active,1e-4)
    num=den=0.;h=1e-6
    for k in p:
        for idx in np.ndindex(p[k].shape):
            orig=p[k][idx];p[k][idx]=orig+h;plus=loss(p,q,y,active,1e-4)[0]
            p[k][idx]=orig-h;minus=loss(p,q,y,active,1e-4)[0];p[k][idx]=orig
            fd=(plus-minus)/(2*h);num+=(fd-g[k][idx])**2;den+=fd**2
    back=np.sqrt(num)/max(1,np.sqrt(den))
    x=np.array([2.,-3,14,.4,120]);u=np.array([300.,.12]);xn,a,b=nn_step(x,u,net,True)
    h=1e-5
    an=np.column_stack([(nn_step(x+np.eye(5)[j]*h,u,net)-nn_step(x-np.eye(5)[j]*h,u,net))/(2*h) for j in range(5)])
    bn=np.column_stack([(nn_step(x,u+np.eye(2)[j]*h,net)-nn_step(x,u-np.eye(2)[j]*h,net))/(2*h) for j in range(2)])
    ae=np.linalg.norm(a-an)/max(1,np.linalg.norm(an));be=np.linalg.norm(b-bn)/max(1,np.linalg.norm(bn))
    N=3;ref,_=reference(Tsim=1,N=N);yr=ref[:3,:N+1];zr=np.vstack((torque(yr[2,1:])/SCALE[0],np.zeros(N)))
    z=.1*rng.normal(size=2*N);prev=np.array([.08,.03])
    fun=lambda z:objective(z,x,prev,yr,zr,net,N)
    _,g=fun(z);h=1e-6
    gn=np.array([(fun(z+np.eye(2*N)[j]*h)[0]-fun(z-np.eye(2*N)[j]*h)[0])/(2*h) for j in range(2*N)])
    ge=np.linalg.norm(g-gn)/max(1,np.linalg.norm(gn))
    shift=np.array([1000.,-800,0,2*np.pi,0])
    eq=np.max(np.abs(nn_step(x+shift,u,net)-xn-shift))
    assert back<1e-6 and ae<1e-6 and be<1e-6 and ge<1e-5 and eq<1e-9
    return dict(backpropRelativeError=back,neuralStateJacobianRelativeError=ae,
                neuralInputJacobianRelativeError=be,objectiveGradientRelativeError=ge,equivarianceError=eq)


def train(parts,prep,epochs=150):
    p=init([32,32],prep["active"]);rng=np.random.default_rng(44)
    t,v,_=parts;initial=loss(p,v["q"],v["y"],prep["active"])[0]
    bestv=initial;best={k:a.copy() for k,a in p.items()};m={k:np.zeros_like(a) for k,a in p.items()};vv={k:np.zeros_like(a) for k,a in p.items()}
    stepcount=0;bestepoch=0
    for epoch in range(1,epochs+1):
        order=rng.permutation(t["q"].shape[1]);rate=.002*.3**((epoch-1)//100)
        for start in range(0,len(order),512):
            idx=order[start:start+512];_,g=loss(p,t["q"][:,idx],t["y"][:,idx],prep["active"],1e-6)
            clip=min(1,5/max(np.sqrt(sum((a*a).sum() for a in g.values())),np.finfo(float).tiny))
            stepcount+=1
            for k in p:
                gr=g[k]*clip;m[k]=.9*m[k]+.1*gr;vv[k]=.999*vv[k]+.001*gr*gr
                p[k]-=rate*(m[k]/(1-.9**stepcount))/(np.sqrt(vv[k]/(1-.999**stepcount))+1e-8)
        val=loss(p,v["q"],v["y"],prep["active"])[0]
        if val<bestv-1e-7:bestv=val;best={k:a.copy() for k,a in p.items()};bestepoch=epoch
        if epoch%25==0:print(f"Independent NumPy training {epoch}/{epochs}, validation={val:.6g}",flush=True)
    assert bestv<initial*.1
    return (best,prep),dict(initialValidationMSE=initial,bestValidationMSE=bestv,bestEpoch=bestepoch)


def recursive(runs,ids,net):
    out=[]
    for H in [1,5,10,20]:
        pe=[];ve=[];pb=[];vb=[]
        for i in ids:
            x,u=runs[i];starts=np.arange(201-H);pred=x[:,starts].copy()
            for j in range(H):pred=nn_step(pred,u[:,starts+j],net)
            truth=x[:,starts+H]
            pe.extend(np.linalg.norm(pred[:2]-truth[:2],axis=0));ve.extend(abs(pred[2]-truth[2]))
            pb.extend(np.linalg.norm(x[:2,starts]-truth[:2],axis=0));vb.extend(abs(x[2,starts]-truth[2]))
        out.append(dict(H=H,startingPoints=len(pe),neuralMeanPositionError_m=float(np.mean(pe)),
                        neuralMeanAbsSpeedError_mps=float(np.mean(ve)),
                        persistenceMeanPositionError_m=float(np.mean(pb)),persistenceMeanAbsSpeedError_mps=float(np.mean(vb))))
    return out


if __name__=="__main__":
    start=time.perf_counter();runs=generate();parts,prep,groups=prepare(runs)
    print("Generated 40 independent valid runs; 28/6/6 split.",flush=True)
    net,training=train(parts,prep)
    report={"execution":"Independent NumPy port; NOT native MATLAB execution or MATLAB controller results",
            "checks":derivative_checks(net),"trainingSmokeCheck":training,
            "recursiveIndependentPort":recursive(runs,groups[2],net),
            "elapsedPython_s":time.perf_counter()-start}
    assert all(np.isfinite(x).all() and np.min(x[2])>0 for x,u in runs)
    path=Path(__file__).with_name("independent_verification.json")
    path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
