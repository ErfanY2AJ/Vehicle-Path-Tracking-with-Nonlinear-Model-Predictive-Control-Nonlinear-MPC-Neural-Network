"""Check the Jacobian expressions read from q2_nn_step.m.

This evaluates only a small numeric expression subset, not native MATLAB.
It supplements the independent port by checking the delivered derivative
expressions themselves against forward-map finite differences.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import re
import numpy as np


def expressions(source):
    start=source.index("    tanhDerivative2=")
    end=source.index("    rotate=",start)
    block=re.sub(r"%[^\n]*","",source[start:end])
    result=[]
    for statement in block.split(";"):
        if not statement.strip():
            continue
        lhs,rhs=statement.strip().split("=",1)
        rhs=rhs.replace("net.prep.affine(:,2:end)","net.prep.affine[:,1:]")
        rhs=rhs.replace(".^","__POWER__").replace(".*","__PRODUCT__")
        rhs=rhs.replace("*","@").replace("__PRODUCT__","*").replace("__POWER__","**")
        rhs=rhs.replace("./","/").replace(".'",".T")
        result.append((lhs.strip(),rhs))
    return result


def run(source):
    rng=np.random.default_rng(20261026)
    statements=expressions(source)
    max_fd=max_formula=0.
    for hidden in [(3,2),(32,32)]:
        h1,h2=hidden
        for _ in range(20):
            p=NS(W1=.5*rng.normal(size=(h1,4)),b1=rng.normal(size=(h1,1)),
                 W2=.3*rng.normal(size=(h2,h1)),b2=rng.normal(size=(h2,1)),
                 W3=.2*rng.normal(size=(5,h2)),b3=rng.normal(size=(5,1)))
            prep=NS(affine=.1*rng.normal(size=(5,5)),
                    residualScale=rng.uniform(.02,2,size=(5,1)),
                    featureStd=rng.uniform(.3,5,size=(4,1)))
            f=rng.normal(size=(4,1))
            def forward(feature):
                q=feature/prep.featureStd
                a1=np.tanh(p.W1@q+p.b1);a2=np.tanh(p.W2@a1+p.b2)
                return (prep.affine@np.vstack((np.ones((1,1)),q))
                        +prep.residualScale*(p.W3@a2+p.b3)),a1,a2
            _,a1,a2=forward(f)
            scope=dict(P=p,net=NS(prep=prep),A1=a1,A2=a2)
            for lhs,rhs in statements:
                scope[lhs]=eval(rhs,{"__builtins__":{}},scope)
            actual=scope["J"]
            reference=(prep.affine[:,1:]+np.diag(prep.residualScale[:,0])@p.W3
                       @np.diag(1-a2[:,0]**2)@p.W2
                       @np.diag(1-a1[:,0]**2)@p.W1)/prep.featureStd.T
            h=1e-5
            fd=np.column_stack([
                ((forward(f+np.eye(4)[:,i:i+1]*h)[0]
                  -forward(f-np.eye(4)[:,i:i+1]*h)[0])/(2*h))[:,0]
                for i in range(4)])
            max_fd=max(max_fd,float(np.linalg.norm(actual-fd)/max(1,np.linalg.norm(fd))))
            max_formula=max(max_formula,float(np.linalg.norm(actual-reference)))
    return max_fd,max_formula


if __name__=="__main__":
    source=Path(__file__).resolve().parents[1].joinpath("q2_nn_step.m").read_text()
    fd,formula=run(source)
    assert fd<1e-6 and formula<1e-12,(fd,formula)
    # Confirm that the omitted-square bug cannot pass this regression check.
    broken=source.replace("1-A2.^2","1-A2").replace("1-A1.^2","1-A1")
    broken_fd,_=run(broken)
    assert broken_fd>1e-3,broken_fd
    result={"execution":"NumPy evaluation of extracted MATLAB expressions; NOT native MATLAB",
            "networksChecked":40,"maxRelativeFiniteDifferenceError":fd,
            "maxDenseFormulaDifference":formula,"previousBugRelativeError":broken_fd,
            "previousBugRejected":True}
    Path(__file__).with_name("matlab_expression_check.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
