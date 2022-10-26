#from econml.dml import CausalForestDML
#from causallib.estimation import IPW 
from sklearn.linear_model import LogisticRegression

# Q: Can map outside [-1,1] even if output is binary
def estimate_ite_ipw(X, y, t,
                    learner=LogisticRegression(), clip_tr=0.01):
    
    ps = estimate_ps(X = X,
                     t = t,
                     learner = learner)
    ps = clip(ps, clip_tr)
    ite = ((t/ps)-(1-t)/(1-ps))*y
    return ite

def estimate_ps(X, t, learner=LogisticRegression()):
    learner.fit(X, t)
    return learner.predict_proba(X)[:,1]

def clip(array, clip_tr=0.01):
    if clip_tr is not None:
        lv_idx = array<clip_tr
        array[lv_idx] = clip_tr
        hv_idx = array>1-clip_tr
        array[hv_idx] = 1-clip_tr
    return array