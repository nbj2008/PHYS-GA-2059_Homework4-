import numpy as np
import pandas as pd
from scipy import optimize as opt

class Model:
    def __init__(s, T, params, data):
        s.data = data
        s.y_train = np.array(data.data)
        s.t_train = np.array(data.time)
        s.y_σ = np.array(data.σy)

        s.T = T       
        s.set_params(params)
        
    def set_params(s, params):
        pass

    def pred(s, t):
        guess = s.in_pred(t, s.T, *s.params)
        return guess
    
    def negloglike(s):
        return 1/2*np.sum(([((y - s.pred(t))**2/ (σ**2))  for y,t, σ in zip(s.y_train, s.t_train, s.y_σ)])) 
    
    def fit(s, params, p=False):
        s.set_params(params)
        if p:
            print(s.negloglike(), s.params)
        return s.negloglike()

    @staticmethod
    def in_pred(t):
        pass
        
        
class DCOffset(Model):
    def __init__(s, T, params, data):
        super().__init__(T, params, data) 
        
    def set_params(s, params):
        s.params = params
        s.DC = params[0]
    
    def analytic_params(s):
        A, y, c = generate_mats(s.data, None)
        return calc_params(A, y, c), np.linalg.inv(A @(A * c).T)
        
    @staticmethod
    def in_pred(t, T, DC):
        return DC * np.ones_like(t)
    
    @property
    def dc_stddev(s):
        X, y, c = generate_mats(s.data, None)
        params = calc_params(X, y, c)
        errors = np.sqrt(np.linalg.inv(X @(X * c).T))
        
        return errors[0][0]


class Signal(Model):
    def __init__(s, T, params, data):
        super().__init__(T, params, data)
        s.data['cos'] = pd.Series(np.cos(2*np.pi* 1/T * data.time))
        s.data['sin'] = pd.Series(np.sin(2*np.pi* 1/T * data.time))
    
    @property
    def amplitude(s):
        return np.sqrt(s.A**2 + s.B**2)
    
    @property
    def analytic_amplitude_error(s):
        X, y, c = generate_mats(s.data, 'sin','cos')
        params = calc_params(X, y, c)
        errors = np.sqrt(np.diag(np.linalg.inv(X @(X * c).T)))
        
        return 1/np.sqrt(params[1]**2 + params[2]**2) * (params[1]*errors[1] + params[2]*errors[2])
        

    @property
    def error(s):
        return np.sqrt(s.A**2 + s.B**2)
    
    def set_params(s, params):
        s.params = params
        s.DC = params[0]
        s.A = params[1]
        s.B = params[2]
        
    def analytic_params(s):
        A, y, c = generate_mats(s.data,'sin','cos')
        return calc_params(A, y, c), np.linalg.inv(A @(A * c).T)
        
    @staticmethod
    def in_pred(t, T, DC, A, B):
        return DC + (A * np.sin(2*np.pi* 1/T * t) + B * np.cos(2*np.pi* 1/T * t))


class AmpSignal(Model):
    def __init__(s, T, A, params, data):
        super().__init__(T, params, data)
        s.Amp = A
    
    @classmethod
    def from_signal(cls, sig):
        return cls(sig.T, sig.amplitude, [sig.DC, np.arctan(sig.A/sig.B)], sig.data) 
        
    def set_params(s, params):
        s.params = params
        s.DC = s.params[0]
        s.ϕ  = s.params[1]
    
    def pred(s, t):
        return s.in_pred(t, s.T, s.Amp, s.DC, s.ϕ)
    
    @staticmethod
    def in_pred(t, T, A, DC, ϕ):
        return DC + A * np.cos(2 * np.pi * t /T + ϕ)

class TwoAmpSignal(Model):
    def __init__(s, T, T2, A, A2, params, data):
        super().__init__(T, params, data)
        s.T2 = T2
        s.Amp1 = A
    
    @classmethod
    def from_signal(cls, sig):
        return cls(sig.T, sig.T2, *sig.amplitude, [sig.DC, np.arctan(sig.A/sig.B), sig.C, sig.D], sig.data) 
        
    def set_params(s, params):
        s.params = params
        s.DC  = s.params[0]
        s.ϕ1  = s.params[1]
        s.C = s.params[2]
        s.D = s.params[3]
    
    def analytic_params(s):
        A, y, c = generate_mats(s.data,'sin2','cos2')
        return calc_params(A, y, c), np.linalg.inv(A @(A * c).T)
    
    def pred(s, t):
        return s.in_pred(t, s.T, s.T2, s.Amp1, s.DC, s.ϕ1, s.C, s.D)
    
    @staticmethod
    def in_pred(t, T, T2, A, DC, ϕ1, C, D):
        return DC + A * np.cos(2 * np.pi * t / T + ϕ1) + (C * np.sin(2*np.pi* 1/T2 * t) + D * np.cos(2*np.pi* 1/T2 * t))
    
class TwoSignal(Model):
    def __init__(s, T, T2, params, data):
        super().__init__(T, params, data) 
        s.T2 = T2
        
        s.data['cos1'] = pd.Series(np.cos(2*np.pi* 1/T * data.time))
        s.data['sin1'] = pd.Series(np.sin(2*np.pi* 1/T * data.time))
        s.data['cos2'] = pd.Series(np.cos(2*np.pi* 1/T2 * data.time))
        s.data['sin2'] = pd.Series(np.sin(2*np.pi* 1/T2 * data.time))
        
    @property
    def amplitude(s):
        return np.sqrt(s.A**2 + s.B**2), np.sqrt(s.C**2 + s.D**2)
    
    def set_params(s, params):
        s.params = params
        s.DC = params[0]
        s.A = params[1]
        s.B = params[2]
        s.C = params[3]
        s.D = params[4]
    
    def analytic_params(s):
        A, y, c = generate_mats(s.data,'sin1','cos1','sin2','cos2')
        return calc_params(A, y, c), np.linalg.inv(A @(A * c).T)
       
    @property
    def analytic_amplitude_error(s):
        X, y, c = generate_mats(s.data, 'sin1','cos1','sin2','cos2')
        params = calc_params(X, y, c)
        errors = np.sqrt(np.diag(np.linalg.inv(X @(X * c).T)))
        
        return 1/np.sqrt(params[1]**2+params[2]**2) * (params[1]*errors[1] + params[2]*errors[2]) \
            , 1/np.sqrt(params[3]**2+params[4]**2) * (params[3]*errors[3] + params[4]*errors[4])
        
    def pred(s, t):
        guess = s.in_pred(t, s.T, s.T2, *s.params)
        return guess
    
    @staticmethod
    def in_pred(t, T, T2, DC, A, B, C, D):
        return DC + (A * np.sin(2*np.pi* 1/T * t) + B * np.cos(2*np.pi* 1/T * t)) \
            + (D * np.sin(2*np.pi* 1/T2 * t) + D * np.cos(2*np.pi* 1/T2 * t))

    @classmethod
    def with_period_param(cls, T, params, data):
        return cls(T, params[0], params[1:], data)
    
    
def generate_mats(data, *cols):
    if cols is None:
        A = np.ones(data.shape[0])
    else:
        cols = [x for x in cols if x is not None]
        A = np.stack((np.ones(data.shape[0]), *data[cols].T.values))
    y = np.array(data.data)
    c = np.array(1/data.σy**2)
    return A, y, c

def calc_params(A, y, c):
    return np.linalg.solve(A @ (A * c).T, A @ (c * y).T)

def fit_data(params, span, num=100, span_funcs=(np.ones_like, lambda x: x)):
    low, high = span
    rang = np.linspace(low, high, num=num)
    fit = params @ np.stack([func(rang) for func in span_funcs])
    return rang, fit


def bootstrap_amp(sig: [Signal, TwoSignal], trials: int = 128):
    boots = []
    issig = isinstance(sig, Signal)
    
    def get_sig_amp():
        return sig.amplitude if issig else sig.amplitude[0]
    
    def get_temp():
        if issig:
            return Signal(sig.T, sig.params, sig.data.iloc[np.random.randint(low=0,high=sig.data.shape[0],size=sig.data.shape[0])])
        else:
            return TwoSignal(sig.T, sig.T2, sig.params, sig.data.iloc[np.random.randint(low=0,high=sig.data.shape[0],size=sig.data.shape[0])])
        
    def get_amp(temp):
        return temp.amplitude if issig else temp.amplitude[0]
    
    for _ in range(trials):
        temp = get_temp()
        _ = opt.minimize(temp.fit, temp.params)
        boots.append(get_amp(temp))
    
    boots = np.array(boots)
    boots_p25 = np.quantile(boots, 0.25)
    boots_97p5 = np.quantile(boots, 0.975)
    return np.sqrt(np.sum([(get_sig_amp() - b)**2/trials for b in boots])), (boots_p25, boots_97p5)