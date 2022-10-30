from numpy import array
from numpy.random import uniform


class MHMCMC:
    def __init__(self, f, q, init_x=0):
        self.f = f
        self.q = q
        self.x = init_x
        
    def sample(self, draws):
        return array([self.selection() for _ in range(draws)])
    
    def proposal(self):
        return self.q(self.x)
    
    def selection(self):
        q = self.proposal()
        r = uniform(0,1)

        if (self.f(q)/self.f(self.x) > r):
            self.x = q 
            
        return self.x