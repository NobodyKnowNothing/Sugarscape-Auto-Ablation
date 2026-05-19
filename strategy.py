import math, numpy as np
from pathlib import Path

class Trader:
    __slots__='w','m','v','p','al'
    def __init__(self,s,sp,m,v,p): self.w,self.m,self.v,self.p,self.al=[s,sp],m,v,p,True

class SugarscapeModel:
    def __init__(self,seed,width=50,height=50,initial_population=200,endowment_min=25,endowment_max=50,metabolism_min=1,metabolism_max=5,vision_min=1,vision_max=5,enable_trade=True,**kwargs):
        self.rng,self.width,self.height,self.initial_population,self.enable_trade=np.random.default_rng(seed),width,height,initial_population,enable_trade
        m=np.genfromtxt(Path(__file__).parent/"sugar-map.txt"); self.R=np.stack([m, np.flip(m,1)]); self.M=self.R.copy(); self.occupancy,self.agents,self.all_trade_prices,self.total_trade_volume={},[],[],0
        for _ in range(initial_population):
            x,y=self.rng.integers(0,width),self.rng.integers(0,height)
            while (x,y) in self.occupancy: x,y=self.rng.integers(0,width),self.rng.integers(0,height)
            a=Trader(self.rng.integers(endowment_min,endowment_max+1),self.rng.integers(endowment_min,endowment_max+1),(self.rng.integers(metabolism_min,metabolism_max+1),self.rng.integers(metabolism_min,metabolism_max+1)),self.rng.integers(vision_min,vision_max+1),(x,y)); self.agents.append(a); self.occupancy[(x,y)]=a
    def step(self):
        self.R=np.minimum(self.R+1,self.M); living=[a for a in self.agents if a.al]; self.rng.shuffle(living)
        for a in living:
            m_t=sum(a.m); c=[((a.w[0]+self.R[0,nx,ny])**(a.m[0]/m_t)*(a.w[1]+self.R[1,nx,ny])**(a.m[1]/m_t),abs(dx)+abs(dy),nx,ny) for dx in range(-a.v,a.v+1) for dy in range(-a.v,a.v+1) if 0<abs(dx)+abs(dy)<=a.v and 0<=(nx:=a.p[0]+dx)<self.width and 0<=(ny:=a.p[1]+dy)<self.height and (nx,ny) not in self.occupancy]
            if c: self.rng.shuffle(c); _,_,nx,ny=max(c,key=lambda x:(x[0],-x[1])); self.occupancy.pop(a.p,None); a.p=(nx,ny); self.occupancy[a.p]=a
            a.w[0]+=self.R[0,a.p[0],a.p[1]]-a.m[0]; a.w[1]+=self.R[1,a.p[0],a.p[1]]-a.m[1]; self.R[0,a.p[0],a.p[1]]=self.R[1,a.p[0],a.p[1]]=0
            if any(v<=0 for v in a.w): a.al=False; self.occupancy.pop(a.p,None)
        if self.enable_trade:
            living=[a for a in self.agents if a.al]; self.rng.shuffle(living)
            for a in living:
                for nx,ny in [(a.p[0]+dx,a.p[1]+dy) for dx in range(-a.v,a.v+1) for dy in range(-a.v,a.v+1) if abs(dx)+abs(dy)<=a.v]:
                    if 0<=(nx:=nx)<self.width and 0<=(ny:=ny)<self.height and (o:=self.occupancy.get((nx,ny))) and o.al and o!=a:
                        while all(v > 0 for v in a.w + o.w):
                            m_ta,m_to=sum(a.m),sum(o.m); mrs_a,mrs_o=(a.w[1]/a.m[1])/(a.w[0]/a.m[0]),(o.w[1]/o.m[1])/(o.w[0]/o.m[0])
                            if math.isclose(mrs_a,mrs_o) or mrs_a<=0 or mrs_o<=0: break
                            p=math.sqrt(mrs_a*mrs_o); s,b=(a,o) if mrs_a>mrs_o else (o,a); m_ts,m_tb=sum(s.m),sum(b.m); sa,spa=(1,int(p)) if p>=1 else (int(1/p),1)
                            if s.w[0]+sa>0 and b.w[0]-sa>0 and s.w[1]-spa>0 and b.w[1]+spa>0 and (s.w[0]+sa)**(s.m[0]/m_ts)*(s.w[1]-spa)**(s.m[1]/m_ts)>s.w[0]**(s.m[0]/m_ts)*s.w[1]**(s.m[1]/m_ts) and (b.w[0]-sa)**(b.m[0]/m_tb)*(b.w[1]+spa)**(b.m[1]/m_tb)>b.w[0]**(b.m[0]/m_tb)*b.w[1]**(b.m[1]/m_tb):
                                s.w[0]+=sa; b.w[0]-=sa; s.w[1]-=spa; b.w[1]+=spa; self.all_trade_prices.append(p); self.total_trade_volume+=1
                            else: break

def create_model(seed=42,**params): return SugarscapeModel(seed=seed,**params)
def run_model(model,steps=200):
    for _ in range(steps): model.step()
    living=[a for a in model.agents if a.al]
    return {"agent_wealths":[sum(a.w) for a in living], "final_population":len(living), "initial_population":model.initial_population, "trade_prices":model.all_trade_prices, "trade_volume":model.total_trade_volume, "agent_positions":[a.p for a in living], "grid_width":model.width, "grid_height":model.height, "steps_run":steps}