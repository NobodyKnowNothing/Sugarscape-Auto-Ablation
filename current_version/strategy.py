import numpy as np
from pathlib import Path

class SugarscapeModel:
    def __init__(self,seed,width=50,height=50,initial_population=200,endowment_min=25,endowment_max=50,metabolism_min=1,metabolism_max=5,vision_min=1,vision_max=5,enable_trade=True,**kwargs):
        self.rng,self.width,self.height,self.initial_population,self.enable_trade=np.random.default_rng(seed),width,height,initial_population,enable_trade
        m=np.genfromtxt(Path(__file__).parent/"sugar-map.txt"); self.R=np.stack([m, m[:,::-1]]); self.M=self.R.copy(); self.occupancy,self.agents,self.all_trade_prices,self.total_trade_volume={},[],[],0
        self.off=[(dx,dy,abs(dx)+abs(dy))for dx in range(-5,6)for dy in range(-5,6)if dx or dy]
        while len(self.agents)<initial_population:
            p=self.rng.integers(0,width),self.rng.integers(0,height)
            if p not in self.occupancy:
                a=[*self.rng.integers(endowment_min,endowment_max+1,2),*self.rng.integers(metabolism_min,metabolism_max+1,2),self.rng.integers(vision_min,vision_max+1),*p,True]
                self.agents.append(a); self.occupancy[p]=a
    def step(self):
        self.R=np.minimum(self.R+1,self.M); living=[a for a in self.agents if a[7]]; self.rng.shuffle(living)
        for a in living:
            if c := max((( (a[0]+self.R[0,nx,ny])**a[2]*(a[1]+self.R[1,nx,ny])**a[3],-d,nx,ny)for dx,dy,d in self.off if d<=a[4]and 0<=(nx:=a[5]+dx)<self.width and 0<=(ny:=a[6]+dy)<self.height and(nx,ny)not in self.occupancy),default=None):
                self.occupancy.pop((a[5],a[6]),None); a[5],a[6]=c[2],c[3]; self.occupancy[(a[5],a[6])]=a
            r=self.R[:,a[5],a[6]]; a[0]+=r[0]-a[2]; a[1]+=r[1]-a[3]; r[:] = 0
            if a[0]<=0 or a[1]<=0: a[7]=False; self.occupancy.pop((a[5],a[6]),None)
        if self.enable_trade:
            for a in [a for a in living if a[7]]:
                for dx,dy,d in self.off:
                    if d<=a[4] and (o:=self.occupancy.get((a[5]+dx,a[6]+dy))):
                        while True:
                            ma,mo=a[1]/a[0]*a[2]/a[3],o[1]/o[0]*o[2]/o[3]; p=(ma*mo)**.5; s,b=(a,o) if ma>mo else (o,a); sa,spa=(1,int(p)) if p>=1 else (int(1/p),1)
                            if s[1]<=spa or b[0]<=sa or (s[0]+sa)**s[2]*(s[1]-spa)**s[3]<=s[0]**s[2]*s[1]**s[3] or (b[0]-sa)**b[2]*(b[1]+spa)**b[3]<=b[0]**b[2]*b[1]**b[3]: break
                            s[0]+=sa; b[0]-=sa; s[1]-=spa; b[1]+=spa; self.all_trade_prices.append(p); self.total_trade_volume+=1

def create_model(seed=42,**params): return SugarscapeModel(seed=seed,**params)
def run_model(m,steps=200):
    for _ in range(steps): m.step()
    l=[a for a in m.agents if a[7]]
    return {"agent_wealths":[a[0]+a[1] for a in l], "final_population":len(l), "initial_population":m.initial_population, "trade_prices":m.all_trade_prices, "trade_volume":m.total_trade_volume, "agent_positions":[(a[5],a[6]) for a in l], "grid_width":m.width, "grid_height":m.height, "steps_run":steps}