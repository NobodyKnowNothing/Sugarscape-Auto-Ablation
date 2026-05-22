import math, numpy as np
from pathlib import Path

class SugarscapeModel:
    def __init__(self,seed,width=50,height=50,initial_population=200,endowment_min=25,endowment_max=50,metabolism_min=1,metabolism_max=5,vision_min=1,vision_max=5,enable_trade=True,**kwargs):
        self.rng,self.width,self.height,self.initial_population,self.enable_trade=np.random.default_rng(seed),width,height,initial_population,enable_trade
        m=np.genfromtxt(Path(__file__).parent/"sugar-map.txt"); self.R=np.stack([m, np.flip(m,1)]); self.M=self.R.copy(); self.occupancy,self.agents,self.all_trade_prices,self.total_trade_volume={},[],[],0
        for _ in range(initial_population):
            x,y=self.rng.integers(0,width),self.rng.integers(0,height)
            while (x,y) in self.occupancy: x,y=self.rng.integers(0,width),self.rng.integers(0,height)
            a=[self.rng.integers(endowment_min,endowment_max+1),self.rng.integers(endowment_min,endowment_max+1),self.rng.integers(metabolism_min,metabolism_max+1),self.rng.integers(metabolism_min,metabolism_max+1),self.rng.integers(vision_min,vision_max+1),x,y,True]
            self.agents.append(a); self.occupancy[(x,y)]=a
    def step(self):
        self.R=np.minimum(self.R+1,self.M); living=[a for a in self.agents if a[7]]; self.rng.shuffle(living)
        for a in living:
            m_t=a[2]+a[3]; c=[((a[0]+self.R[0,nx,ny])**(a[2]/m_t)*(a[1]+self.R[1,nx,ny])**(a[3]/m_t),-(abs(dx)+abs(dy)),self.rng.random(),nx,ny) for dx in range(-a[4],a[4]+1) for dy in range(-a[4],a[4]+1) if 0<abs(dx)+abs(dy)<=a[4] and 0<=(nx:=a[5]+dx)<self.width and 0<=(ny:=a[6]+dy)<self.height and (nx,ny) not in self.occupancy]
            if c: _,_,_,nx,ny=max(c); self.occupancy.pop((a[5],a[6]),None); a[5],a[6]=nx,ny; self.occupancy[(nx,ny)]=a
            a[0]+=self.R[0,a[5],a[6]]-a[2]; a[1]+=self.R[1,a[5],a[6]]-a[3]; self.R[0,a[5],a[6]]=self.R[1,a[5],a[6]]=0
            if a[0]<=0 or a[1]<=0: a[7]=False; self.occupancy.pop((a[5],a[6]),None)
        if self.enable_trade:
            living=[a for a in self.agents if a[7]]; self.rng.shuffle(living)
            for a in living:
                for nx,ny in [(a[5]+dx,a[6]+dy) for dx in range(-a[4],a[4]+1) for dy in range(-a[4],a[4]+1) if abs(dx)+abs(dy)<=a[4]]:
                    if 0<=nx<self.width and 0<=ny<self.height and (o:=self.occupancy.get((nx,ny))) and o[7] and o!=a:
                        while a[0]>0 and a[1]>0 and o[0]>0 and o[1]>0:
                            mrs_a,mrs_o=(a[1]/a[3])/(a[0]/a[2]),(o[1]/o[3])/(o[0]/o[2]); p=math.sqrt(mrs_a*mrs_o); s,b=(a,o) if mrs_a>mrs_o else (o,a); m_ts,m_tb=s[2]+s[3],b[2]+b[3]; sa,spa=(1,int(p)) if p>=1 else (int(1/p),1); u_s,u_b=s[0]**(s[2]/m_ts)*s[1]**(s[3]/m_ts), b[0]**(b[2]/m_tb)*b[1]**(b[3]/m_tb)
                            if mrs_a<=0 or mrs_o<=0 or abs(mrs_a-mrs_o)<1e-5 or s[1]<=spa or b[0]<=sa or (s[0]+sa)**(s[2]/m_ts)*(s[1]-spa)**(s[3]/m_ts)<=u_s or (b[0]-sa)**(b[2]/m_tb)*(b[1]+spa)**(b[3]/m_tb)<=u_b: break
                            s[0]+=sa; b[0]-=sa; s[1]-=spa; b[1]+=spa; self.all_trade_prices.append(p); self.total_trade_volume+=1

def create_model(seed=42,**params): return SugarscapeModel(seed=seed,**params)
def run_model(model,steps=200):
    for _ in range(steps): model.step()
    living=[a for a in model.agents if a[7]]
    return {"agent_wealths":[a[0]+a[1] for a in living], "final_population":len(living), "initial_population":model.initial_population, "trade_prices":model.all_trade_prices, "trade_volume":model.total_trade_volume, "agent_positions":[(a[5],a[6]) for a in living], "grid_width":model.width, "grid_height":model.height, "steps_run":steps}