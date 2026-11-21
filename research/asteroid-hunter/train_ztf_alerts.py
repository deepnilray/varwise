"""Real-data ZTF alert CNN baseline; this is NOT full-image asteroid discovery."""
from pathlib import Path
from collections import Counter
import json,random,hashlib
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,average_precision_score,confusion_matrix
from scipy.special import softmax,log_softmax
from scipy.optimize import minimize_scalar
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
SEED=137
LABELS=['AGN','SN','VS','asteroid','bogus']
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
root=Path(snapshot_download(repo_id='AnonymousUser16384/AstroAlertBench',repo_type='dataset',allow_patterns=['stamps_llm_updated/**','README.md'],local_dir='ztf_alert_data'))
rows=[]
for c in LABELS:
 for p in (root/'stamps_llm_updated'/c).glob('*/montage.png'):
  rows.append((str(p),LABELS.index(c),p.parent.name,hashlib.sha256(p.read_bytes()).hexdigest()))
if len(rows)<1000:raise RuntimeError('Too few real images: '+str(len(rows)))
if len(set(r[3] for r in rows))!=len(rows):raise RuntimeError('Duplicated montage images')
idx=np.arange(len(rows));y=np.array([r[1] for r in rows]);groups=np.array([r[2] for r in rows])
trv,te=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(idx,y,groups))
a,b=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1).split(trv,y[trv],groups[trv]))
splits={'train':trv[a],'validation':trv[b],'test':te}
for k,v in splits.items():
 if len(set(y[v]))!=len(LABELS) or min(Counter(y[v]).values())<3:raise RuntimeError('Unbalanced split '+k)
for x in splits:
 for z in splits:
  if x!=z and set(groups[splits[x]])&set(groups[splits[z]]):raise RuntimeError('OID leakage')
output=Path('asteroid_train_results');output.mkdir(exist_ok=True)
(output/'training_manifest.json').write_text(json.dumps({'source':'https://huggingface.co/datasets/AnonymousUser16384/AstroAlertBench','scope':'Real ZTF/ALeRCE pre-detected alert cutout montages ONLY','classes':LABELS,'seed':SEED,'total':len(rows),'split_kind':'oid grouped; not sky-disjoint','split_membership':{k:[{'oid':rows[int(i)][2],'label':LABELS[rows[int(i)][1]],'sha256':rows[int(i)][3]} for i in v] for k,v in splits.items()},'torch_version':torch.__version__},indent=2))
class Data(Dataset):
 def __init__(self,ids):self.ids=ids
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  p,c,_,_=rows[int(self.ids[i])]
  with Image.open(p) as im:img=np.asarray(im.convert('RGB').resize((192,96)),dtype=np.float32)/255.
  return torch.from_numpy(img.transpose(2,0,1).copy()),c
load={k:DataLoader(Data(v),batch_size=48,shuffle=k=='train') for k,v in splits.items()}
class Net(nn.Module):
 def __init__(self):
  super().__init__();layers=[];channels=[3,16,32,64]
  for a,b in zip(channels[:-1],channels[1:]):
   layers.extend([nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2)])
  self.features=nn.Sequential(*layers)
  self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,5))
 def forward(self,x):return self.head(self.features(x))
model=Net();optim=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.0001)
def predict(name):
 model.eval();truth=[];logits=[]
 with torch.no_grad():
  for im,lab in load[name]:
   logits.append(model(im).numpy());truth.extend(lab.tolist())
 return np.array(truth),np.concatenate(logits)
best=-1.;stale=0;history=[]
for ep in range(24):
 model.train();losses=0.
 for im,lab in load['train']:
  z=model(im);loss=nn.functional.cross_entropy(z,lab)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  optim.zero_grad();loss.backward();optim.step();losses+=float(loss.detach())*len(im)
 truth,lg=predict('validation')
 f=float(f1_score(truth,lg.argmax(1),average='macro',zero_division=0))
 history.append({'epoch':ep+1,'val_macro_f1':f,'train_loss_sum':losses})
 print('epoch',ep+1,'val_macro_f1',f,flush=True)
 if f>best+1.e-6:
  best=f;stale=0;torch.save(model.state_dict(),output/'model.pt')
 else:
  stale+=1
  if stale>=7:break
model.load_state_dict(torch.load(output/'model.pt',map_location='cpu',weights_only=True));model.eval()
truth,logits=predict('validation')
t=float(minimize_scalar(lambda x:-float(log_softmax(logits/x,axis=1)[np.arange(len(truth)),truth].mean()),bounds=(.25,8),method='bounded').x)
truth,logits=predict('test');prob=softmax(logits/t,axis=1)
report={'source':'ZTF AstroAlertBench real alert cutouts, NOT full survey images','n_real_images':len(rows),'test_macro_f1':float(f1_score(truth,prob.argmax(1),average='macro',zero_division=0)),'test_asteroid_average_precision':float(average_precision_score(truth==3,prob[:,3])),'test_confusion_matrix':confusion_matrix(truth,prob.argmax(1),labels=list(range(5))).tolist(),'calibration_temperature':t,'history':history,'limitations':'No full frame detection, orbit validation, sky/time-disjoint test, or DECam transfer evaluation; not discovery ready'}
(output/'metrics.json').write_text(json.dumps(report,indent=2))
torch.jit.trace(model,torch.zeros(1,3,96,192)).save(str(output/'model_cpu.ts'))
(output/'MODEL_CARD.md').write_text('# ZTF alert triage model\n\nResearch only. Trained on real ZTF-derived AstroAlertBench montages, not on DECam/IASC. It classifies EXISTING alerts, not unreported asteroids. Do not submit to MPC.\n')
print('FINAL_METRICS',json.dumps({'images':len(rows),'test_macro_f1':report['test_macro_f1'],'asteroid_ap':report['test_asteroid_average_precision']}),flush=True)
