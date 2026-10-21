import csv,json,os,sys,math,collections
from pathlib import Path
import av
from PIL import Image
root=Path(os.environ['WORK']); out=root/'results/error_review'; run=root/'runs/stageA_q3vl8b_r16'
selected=json.loads((out/'selected_examples.json').read_text()); stats=json.loads((out/'audit_statistics.json').read_text())
for split in ['valQ','valT','val']:
 rows=list(csv.DictReader((run/f'scores_{split}_eval.csv').open()))
 stats[split]['tta_harmed']=sum(r['ok']=='False' and r['ok_single']=='True' for r in rows)
 scores=list(map(json.loads,(run/f'scores_{split}.jsonl').open()))
 assert len(scores)==len(rows)
 assert all(len(s['views'])==4 and s['pred']=='ABCD'[max(range(4),key=lambda i:s['p'][i])] for s in scores)
 assert all(all(math.isfinite(v) and 0<=v<=1 for v in s['p']) and abs(sum(s['p'])-1)<1e-5 for s in scores)
 assert all(max(abs(s['p'][i]-sum(v['p'][i] for v in s['views'])/4) for i in range(4))<1e-6 for s in scores)
 stats[split]['aggregation_valid']=True
fit=json.loads((root/'data/splits/fit_A.json').read_text())['QA']; valq=json.loads((root/'data/splits/valQ.json').read_text())['QA']
keys=lambda rows:{(r['video_path'],r['question']) for r in rows}
stats['integrity']['fit_question_group_overlap_valQ']=len(keys(fit)&keys(valq))
checks={}
for r in selected:
 if r['video'] in checks: continue
 with av.open(str(root/'data/ReVA'/r['video'])) as container:
  st=container.streams.video[0]; st.thread_type='AUTO'; first=None; last=None; count=0; max_drift=0
  for i,f in enumerate(container.decode(st)):
   if first is None: first=f.to_image(); first_pts=float(f.time or 0)
   last=f.to_image(); last_pts=float(f.time or 0); count+=1
   max_drift=max(max_drift,abs(last_pts-first_pts-i/r['metadata']['fps']))
  first.save(out/(r['qa_id']+'_first.jpg')); last.save(out/(r['qa_id']+'_last.jpg'))
  checks[r['video']]={'manifest_frames':r['metadata']['n_frames'],'decoded_frames':count,'first_pts':first_pts,'last_pts':last_pts,'max_timestamp_drift_seconds':max_drift}
  print(r['qa_id'],checks[r['video']],flush=True)
stats['decode_checks']=checks
(out/'audit_statistics.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
