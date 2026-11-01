import collections, csv, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
root=Path(os.environ['WORK'])
sys.path.insert(0,str(root/'code'))
import common as C
out=root/'results/error_review'
out.mkdir(parents=True,exist_ok=True)
run=root/'runs/stageA_q3vl8b_r16'
report={}
selected=[]
for split in ['valQ','valT','val']:
    rows=list(csv.DictReader((run/f'scores_{split}_eval.csv').open()))
    scores={s['qa_id']:s for s in map(json.loads,(run/f'scores_{split}.jsonl').open())}
    errors=[r for r in rows if r['ok']=='False']
    def pred(v): return C.LETTERS[int(np.argmax(v))]
    changed=[r for r in rows if r['pred']!=r['pred_single']]
    stable=[r for r in errors if len({pred(v['p']) for v in scores[r['qa_id']]['views']})==1]
    confident=[r for r in errors if float(r['conf'])>=.9]
    report[split]={'n':len(rows),'errors':len(errors),'correct':len(rows)-len(errors),'tta_rescued':sum(r['ok']=='True' for r in changed),'tta_harmed':sum(r['ok']=='False' and r['ok_single']=='True' for r in changed),'errors_all_views_agree':len(stable),'errors_confidence_ge_90':len(confident),'mean_confidence_correct':float(np.mean([float(r['conf']) for r in rows if r['ok']=='True'])),'mean_confidence_wrong':float(np.mean([float(r['conf']) for r in errors])),'errors_by_task':dict(collections.Counter(r['task'] for r in errors))}
    if split=='valQ':
        for task,count in [('Temporal Grounding',4),('Geometric Relation',3),('Change Detection',2),('Trend and Pattern',2),('Structural Layout',1)]:
            selected.extend(sorted([r for r in errors if r['task']==task],key=lambda r:float(r['conf']),reverse=True)[:count])
        (out/'valQ_errors.json').write_text(json.dumps(errors,indent=2))
manifest=C.load_manifest()
fit=json.loads((root/'data/splits/fit_A.json').read_text())['QA']
valq=json.loads((root/'data/splits/valQ.json').read_text())['QA']
valt=json.loads((root/'data/splits/valT.json').read_text())['QA']
report['integrity']={'fit_qa_overlap_valQ':len({r['qa_id'] for r in fit}&{r['qa_id'] for r in valq}),'fit_qa_overlap_valT':len({r['qa_id'] for r in fit}&{r['qa_id'] for r in valt}),'fit_video_overlap_valT':len({r['video_path'] for r in fit}&{r['video_path'] for r in valt}),'frame_cap':C.MAX_FRAMES,'target_fps':C.FPS_TARGET}
for row in selected:
    info=manifest[row['video']]
    row['metadata']=info
    row['indices']={}
    for offset in [0.,.5]:
        inds=C.sample_index_list(info['n_frames'],info['fps'],offset)
        row['indices'][str(offset)]=inds
        frames=C.read_frames(row['video'],inds)
        assert len(frames)==len(inds)
        board=Image.new('RGB',(1280,4*210+45),'white'); draw=ImageDraw.Draw(board)
        draw.text((8,8),f"{row['qa_id']} | {row['task']} | gold {row['gold']} / pred {row['pred']} | offset {offset}",fill='black')
        for i,(idx,frame) in enumerate(zip(inds,frames)):
            im=Image.fromarray(frame); im.thumbnail((320,180))
            x=(i%4)*320; y=45+(i//4)*210
            board.paste(im,(x,y)); draw.text((x+3,y+182),f'frame {idx}, {idx/info["fps"]:.2f}s',fill='black')
        board.save(out/f"{row['qa_id']}_offset{offset}.jpg",quality=88)
    row['last_frame_gap_seconds']=(info['n_frames']-1-row['indices']['0.5'][-1])/info['fps']
    print(row['qa_id'],row['question'],flush=True)
(out/'selected_examples.json').write_text(json.dumps(selected,indent=2))
(out/'audit_statistics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
