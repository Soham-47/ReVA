import json,sys
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[4]; RUN=ROOT/"runs/phase3a_retrieval_smoke"; OUT=RUN/"review/manual_audit"
sys.path.insert(0,str(ROOT/"code")); import replicate_sam3 as R
m=json.loads((RUN/"manifest.json").read_text()); d=json.loads((RUN/"hpc_report.json").read_text()); font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
frames={}
for qa,p in m["source_policies"].items():
 fs=sorted((OUT/"source_frames"/qa).glob("frame-*.png")); frames[qa]={int(idx):Image.open(f).convert("RGB") for idx,f in zip(p["source_frame_indices"],fs)}
entries_by_call={}; cropdir=OUT/"candidate_crops"; cropdir.mkdir(exist_ok=True)
for phrase in ["black car","parking lot","baseball field"]:
 cards=[]
 for row in (r for r in d["rankings"] if r["phrase"]==phrase):
  qa=row["qa_id"]
  for style in ("padded_rgb_box","masked_rgb"):
   cs=d["examples"][f"{qa}|{row['arm']}|{phrase}|{style}"]
   for rank,c in enumerate(cs,1):
    cid=c["candidate_id"]; call=int(cid.split("-call")[1][:2]); packed=int(cid.split("-frame")[1][:2]); ordinal=int(cid.split("-mask")[1])
    if call not in entries_by_call: entries_by_call[call]=R.read_mask_zip((RUN/m["records"][call]["zip"]).read_bytes())
    entry=entries_by_call[call][ordinal]; assert int(entry["frame_index"])==packed
    with Image.open(BytesIO(entry["payload"])) as im: mask=np.asarray(im.convert("L"))>0
    frame=np.asarray(frames[qa][int(c["source_frame_index"])])
    b=c["crop_box_pixels"]; l,t,r,bb=(int(b[k]) for k in ("left","top","right","bottom")); crop=frame[t:bb,l:r].copy()
    if style=="masked_rgb": crop[~mask[t:bb,l:r]]=0
    img=Image.fromarray(crop); img.save(cropdir/f"{cid}_{style}.png")
    
    if phrase == "black car": judgment = "TARGET IN UNION; NOT INSTANCE-PURE" if call in (0,10) else "NO TARGET CAR"
    elif phrase == "parking lot": judgment = "LOT OVERLAP; ROAD SPILL" if call==11 else ("ROAD; PARTIAL LOT OVERLAP" if call==2 else "NO LOT TARGET")
    else: judgment = "TARGET FIELD; PARTIAL / FRAGMENTED" if call in (12,15) and int(c["source_frame_index"])<=78 else "WRONG FIELD / GREEN AREA"
    line1=f"{judgment} | r{rank} | sim={c['similarity']:.4f} | {row['arm']} | {style}"
    line2=f"src={c['source_frame_index']} t={c['timestamp']:.2f}s | proposal={m['records'][call]['prompt']} | crop={img.width}x{img.height}"
    cards.append((f"{line1}\n{line2}",img.copy()))
 sheet=Image.new("RGB",(4*640,5*410),"white")
 for i,(label,img) in enumerate(cards[:20]):
  x,y=(i%4)*640,(i//4)*410; ImageDraw.Draw(sheet).multiline_text((x+4,y+3),label,font=font,fill="black",spacing=2); img.thumbnail((632,356),Image.Resampling.LANCZOS); sheet.paste(img,(x+4,y+50))
 path=OUT/f"ranked_crops_{phrase.replace(' ','_')}.png"; sheet.save(path,optimize=True); print(phrase,len(cards),path.stat().st_size)
print("crop_assets",len(list(cropdir.glob("*.png"))))
