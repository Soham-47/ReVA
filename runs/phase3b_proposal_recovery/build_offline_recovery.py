"""Offline connected-component and contextual-crop audit using Phase 3A caches."""
import csv, io, json, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/"runs/phase3a_retrieval_smoke"
FRAMES=RUN/"review/manual_audit/source_frames"
OUT=ROOT/"runs/phase3b_proposal_recovery"
OUT.mkdir(parents=True,exist_ok=True)
SHEETS=OUT/"sheets"; SHEETS.mkdir(exist_ok=True)
policy=json.loads((RUN/"manifest.json").read_text())["source_policies"]
specs=[("train_010397",10,"black_car_qcond"),("train_010397",0,"vehicle_fixed"),
       ("train_010397",11,"parking_lot_qcond"),("train_009869",15,"baseball_field_qcond"),
       ("train_009869",12,"field_fixed")]
font=ImageFont.load_default()

def frame(qa,i): return Image.open(FRAMES/qa/f"frame-{i+1:02d}.png").convert("RGB")
def masks(qa,c):
    with zipfile.ZipFile(RUN/"replicate"/qa/f"call-{c:02d}.masks.zip") as z:
        return [np.asarray(Image.open(io.BytesIO(z.read(n))).convert("L"))>0 for n in sorted(n for n in z.namelist() if n.endswith(".png"))]
def bbox(a):
    y,x=np.nonzero(a)
    return None if not len(x) else (int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1)
def cropbox(b,p):
    l,t,r,bm=b; dx=round((r-l)*p);dy=round((bm-t)*p)
    return max(0,l-dx),max(0,t-dy),min(640,r+dx),min(360,bm+dy)

def main():
    comps=[]; frames=[]
    for qa,call,name in specs:
        ms=masks(qa,call); sheet=Image.new("RGB",(2560,1560),"white")
        for i in range(16):
            src=policy[qa]["source_frame_indices"][i]; ts=policy[qa]["source_timestamps"][i]; im=frame(qa,i)
            if i<len(ms):
                m=ms[i]; assert m.shape==(360,640)
                a=np.asarray(im).copy();a[m]=(0.58*a[m]+0.42*np.array([250,45,45])).astype(np.uint8);im=Image.fromarray(a)
                lab,n=ndimage.label(m,structure=np.ones((3,3),dtype=np.uint8)); boxes=ndimage.find_objects(lab); areas=np.bincount(lab.ravel())[1:]
                frames.append({"mask":name,"qa_id":qa,"call":call,"packed_frame":i,"source_frame":src,"timestamp_s":round(ts,6),"area_px":int(m.sum()),"area_fraction":round(float(m.mean()),6),"components_8conn":int(n),"union_bbox_xyxy":bbox(m),"png_values":[0,255]})
                for cid,sl in enumerate(boxes,1):
                    if sl is None: continue
                    sy,sx=sl; box=(sx.start,sy.start,sx.stop,sy.stop); w=box[2]-box[0];h=box[3]-box[1]
                    comps.append({"mask":name,"qa_id":qa,"call":call,"packed_frame":i,"source_frame":src,"timestamp_s":round(ts,6),"component_id_frame_local":cid,"bbox_xyxy":box,"width":w,"height":h,"area_px":int(areas[cid-1]),"area_fraction":round(float(areas[cid-1])/(640*360),7),"aspect_ratio":round(w/h,4)})
                    if areas[cid-1]>=8:
                        dr=ImageDraw.Draw(im);dr.rectangle((box[0],box[1],box[2]-1,box[3]-1),outline=(20,255,35),width=1);dr.text((box[0],box[1]),str(cid),font=font,fill=(255,255,0),stroke_width=1,stroke_fill="black")
                status=f"{n} components (8-connected)"
            else: status="NO MASK"
            x=(i%4)*640;y=(i//4)*390; panel=Image.new("RGB",(640,390),"white");panel.paste(im,(0,25))
            ImageDraw.Draw(panel).text((4,4),f"{name} f{i} src={src} t={ts:.3f}s {status}",font=font,fill="black");sheet.paste(panel,(x,y))
        sheet.save(SHEETS/f"{name}_components.jpg",quality=58,optimize=True)

        # Representative native-resolution RGB crops: tight and 20%-context boxes.
        if name in ("parking_lot_qcond","baseball_field_qcond","field_fixed"):
            crop_sheet=Image.new("RGB",(1280,780),"white")
            for row,(i,pad,label) in enumerate(((0,0,"tight"),(0,.2,"context20"),(8,0,"tight"),(8,.2,"context20"))):
                if i>=len(ms):continue
                b=bbox(ms[i]); cb=cropbox(b,pad); crop=frame(qa,i).crop(cb)
                x=(row%2)*640;y=(row//2)*390; panel=Image.new("RGB",(640,390),"white");panel.paste(crop,(0,25))
                src=policy[qa]["source_frame_indices"][i]; ts=policy[qa]["source_timestamps"][i]
                ImageDraw.Draw(panel).text((4,4),f"{name} {label} src={src} t={ts:.3f}s bbox={cb}",font=font,fill="black");crop_sheet.paste(panel,(x,y))
            crop_sheet.save(SHEETS/f"{name}_native_crops.jpg",quality=65,optimize=True)
    for path,rows in (("components.csv",comps),("frames.csv",frames)):
        with (OUT/path).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"scope":"offline cached Phase 3A data only","models_called":0,"dimensions_px":[640,360],"connectivity":8,"components_total":len(comps),"components_by_mask":{n:sum(x["mask"]==n for x in comps) for _,_,n in specs},"component_counts_per_frame":{n:[x["components_8conn"] for x in frames if x["mask"]==n] for _,_,n in specs},"frame_counts_by_mask":{n:sum(x["mask"]==n for x in frames) for _,_,n in specs},"mask_values":"Inspected archive PNGs are binary 0/255; local reader thresholds L-mode values >0.","mapping":"Packed frames use manifest source_frame_indices and source_timestamps.","component_ids":"Frame-local connected-component labels; not SAM3 native IDs or cross-frame tracks.","manual_annotations_used_to_generate_candidates":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    assert all(r["source_frame"]==policy[r["qa_id"]]["source_frame_indices"][r["packed_frame"]] for r in comps)
    assert all(0<=r["bbox_xyxy"][0]<r["bbox_xyxy"][2]<=640 and 0<=r["bbox_xyxy"][1]<r["bbox_xyxy"][3]<=360 for r in comps)
    assert len(frames)==sum(len(masks(q,c)) for q,c,_ in specs)
if __name__=="__main__":main()
