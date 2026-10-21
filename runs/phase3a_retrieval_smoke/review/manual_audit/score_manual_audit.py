"""Create a blinded, source-frame-based annotation of existing Phase 3A outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'runs/phase3a_retrieval_smoke'
OUT = RUN / 'review/manual_audit'
manifest = json.loads((RUN / 'manifest.json').read_text())
report = json.loads((RUN / 'hpc_report.json').read_text())

# Visual labels are applied to proposal groups after reviewing all source-resolution
# overlays. A positive retrieval judgment means the proposal overlaps the requested
# target in that frame; it does not mean the mask is pure or complete.
def label(phrase, candidate):
    cid = candidate['candidate_id']
    call = int(cid.split('-call')[1][:2])
    source = int(candidate['source_frame_index'])
    if phrase == 'black car':
        if call in (0, 10):
            return {
                'target_overlap': 'yes', 'category': 'yes', 'instance_presence': 'yes',
                'instance_isolated': 'no', 'mask_completeness': 'mostly complete for car pixels',
                'oversegmentation': 'severe: union of many parked cars',
                'wrong_instance': 'many other cars are included', 'confidence': 'medium',
            }
        return {'target_overlap':'no','category':'no','instance_presence':'no','instance_isolated':'no',
                'mask_completeness':'not applicable','oversegmentation':'not applicable',
                'wrong_instance':'not target object','confidence':'high'}
    if phrase == 'parking lot':
        if call == 11:
            return {'target_overlap':'yes','category':'yes','instance_presence':'yes',
                    'instance_isolated':'no','mask_completeness':'partial to good for paved lot surface',
                    'oversegmentation':'high: lot aisles and adjacent access roads merge',
                    'wrong_instance':'road and non-stall pavement included','confidence':'medium'}
        if call == 2:
            return {'target_overlap':'partial','category':'no (road proposal)',
                    'instance_presence':'partial','instance_isolated':'no',
                    'mask_completeness':'partial','oversegmentation':'moderate to high',
                    'wrong_instance':'road surface adjacent to / within lot','confidence':'medium'}
        return {'target_overlap':'no','category':'no','instance_presence':'no','instance_isolated':'no',
                'mask_completeness':'not applicable','oversegmentation':'not applicable',
                'wrong_instance':'not target region','confidence':'high'}
    if phrase == 'baseball field':
        if call in (12, 15) and source <= 78:
            return {'target_overlap':'yes','category':'yes','instance_presence':'yes',
                    'instance_isolated':'uncertain: multiple similar playing areas; no track IDs',
                    'mask_completeness':'partial; infield/outfield boundaries are incomplete or fragmented',
                    'oversegmentation':'high: adjacent grass / playing surfaces are included',
                    'wrong_instance':'possible adjacent field/grass; field identity unverified',
                    'confidence':'medium'}
        return {'target_overlap':'no','category':'no for target baseball field',
                'instance_presence':'no','instance_isolated':'uncertain',
                'mask_completeness':'not applicable','oversegmentation':'false positive on generic grass/courts',
                'wrong_instance':'generic field/court or other scene region','confidence':'medium'}
    raise ValueError(phrase)

frame_totals = {'black car':16, 'parking lot':16, 'baseball field':8}
annotations=[]; metrics=[]
for row in report['rankings']:
    phrase=row['phrase']; qa=row['qa_id']; denom=frame_totals[phrase]
    candidates=report['examples'][f"{qa}|{row['arm']}|{phrase}|{row['crop_style']}"]
    pool={c['candidate_id']:c for draw in row['random_top5'] for c in draw}
    assert len(pool)==row['candidate_count'],(row['qa_id'],phrase,row['arm'],len(pool),row['candidate_count'])
    is_positive=lambda c: label(phrase,c)['category']=='yes'
    positive_ids={cid for cid,c in pool.items() if is_positive(c)}
    assert len(positive_ids) == (denom if phrase!='parking lot' or row['arm']=='question-conditioned' else 0) or phrase=='parking lot'
    for cid,c in pool.items():
        annotations.append({'query_id':qa,'entity':phrase,'proposal_arm':row['arm'],
                            'candidate_id':cid,'proposal_prompt':c['proposal_prompt'],
                            'source_frame_index':c['source_frame_index'],'timestamp':c['timestamp'],
                            'siglip_rank_padded_rgb_box':next((i for i,x in enumerate(row['siglip_top5'],1) if x['candidate_id']==cid),None) if row['crop_style']=='padded_rgb_box' else None,
                            'siglip_rank_masked_rgb':next((i for i,x in enumerate(row['siglip_top5'],1) if x['candidate_id']==cid),None) if row['crop_style']=='masked_rgb' else None,
                            **label(phrase,c),
                            'cross_frame_identity':'not verified; candidates are independent frame masks'})
    def relevant_count(batch,k,overlap=False):
        return len({c['candidate_id'] for c in batch[:k] if (label(phrase,c)['target_overlap'] in ('yes','partial') if overlap else is_positive(c))})
    sig={}; rnd={}; overlap_sig={}; overlap_rnd={}
    for k in (1,5):
        count=relevant_count(candidates,k)
        sig[f'recall@{k}']=count/denom
        sig[f'hit@{k}']=bool(count)
        sig[f'precision@{k}']=count/k
        oc=relevant_count(candidates,k,overlap=True); overlap_sig[f'recall@{k}']=oc/denom; overlap_sig[f'hit@{k}']=bool(oc)
        draw_vals=[]
        for draw in row['random_top5']:
            n=relevant_count(draw,k)
            draw_vals.append((n/denom,float(bool(n)),n/k))
        rnd[f'recall@{k}']=sum(x[0] for x in draw_vals)/len(draw_vals)
        rnd[f'hit@{k}']=sum(x[1] for x in draw_vals)/len(draw_vals)
        rnd[f'precision@{k}']=sum(x[2] for x in draw_vals)/len(draw_vals)
        overlap_vals=[]
        for draw in row['random_top5']:
            n=relevant_count(draw,k,overlap=True); overlap_vals.append((n/denom,float(bool(n)),n/k))
        overlap_rnd[f'recall@{k}']=sum(x[0] for x in overlap_vals)/len(overlap_vals); overlap_rnd[f'hit@{k}']=sum(x[1] for x in overlap_vals)/len(overlap_vals)
    metrics.append({'query_id':qa,'entity':phrase,'proposal_arm':row['arm'],
                    'crop_style':row['crop_style'],'candidate_count':row['candidate_count'],
                    'manually_relevant_candidates':len(positive_ids),
                    'relevant_source_frames_in_video':denom,
                    'siglip2':sig,'random_1000_draw_mean':rnd,
                    'region_overlap_sensitivity':{'siglip2':overlap_sig,'random_1000_draw_mean':overlap_rnd},
                    'native_sam3_scores':'unavailable'})

# Proposal-level coverage judgments based on the inspected 16-frame source overlays.
coverage=[
 {'entity':'black car','target_visible_frames':16,'question_conditioned_masks':16,
  'qcond_target_overlap_frames':16,'fixed_vehicle_masks':16,'fixed_target_overlap_frames':16,
  'category':'car category is covered; masks merge multiple cars',
  'correct_instance':'target car is included in the group mask, but no candidate isolates it; 0/16 pure-instance proposals',
  'mask_quality':'tight-ish contours around individual cars but a frame-wide, disconnected multi-car union; padded bbox spans nearly the full frame',
  'cross_frame_identity':'not verified'},
 {'entity':'parking lot','target_visible_frames':16,'question_conditioned_masks':16,
  'qcond_target_overlap_frames':16,'fixed_literal_parking_lot_prompts':0,
  'fixed_road_masks':16,'fixed_target_overlap_frames':'partial in the road proposal; no exact parking-lot region',
  'category':'question-conditioned mask overlaps lot but merges lot aisles with roads; fixed bank has no parking-lot prompt',
  'correct_instance':'parking region overlaps target, but boundaries do not isolate the lot',
  'mask_quality':'broad connected pavement; mean area 11.1% of frame; spills to access roads',
  'cross_frame_identity':'not verified'},
 {'entity':'baseball field','target_visible_frames':8,'question_conditioned_masks':15,
  'qcond_target_overlap_frames':8,'fixed_field_masks':16,'fixed_target_overlap_frames':8,
  'category':'target field overlaps in the first 8 sampled frames; generic grass/court false positives afterward',
  'correct_instance':'per-frame target region is only partial; identity among similar fields not verified',
  'mask_quality':'question masks average 16.9% area (1.1–32.7%); fragmented / incomplete boundaries; fixed field masks average 63.6% area and spill widely',
  'cross_frame_identity':'not verified'}]

out={'schema_version':1,'scope':'blinded manual retrieval audit of existing two-video smoke outputs',
     'annotation_basis':'640x360 source frames and saved SAM3 masks; no gold/predicted QA answers used',
     'rubric':{'correct_category_retrieval_positive':'manual category match yes; partial overlap is excluded from strict Recall@1/5',
               'region_overlap_sensitivity':'also count partial overlap as relevant to show nearby-region retrieval separately',
               'recall_denominator':'visually target-present sampled frames (16 black-car, 16 parking-area, 8 baseball-field)',
               'candidate_ranks':'SigLIP2 top five from the frozen smoke output; random is the saved 1,000 seeded draws',
               'limitations':'qualitative human masks/region labels, no pixel-ground-truth masks; no SAM3 per-mask confidence; no cross-frame track IDs'},
     'proposal_coverage':coverage,'retrieval_metrics':metrics,
     'annotations':annotations,
     'native_score_status':'unavailable: Replicate ZIP outputs contain no attributable per-mask scores',
     'provider_reconciliation':{'client_prediction_attempts':manifest['replicate_attempts'],
      'planned_calls':20,'retries':manifest['replicate_retries'],'successful_record_count':sum(r['status']=='complete' for r in manifest['records']),
      'billable_predictions':'not verifiable from local artifacts; no provider prediction IDs or billing data were stored',
      'actual_cost_usd':None,'planning_estimate_usd_for_22_attempts':round(22*manifest['replicate_budget']['planning_cost_per_attempt_usd'],4),
      'billing_access':'Replicate API credential is not configured in this workspace; browser-harness account access was unavailable; exposed chat token was not reused'},
     'execution_provenance':{'SAM3':'Replicate-hosted model predictions (remote)',
       'SigLIP2_embeddings_and_rankings':'HPC H200 NVL; 92.072 GPU-seconds reported, 1.66 GiB CUDA allocated, 1.69 GiB reserved',
       'mask/source-frame decode and crop generation':'HPC H200 job as part of smoke preprocessing',
       'this_manual_audit':'local CPU/Pillow/ffmpeg only; no HPC, Modal, Replicate, or additional model inference'} }
path=OUT/'manual_audit.json'; path.write_text(json.dumps(out,indent=2)+'\n')
print('annotations',len(annotations),'retrieval rows',len(metrics),'report',path)
for m in metrics:
 print(m['entity'],m['proposal_arm'],m['crop_style'],'Sig R1/R5',round(m['siglip2']['recall@1'],3),round(m['siglip2']['recall@5'],3),'Rand R1/R5',round(m['random_1000_draw_mean']['recall@1'],3),round(m['random_1000_draw_mean']['recall@5'],3),'hit5',round(m['siglip2']['hit@5'],3),round(m['random_1000_draw_mean']['hit@5'],3))
