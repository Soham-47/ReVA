import json
import sys
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'runs/phase3a_retrieval_smoke'
OUT = RUN / 'review/manual_audit'
sys.path.insert(0, str(ROOT / 'code'))
import replicate_sam3 as R

manifest = json.loads((RUN / 'manifest.json').read_text())
report = json.loads((RUN / 'hpc_report.json').read_text())
font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
small = ImageFont.truetype(font_file, 12)
rank_map = {}
for row in report['rankings']:
    key = (row['qa_id'], row['phrase'], row['arm'], row['crop_style'])
    for rank, candidate in enumerate(row['siglip_top5'], 1):
        rank_map.setdefault(candidate['candidate_id'], {})[key] = rank
frames = {}
for qa, policy in manifest['source_policies'].items():
    files = sorted((OUT / 'source_frames' / qa).glob('frame-*.png'))
    assert len(files) == len(policy['source_frame_indices']) == 16
    frames[qa] = {int(idx): Image.open(file).convert('RGB') for idx, file in zip(policy['source_frame_indices'], files)}
questions = {'train_010397': ['black car', 'parking lot'], 'train_009869': ['baseball field']}

for call, record in enumerate(manifest['records']):
    if record['arm'] == 'tile-probe' or record['status'] != 'complete' or not record['mask_count']:
        continue
    qa = record['qa_id']; policy = manifest['source_policies'][qa]
    entries = R.read_mask_zip((RUN / record['zip']).read_bytes())
    by_packed = {int(e['frame_index']): (ordinal, e) for ordinal, e in enumerate(entries)}
    assert len(by_packed) == record['mask_count']
    sheet = Image.new('RGB', (4 * 640, 4 * 400), (245, 245, 245))
    draw_sheet = ImageDraw.Draw(sheet)
    for packed, source in enumerate(policy['source_frame_indices']):
        x, y = (packed % 4) * 640, (packed // 4) * 400
        frame = np.asarray(frames[qa][int(source)]).copy()
        timestamp = policy['source_timestamps'][packed]
        if packed in by_packed:
            ordinal, entry = by_packed[packed]
            with Image.open(BytesIO(entry['payload'])) as im:
                mask = np.asarray(im.convert('L')) > 0
            assert mask.shape == frame.shape[:2]
            frame[mask] = (0.58 * frame[mask] + 0.42 * np.array([245, 50, 45])).astype('uint8')
            inner = mask.copy()
            inner[1:] &= mask[:-1]; inner[:-1] &= mask[1:]
            inner[:, 1:] &= mask[:, :-1]; inner[:, :-1] &= mask[:, 1:]
            edge = mask & ~inner
            ys, xs = np.nonzero(edge)
            frame[ys, xs] = (255, 230, 35)
            candidate_id = f"{qa}-call{call:02d}-frame{packed:02d}-mask{ordinal:04d}"
            rank_labels = []
            for phrase in questions[qa]:
                for style, letter in [('padded_rgb_box', 'P'), ('masked_rgb', 'M')]:
                    rank = rank_map.get(candidate_id, {}).get((qa, phrase, record['arm'], style))
                    if rank:
                        rank_labels.append(f'{phrase} {letter}#{rank}')
            rank_text = ' '.join(rank_labels) if rank_labels else 'outside top-5'
            jshort = []
            for phrase in questions[qa]:
                if phrase == 'black car': verdict = 'BC=Y' if call in (0,10) else 'BC=N'
                elif phrase == 'parking lot': verdict = 'PL=Y' if call == 11 else ('PL=P' if call == 2 else 'PL=N')
                else: verdict = 'BF=Y' if call in (12,15) and int(source) <= 78 else 'BF=N'
                jshort.append(verdict)
            label = f"call{call:02d} {record['prompt']} | f{packed:02d} src={source} t={timestamp:.2f}s"
            label += f"\n{rank_text} | {' '.join(jshort)}"
            color = 'black'
        else:
            label = f"{record['arm']}:{record['prompt']} | src={source} t={timestamp:.2f}s | NO MASK"
            color = '#b00020'
        panel = Image.new('RGB', (640, 400), 'white')
        panel.paste(Image.fromarray(frame), (0, 40))
        ImageDraw.Draw(panel).multiline_text((4, 2), label, font=small, fill=color, spacing=1)
        sheet.paste(panel, (x, y))
    fname = f"proposals_{qa}_call{call:02d}_{record['prompt'].replace(' ', '_')}.png"
    sheet.save(OUT / fname, optimize=True)

crop_dir = OUT / 'candidate_crops'; crop_dir.mkdir(exist_ok=True)
for phrase in questions['train_010397'] + questions['train_009869']:
    cards = []
    for row in (r for r in report['rankings'] if r['phrase'] == phrase):
        qa = row['qa_id']; policy = manifest['source_policies'][qa]
        for style in ('padded_rgb_box', 'masked_rgb'):
            candidates = report['examples'][f"{qa}|{row['arm']}|{phrase}|{style}"]
            for rank, c in enumerate(candidates, 1):
                call = int(c['candidate_id'].split('-call')[1][:2])
                packed = int(c['candidate_id'].split('-frame')[1][:2])
                ordinal = int(c['candidate_id'].split('-mask')[1])
                entries = R.read_mask_zip((RUN / manifest['records'][call]['zip']).read_bytes())
                entry = entries[ordinal]
                assert int(entry['frame_index']) == packed
                with Image.open(BytesIO(entry['payload'])) as im:
                    mask = np.asarray(im.convert('L')) > 0
                frame = np.asarray(frames[qa][int(c['source_frame_index'])])
                box = c['crop_box_pixels']; l, t, r, b = (int(box[k]) for k in ('left','top','right','bottom'))
                crop = frame[t:b, l:r].copy(); local_mask = mask[t:b, l:r]
                if style == 'masked_rgb': crop[~local_mask] = 0
                crop_image = Image.fromarray(crop)
                crop_image.save(crop_dir / f"{c['candidate_id']}_{style}.png")
                cards.append((f"{qa} {row['arm']} {style} r{rank} sim={c['similarity']:.4f} src={c['source_frame_index']} t={c['timestamp']:.2f}s prompt={manifest['records'][call]['prompt']} native=unavailable", crop_image.copy()))
    sheet = Image.new('RGB', (4 * 640, 5 * 400), 'white')
    for i, (label, crop) in enumerate(cards[:20]):
        x, y = (i % 4) * 640, (i // 4) * 400
        ImageDraw.Draw(sheet).text((x + 4, y + 4), label[:110], font=small, fill='black')
        crop.thumbnail((632, 356), Image.Resampling.LANCZOS)
        sheet.paste(crop, (x + 4, y + 34))
    sheet.save(OUT / f"ranked_crops_{phrase.replace(' ', '_')}.png", optimize=True)
print('proposal sheets', len(list(OUT.glob('proposals_*.png'))))
print('ranked crop sheets', len(list(OUT.glob('ranked_crops_*.png'))))
print('native source crop assets', len(list(crop_dir.glob('*.png'))))
