# SAM 3.1 Phase 3 API preflight

Status: blocked before API calls, uploads, installation, or checkpoint download.

## Access gate

Checks performed on 2026-10-16:

- No local `MODEL_API_KEY` was configured.
- Modal secret listing returned no configured secrets.
- No Meta API call was attempted.
- No SAM 3 checkpoint, dataset upload, or pilot output was created.
- The prior exploratory Modal initialization was stopped before any GPU function ran; existing SAM 2 artifacts are unchanged.

The only remaining blocker is securely configuring `MODEL_API_KEY` in the approved execution environment. Do not paste the key into chat. Once configured, use the standard/non-contributor mode; contributor mode is intentionally excluded because it sends input files and prompts for training.

## Fixed 15-video subset

The subset uses the existing approved 32-question manifest and local split checksums. It selects one question per unique video, all eight `Geometric Relation` questions, and seven `Temporal Grounding` questions. The temporal question `train_012498` is omitted because its video is already represented by geometric question `train_012502`.

`data/manifest.json` contains frame metadata for every selected video. The video assets themselves are not local (`find data -iname '*.mp4'` returned zero files), so upload bytes and encoded-duration checks remain unknown until the authorized API run stages the existing dataset clips.

| split | question ID | video path | task | frames | expected SAM 3.1 text concept |
|---|---|---|---|---:|---|
| valQ | `train_014717` | `Hawk_UAV/BE/DJI_0353_03.mp4` | Geometric Relation | 240 | `parking lot` |
| valQ | `train_002581` | `VisDrone/uav0000086_00000_v_01.mp4` | Geometric Relation | 180 | `volleyball court` |
| valQ | `train_013068` | `Hawk_UAV/BE/DJI_0378_05.mp4` | Geometric Relation | 240 | `red barn` |
| valQ | `train_003645` | `ERA_Select/Boating/Boating_010.mp4` | Geometric Relation | 140 | `rowing boats` |
| valT | `train_011808` | `Hawk_UAV/BE/DJI_0364_04.mp4` | Geometric Relation | 240 | `parking lot` |
| valT | `train_012156` | `Hawk_UAV/BE/DJI_0339_11.mp4` | Geometric Relation | 240 | `basketball court` |
| valT | `train_012502` | `Hawk_UAV/BE/DJI_0352_07.mp4` | Geometric Relation | 240 | `red barn` |
| valT | `train_014232` | `UAVDT/M0208.mp4` | Geometric Relation | 265 | `bridge` |
| valQ | `train_003181` | `ERA_Select/Baseball/Baseball_024.mp4` | Temporal Grounding | 140 | `sports court` |
| valQ | `train_009393` | `Hawk_UAV/NJ/DJI_0186_d4_02.mp4` | Temporal Grounding | 180 | `soccer field` |
| valQ | `train_009869` | `Hawk_UAV/PH/DJI_0226_d4_03.mp4` | Temporal Grounding | 180 | `baseball field` |
| valQ | `train_015573` | `Hawk_UAV/BE/DJI_0345_05.mp4` | Temporal Grounding | 240 | `car` |
| valT | `train_002906` | `ERA_Select/Baseball/Baseball_007.mp4` | Temporal Grounding | 140 | `baseball fields` |
| valT | `train_009901` | `Hawk_UAV/PH/DJI_0227_d4_02.mp4` | Temporal Grounding | 180 | `soccer fields` |
| valT | `train_010397` | `Hawk_UAV/BE/DJI_0376_03.mp4` | Temporal Grounding | 240 | `black car` |

Totals: 15 unique videos, 8 geometric questions, 7 temporal questions, 8 valQ/7 valT, and 3,085 manifest frames.

## API cost estimate

Using the current official SAM 3.1 video price of **$0.20 per 1,000 video frames**, the known-frame estimate is:

```text
3,085 frames / 1,000 × $0.20 = $0.617 (about $0.62)
```

This assumes the endpoint bills all decoded frames in each submitted clip. The estimate excludes unknown upload/encoding overhead and any retries. The intended call is one noun-phrase concept per video through `sam-3.1` at `https://api.meta.ai/v1`; the returned frame masks/tracks will be mapped to the existing VLM frames before subset scoring with the unchanged Stage A adapter and prompt.

## Existing subset comparison

Filtering the already saved baseline, detector-only, and SAM 2 score files to these exact 15 question IDs gives the same result in all three arms: **9/15 correct (60.0%)**. Per task, each arm is `5/8` geometric and `4/7` temporal. This is a preflight reference only; no SAM 3.1 result exists.

Official references:

- https://ai.meta.com/docs/meta-ai-api/video-understanding/
- https://api.meta.ai/v1
- https://github.com/facebookresearch/sam3#basic-usage
- https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/sam3.md#text-only-prompts

