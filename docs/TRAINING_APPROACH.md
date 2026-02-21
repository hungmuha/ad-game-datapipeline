# Training Approach: CP Classifier

**Last Updated**: 2026-01-27
**Status**: CORRECTED - Strong Supervision Required

---

## CORRECTION: Strong Supervision Required (Following Paper)

**PREVIOUS INTERPRETATION WAS INCORRECT**

After careful re-analysis, the paper uses **STRONG SUPERVISION** with annotated CP boundaries, NOT weak supervision.

### Why Strong Supervision is Required

**Inference Task**:
```python
# For each silent segment, compare audio before vs after:
wav_before = extract_2s_before_silence()
wav_after = extract_2s_after_silence()
distance = ||f(wav_before) - f(wav_after)||²

if distance > threshold:
    → Silent segment IS a CP (boundary)
else:
    → Silent segment is NOT a CP (pause)
```

**Training Requirement**:
For the model to learn this task, training data MUST have the same structure:
- Pairs of (2s_before, 2s_after) at silent segments
- Labels indicating if the silent segment is a TRUE CP or not

You CANNOT train on random clips and expect it to work on before/after comparisons.

## Correct Approach: Strong Supervision (Following Paper)

### Evidence from Paper

**From model_and_training.md:437-439**:
```
For CP Classifier:
- Binary labels: "Same video" vs "Different video"
- Silent segment timestamps  ← They HAD annotated silent segments!
```

**From method_explanation.md:165-167** (Inference):
```
Extract audio context:
- wav1: 2 seconds of audio BEFORE silent segment
- wav2: 2 seconds of audio AFTER silent segment
```

**Conclusion**: Training data must match inference structure - 2s before/after silent segments.

### How It Actually Works

#### Training Data Preparation
1. **Detect Silent Segments**: Run CPPreData on raw videos → EDL with all silent segments
2. **Manual Annotation**: Review in DaVinci Resolve, label which silent segments are TRUE CPs
3. **Extract CP Pairs**: For each annotated silent segment, extract 2s before/after

#### Pair Generation (Preprocessing Step)

**Positive Pairs (Non-CP)**:
```python
# From silent segments that are NOT change points
For each non-CP silent segment within same video:
    wav_before = 2s before silent segment (same content)
    wav_after = 2s after silent segment (same content)
    → Positive pair (same playback, just a pause)

Example:
  Non-CP silence at 145s (pause in commentary)
  wav_before = audio[143s : 145s]  (commentary)
  wav_after = audio[145s : 147s]   (commentary continues)
  → Positive pair (distance should be small)
```

**Negative Pairs (CP)**:
```python
# From silent segments that ARE change points
For each TRUE CP at annotated boundary:
    wav_before = 2s before CP (end of previous segment)
    wav_after = 2s after CP (start of next segment)
    → Negative pair (different playbacks at boundary)

Example:
  CP at 1245.3s (content → ad transition)
  wav_before = audio[1243.3s : 1245.3s]  (game content)
  wav_after = audio[1245.3s : 1247.3s]   (ad audio)
  → Negative pair (distance should be large)
```

#### Loss Function
**Triplet Loss** with margin = 1.0:
```
L(A, P, N) = max(||f(A) - f(P)||² - ||f(A) - f(N)||² + 1.0, 0)

where:
  A = anchor (wav_before from a silent segment)
  P = positive (wav_after from non-CP silent segment - same content)
  N = negative (wav_after from CP silent segment - different content)
  f = ResNet-34 feature extractor
```

### Key Characteristics

**Requirements**:
- ✅ Annotated silent segments (TRUE CPs vs non-CPs)
- ✅ Extract 2s before/after each silent segment
- ✅ Labels: "Same playback" (non-CP) vs "Different playback" (CP)

**Advantages**:
- ✅ Matches published research exactly
- ✅ Direct supervision on CP boundaries
- ✅ Model learns exact task it will perform at inference
- ✅ High-quality training signal

**Data Requirements**:
- ⚠️ Requires manual annotation of silent segments (CP vs non-CP)
- ⚠️ Limited by number of detected silent segments
- ✅ DaVinci Resolve workflow provides exactly this annotation!

---

## Correct Interpretation of Paper

From `model_and_training.md:72-81`:

> **Positive Pairs (Non-CP)**:
> - Sample from the same video playback
> - Find silent segments within a clip
> - Audio clips before and after this point are temporally adjacent

**Correct interpretation**: Find non-CP silent segments, extract 2s before/after

> **Negative Pairs (CP)**:
> - Sample audio clips from different playbacks

**Correct interpretation**: "Different playbacks" means:
- wav_before from one segment (e.g., content)
- wav_after from different segment (e.g., ad)
- At a TRUE CP boundary between them

From `model_and_training.md:437-439`:
> For CP Classifier:
> - Binary labels: "Same video" vs "Different video"
> - **Silent segment timestamps** ← They annotated which silent segments were CPs!

---

## Implementation Requirements

### Required Workflow (DaVinci Resolve)

Our DaVinci Resolve workflow provides EXACTLY what the paper requires:

#### Step 1: Detect All Silent Segments
CPPreData detects ALL silent segments → EDL file (~200-300 per game)

#### Step 2: Annotate Silent Segments
In DaVinci Resolve, label each silent segment marker:
- **TRUE CP** (1): Boundary between content/ad or ad/content
- **Non-CP** (0): Pause within same content (commentary pause, crowd noise dip)

#### Step 3: Extract Training Pairs
**For TRUE CPs** (Negative Pairs):
```python
# From annotated change_points in JSON
For each change_point at timestamp T where is_CP = True:
  1. Extract 2s BEFORE CP: [T-2.0s, T] (end of previous segment)
  2. Extract 2s AFTER CP:  [T, T+2.0s] (start of next segment)
  3. Save as negative pair (different playbacks)

Example:
  change_point = 1245.3s (content → ad, marked as TRUE CP in DaVinci)
  wav_before = audio[1243.3s : 1245.3s]  (game content)
  wav_after = audio[1245.3s : 1247.3s]   (ad audio)
  → Negative pair for triplet loss
```

**For Non-CPs** (Positive Pairs):
```python
# From silent segments marked as non-CP
For each silent segment at timestamp T where is_CP = False:
  1. Extract 2s BEFORE silence: [T-2.0s, T]
  2. Extract 2s AFTER silence: [T, T+2.0s]
  3. Save as positive pair (same playback)

Example:
  silent_segment = 145s (pause in commentary, marked as non-CP in DaVinci)
  wav_before = audio[143s : 145s]  (commentary)
  wav_after = audio[145s : 147s]   (commentary continues)
  → Positive pair for triplet loss
```

### Why This Matches the Paper

**Evidence 1**: Model architecture processes 2s before/after silent segments (method_explanation.md:165-167)

**Evidence 2**: Paper mentions "Silent segment timestamps" in annotation requirements (model_and_training.md:437-439)

**Evidence 3**: Positive pairs are "temporally adjacent" around silent segments (model_and_training.md:76)

**Evidence 4**: Negative pairs from "different playbacks" = different segments at CP boundary (model_and_training.md:79)

### Data Requirements

**Per Game (~3 hours)**:
- CPPreData detects: ~200-300 silent segments
- Manual annotation identifies: ~10-20 TRUE CPs, ~180-280 non-CPs
- Training pairs generated:
  - ~10-20 negative pairs (TRUE CPs)
  - ~180-280 positive pairs (non-CPs)
  - Total: ~200-300 pairs per game

**For 10-15 Games**:
- ~100-300 TRUE CP pairs (negative examples)
- ~1800-4200 non-CP pairs (positive examples)
- Balanced dataset with appropriate class weighting

---

## Implementation Approach

**We implement strong supervision (paper's actual approach)**:

1. ✅ CPPreData detects all silent segments → EDL
2. ✅ DaVinci Resolve review labels each silent segment (CP vs non-CP)
3. ✅ Extract 2s before/after each silent segment
4. ✅ Train with triplet loss on annotated pairs
5. ✅ Exactly matches paper's methodology

---

## References

- **Paper Section**: Model Training → CP Classifier → Positive/Negative Pair Generation
- **Implementation**: `Technical_Info/implementation_gameplan.md:635-711`
- **Dataset Class**: `CPDataset.__getitem__()` generates pairs dynamically
- **Training Data**: `model_and_training.md:62-81`

---

## Summary

**Current**: Weak supervision (same clip = non-CP, different clips = CP)
**Annotated CPs**: Used to define segments, not for direct pair extraction
**Future**: Strong supervision with annotated boundaries (potential improvement)
**Rationale**: Follow paper exactly, document improvement opportunity

---

**Questions?** See:
- `CLAUDE.md` → Training Processes → CP Classifier Training
- `DATA_PIPELINE.md` → Step 4 → CP Classifier Training
- `Technical_Info/implementation_gameplan.md` → Phase 2 → CP Classifier Training
