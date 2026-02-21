# Advertisement Detection Method: Complete Explanation

## Quick Summary

This paper presents a non-intrusive, non-reference deep learning framework to detect and segment individual advertisements from video streams. The method uses lightweight audio data for global segmentation and audiovisual features for classification, achieving 97% accuracy across movies, TV shows, and live sports from streaming services like Amazon Freevee and Prime Video.

---

## Problem Statement

### The Challenge

Streaming service providers need to measure user experience metrics related to advertisements:
- **Commercial break frequency**: How often are ads inserted?
- **Commercial break duration**: How long do ad breaks last?
- **Ad placement quality**: Are ads played at appropriate moments?
- **Ad relevance**: How relevant is an ad to the current content?

To compute these metrics, we need an algorithm that can **reliably segment out individual advertisements** from a continuous video stream.

### Two Approaches to Ad Detection

#### 1. Intrusive Approach
- Intercept network traffic
- Parse service provider data and logs
- **Problems**:
  - Data is usually encrypted
  - Log formats change frequently
  - Requires access to network infrastructure
  - Not scalable across different providers

#### 2. Non-Intrusive Approach (This Paper)
- Capture streamed video
- Analyze using computer vision and audio processing
- **Advantages**:
  - Platform-agnostic
  - No need for provider cooperation
  - Measures actual user experience
  - Scalable

### Reference-Based vs. Non-Reference Methods

**Reference-Based**:
- Requires an ad gallery (database of known ads)
- Match current video against gallery
- **Problems**:
  - Gallery difficult to obtain for providers like Freevee
  - Gallery updated frequently
  - Doesn't generalize to new ads

**Non-Reference (This Paper)**:
- No ad gallery required
- Detects ads based on audio/visual features alone
- More challenging but more broadly applicable
- Better for Video Quality Analysis (VQA) use cases

---

## Key Innovations

### 1. Audio-Based Global Segmentation

**Observation**: There is usually a brief silent transition between content and ads, or between individual ads.

**Innovation**: Use lightweight audio data (16kHz+) instead of heavy video data for segmentation.

**Advantages**:
- **Higher temporal resolution**: Audio at 16kHz has 16,000 samples/second vs video at 60 fps
- **Lightweight**: 20 minutes of audio (16kHz) = 19.2M data points; 20 minutes of RGB video (60fps) = 7,465M data points (388× larger)
- **6-12× faster** than video-based segmentation methods

### 2. Deep Learning for Change Point Detection

**Previous work**: Hand-crafted features + classical ML (SVM, AdaBoost)

**This paper's innovation**:
- Deep neural network (ResNet-34) trained with Siamese architecture
- Learns to distinguish "same video" vs "different video" transitions
- Transfer learning from speaker verification (VoxCeleb2 dataset)
- Triplet loss for better discrimination

### 3. Temporal Attention for Ad Classification

**Previous work**:
- Pre-trained CNN for single-frame classification
- Hand-crafted features
- No temporal modeling

**This paper's innovation**:
- Audiovisual SlowFast Network (fuses audio + video)
- Temporal attention models (TAP, LSTM, ViT)
- Learns temporal patterns in ads vs content
- 4.3-second clips provide sufficient context

### 4. Hybrid Audio-Video Approach

- **Segmentation**: Primarily audio-based (fast)
- **Validation**: Video-based scene detection for missed boundaries
- **Classification**: Audiovisual features (accurate)
- **Post-processing**: Audio classification (PANNs) to prevent over-segmentation

---

## Method Overview: End-to-End Pipeline

```
Input: Video Playback (20-120 minutes)
         ↓
┌────────────────────────────────────────────┐
│  STAGE 1: Audio-Based Segmentation         │
│  - Detect silent segments                  │
│  - Classify as Change Points (CP)          │
│  Output: Segmented video                   │
└────────────────────────────────────────────┘
         ↓
   [SG0, SG1, SG2, ..., SGn]
         ↓
┌────────────────────────────────────────────┐
│  STAGE 2: Ad Classification                │
│  - Extract AV features (4.3s clips)        │
│  - Temporal attention classification       │
│  Output: Ad/Content labels                 │
└────────────────────────────────────────────┘
         ↓
   [Content, Ad, Ad, Content, ...]
         ↓
┌────────────────────────────────────────────┐
│  STAGE 3: Post-Processing                  │
│  - Fix under-segmentation (short ads)      │
│  - Fix over-segmentation (vocal pauses)    │
│  Output: Final ad segments                 │
└────────────────────────────────────────────┘
         ↓
Individual Ad Segments + Timestamps
```

---

## Detailed Method: Step-by-Step

### STAGE 1: Audio-Based Segmentation

#### Step 1.1: Silent Segment Detection

**Input**: Raw audio stream from video

**Processing**:
1. **Normalize audio**: Convert to int16 format
2. **Apply band-pass filter**: [300Hz, 6000Hz] to remove noise
3. **Scan for silence**: Find time periods [t1, t2] where:
   - Maximum volume < threshold (set to 4 after normalization)
   - Minimum duration ≥ 10 milliseconds

**Output**: List of silent segments [SS1, SS2, SS3, ..., SSm]

**Intuition**: Ads and content typically have a brief silence at transitions. This silence serves as a potential "change point" indicator.

---

#### Step 1.2: Change Point Classification

**Question**: Is this silent segment a true boundary between content/ad, or just a pause in dialogue?

**Process**:
1. **Extract audio context**:
   - wav1: 2 seconds of audio BEFORE silent segment
   - wav2: 2 seconds of audio AFTER silent segment

2. **Compute Log Mel Spectrograms**:
   - Convert wav1 → spectrogram1
   - Convert wav2 → spectrogram2
   - Log Mel Spectrogram captures frequency patterns over time

3. **Extract features using ResNet-34**:
   - spectrogram1 → 512-dimensional feature vector f1
   - spectrogram2 → 512-dimensional feature vector f2
   - Same network processes both (Siamese architecture)

4. **Compute distance**:
   - D = Euclidean distance between f1 and f2
   - D = ||f1 - f2||²

5. **Decision**:
   ```
   IF D > T_CP:
       Silent segment is a Change Point (CP)
       → Mark as boundary
   ELSE:
       Silent segment is NOT a Change Point
       → Just a pause, ignore
   ```

**Intuition**:
- Audio before and after a TRUE ad boundary sounds very different (e.g., movie dialogue → upbeat ad music)
- Audio before and after a PAUSE in dialogue sounds similar
- ResNet-34 learns to extract features that capture this difference

---

#### Step 1.3: Video-Based Validation (Backup)

**Problem**: Audio-based CP detection might miss some boundaries (under-segmentation)

**Solution**: For silent segments classified as "NOT CP", run video scene detection
1. Extract video frames near the silent segment
2. Run PySceneDetect algorithm
3. If scene change detected → Re-label as CP

**Benefit**: Catches visual transitions without audio silence (e.g., quick cuts)
**Cost**: Minimal (only run on small subset of frames)

---

#### Step 1.4: Segment Cleanup

After CP detection, we have segments: [SG0, SG1, SG2, ..., SGn]

**Problem**: Some segments might be too short (< 8 seconds)
- Hard to classify accurately
- May be segmentation errors

**Solution**: Merge or discard short segments
1. **Check neighbors**: If adjacent segments are also short → Merge them
2. **Repeat**: Until no more short adjacent segments
3. **Discard**: Remaining isolated short segments

**Threshold**: 8 seconds (empirically chosen)

**Trade-off**:
- Improves classification accuracy (more context per segment)
- Creates under-segmentation for ads < 8 seconds (fixed in post-processing)

---

### STAGE 2: Ad Classification

#### Step 2.1: Feature Extraction (Audiovisual SlowFast)

For each segment [SGi]:

**If segment > 85 seconds**:
- Skip classification
- Automatically label as "Content"
- **Rationale**: Individual ads rarely exceed 60 seconds; this saves massive computation

**Else**:

1. **Crop into clips**:
   - Divide segment into 4.3-second clips
   - Use 2-second hop size (overlapping)
   - Example: 20s segment → [0-4.3s, 2-6.3s, 4-8.3s, ..., 15.7-20s]

2. **Process each clip through Audiovisual SlowFast**:

   **Slow Pathway** (spatial detail):
   - Low frame rate sampling
   - High channel count (2048)
   - Processes 4 frames

   **Fast Pathway** (motion detail):
   - High frame rate sampling
   - Low channel count (256)
   - Processes 32 frames

   **Audio Pathway** (audio detail):
   - Process Log Mel Spectrogram
   - 256 temporal frames
   - 1024 channels

   **Fusion**:
   - All three pathways processed by ResNet-50
   - Cross-pathway fusion (audio → fast → slow)
   - XY average pooling
   - Output: Concatenated feature vector [3328 dimensions, temporal resolution]

**Output**: Feature tensor [3328, N_T] where N_T = 4 or 32 depending on fusion strategy

---

#### Step 2.2: Temporal Attention Classification

**Input**: Feature tensor [3328, N_T] from each 4.3s clip

**Process** (using LSTM model - best performance):
1. Feed feature tensor through bi-directional LSTM
2. LSTM learns temporal dependencies across the N_T timesteps
3. Extract feature at initial timestamp T_0
4. Pass through fully connected layers: [1000] → [256] → [10] → [2]
5. Apply Softmax to get probabilities: [P(Ad), P(Content)]

**Decision for single clip**:
```
IF P(Ad) > P(Content):
    Clip is "Ad"
ELSE:
    Clip is "Content"
```

---

#### Step 2.3: Segment-Level Aggregation

**Problem**: We have classifications for many 4.3s clips, but need one label for the entire segment

**Solution**: Double-threshold decision (inspired by Canny edge detector)

1. **Compute Ad ratio**:
   ```
   Ad_ratio = (Number of clips labeled "Ad") / (Total number of clips)
   ```

2. **Three-level classification**:
   ```
   IF Ad_ratio ≥ 0.3333:
       Segment is "Strong Ad"
   ELIF Ad_ratio ≥ 0.1:
       Segment is "Weak Ad"
   ELSE:
       Segment is "Non-Ad" (Content)
   ```

3. **Hysteresis tracking** (noise suppression):
   - Keep weak Ad segments ONLY if connected to strong Ad segments
   - Discard isolated weak Ad segments
   - Prevents false positives from noisy clips

**Final Output**: Each segment labeled as "Ad" or "Content"

---

### STAGE 3: Post-Processing

#### Step 3.1: Fix Ad Under-Segmentation

**Problem**: Some short ads (4-5 seconds) were merged in Stage 1 cleanup

**Solution**: Re-segment with more aggressive thresholds

For each segment labeled "Ad":
1. **Re-run silent segment detection** with:
   - Minimum duration: 2ms (vs 10ms originally)
   - Short segment threshold: 4s (vs 8s originally)

2. **Prevent over-correction** (vocal-only scenes have natural pauses):
   - Run PANNs audio classifier → 527 audio types
   - Group into:
     - **Continuous**: Instruments, fire, water (less likely false positives)
     - **Non-continuous**: Vocals, birds, reptiles (natural pauses)

3. **Handle mixed audio** (vocals + background music):
   - Apply voice/music separation (Librosa)
   - Weaken vocal signal
   - Re-classify: vocal+music → "continuous"; vocal-only → "non-continuous"

4. **Merge strategy**:
   - If adjacent segments are BOTH non-continuous → Merge
   - Else → Keep separate

5. **Validate remaining segments**:
   - Re-run CP classifier on newly detected silent segments
   - Keep only if classified as true CP

**Outcome**: Short ads properly separated without over-segmenting vocal scenes

---

#### Step 3.2: Fix Content Under-Segmentation

**Problem**: Long content segments (>85s) might contain ads

**Solution**: Check long segments for hidden ads

For each segment labeled "Content" and duration > 85 seconds:
1. **Re-segment** using audio-based detection (same as Step 3.1)
2. **If partitioned** into multiple sub-segments:
   - Run Ad classifier on HEAD and TAIL sub-segments only
   - If either is classified as "Ad":
     - Cut it out
     - Create new "Ad" segment
     - Adjust content segment boundaries

**Benefit**: Catches ads that were missed while maintaining computational efficiency (only check head/tail)

---

## How the Method Works End-to-End: Example

### Example Input
A 60-minute movie with 5 ad breaks, each containing 3 ads of 15 seconds each.

### Step-by-Step Processing

**Stage 1: Segmentation**
1. **Silent segment detection** finds ~200 silent moments in 60 minutes
2. **CP classifier** narrows down to ~15 true change points:
   - 5 content → ad transitions
   - 5 ad → content transitions
   - ~10 boundaries between individual ads (not all detected)
3. **Segment cleanup** merges some short segments
4. **Output**: ~20 segments (some ad breaks under-segmented)

**Stage 2: Classification**
1. Long content segments (>85s) → Auto-label as "Content" (fast)
2. Ad break segments (10-45s):
   - Crop into 4.3s clips
   - Extract AV features
   - LSTM classification
   - Aggregate with double threshold
3. **Output**: ~15 segments labeled (5 Content, 10 Ad)

**Stage 3: Post-Processing**
1. **Fix ad under-segmentation**:
   - Some ad breaks contain 2-3 merged ads
   - Re-segment with aggressive thresholds
   - PANNs prevents over-segmentation on dialogue
   - **Result**: 15 individual ads detected
2. **Fix content under-segmentation**:
   - Check long content segments
   - Find 1 missed ad at beginning of a content segment
   - **Result**: 16 total ads detected (1 recovered)

**Final Output**:
- 5 Content segments (movie scenes)
- 16 Ad segments (15 correct + potentially 1 false positive)
- Timestamps for each segment

---

## Key Design Principles

### 1. Hierarchical Processing
- **Coarse segmentation first** (audio, fast)
- **Fine classification second** (audiovisual, slower)
- **Refinement third** (post-processing, targeted)

### 2. Computational Efficiency
- Audio-based segmentation: 6-12× faster than video
- Skip long segments in classification: Massive savings
- Targeted post-processing: Only where needed

### 3. Multi-Modal Fusion
- Audio for segmentation (fast, high temporal resolution)
- Video for validation (catches visual-only transitions)
- Audiovisual for classification (most accurate)

### 4. Robust Decision Making
- Double thresholds (strong/weak)
- Hysteresis tracking (noise suppression)
- Multiple validation steps (CP + scene detect + PANNs)

### 5. Transfer Learning
- ResNet-34 from speaker verification (VoxCeleb2)
- AvSlowFast from video recognition
- PANNs from audio classification (AudioSet)
- Reduces training data requirements

---

## Method Strengths

### 1. Non-Reference
- No ad gallery needed
- Generalizes to new ads
- Applicable to any streaming service

### 2. Domain-Crossing
- Works on movies, TV shows, live sports
- Tested on Amazon Freevee, Prime Video
- Robust to different content types

### 3. Scalable
- Audio-based segmentation is fast
- Long segment skipping saves computation
- Can process hours of video efficiently

### 4. Accurate
- 97% accuracy on end-to-end evaluation
- <1% over-segmentation and under-segmentation
- <2.5% false positives

### 5. Interpretable
- Clear pipeline stages
- Explainable decisions (silence → transition, feature distance → CP)
- Adjustable thresholds for different use cases

---

## Method Limitations

### 1. Short Ad Detection
- Ads < 4 seconds may be discarded in initial cleanup
- Requires aggressive post-processing to recover
- Trade-off between accuracy and over-segmentation

### 2. Silent Transitions Required
- Assumes ads have silent boundaries
- May miss:
  - Seamless transitions (rare in practice)
  - Very short transitions (< 2ms)

### 3. Training Data Requirements
- Requires labeled Ad/Content data
- Manual annotation is time-consuming
- Small dataset limits ViT performance

### 4. Computational Cost (Classification)
- Audiovisual processing is expensive
- Ad classifier (4.3s clip): 12s I/O + 0.04s GPU
- Mitigated by skipping long segments

### 5. Similar Content Challenge
- Movie trailers look like ads
- Credits sequences may be misclassified
- Beginning/end of content often confused

---

## Comparison to Prior Work

### Previous Methods

**Hand-Crafted Features + Classical ML (2005-2011)**:
- Extract color histograms, motion vectors, audio energy
- Train SVM, AdaBoost classifier
- **Limitations**: Features don't generalize, limited accuracy

**Reference-Based Template Matching (2013)**:
- Match ad logos using template matching
- **Limitations**: Requires ad gallery, doesn't scale

**Repeated Signal Detection (2006)**:
- Detect repeated acoustic/visual cues
- **Limitations**: Only works for repeated ads

**DNN-Based Classification (2018 - Ad-Net)**:
- PySceneDetect for segmentation
- Pre-trained CNN for classification
- **Limitations**: Video-based segmentation (slow), no temporal modeling

### This Work's Advantages

| Aspect | Prior Work | This Work |
|--------|------------|-----------|
| Segmentation | Video-based (slow) | Audio-based (6-12× faster) |
| Temporal Modeling | None | LSTM, ViT |
| Multi-Modal | Single modality | Audio + Video fusion |
| Reference | Often required | Non-reference |
| Accuracy | ~90% | 97% |
| Speed | Slow | Fast |

---

## Real-World Applications

### 1. User Experience Measurement
- Measure ad frequency and duration
- Identify excessive ad breaks
- Optimize ad placement

### 2. Ad Policy Compliance
- Verify ads played at correct times
- Check ad duration limits
- Monitor competitor's ad strategies

### 3. Content Quality Analysis
- Measure actual viewer experience
- Identify poor ad transitions
- Quality assurance for streaming services

### 4. Ad Relevance Research
- Extract ad segments for sentiment analysis
- Study ad-content relevance
- Personalization research

### 5. Competitive Intelligence
- Analyze competitor's ad strategies
- Benchmark ad frequency across platforms
- Market research

---

## Future Directions

### 1. End-to-End Trainable System
Currently: Two separate models (CP + Ad)
Future: Single unified model trained end-to-end

### 2. Larger Datasets
- Include NBA, NFL live streaming
- More diverse content types
- Release public benchmark dataset

### 3. Improved Temporal Models
- Larger training data for ViT to excel
- Attention mechanisms for long-range dependencies
- Video transformers

### 4. Finer-Grained Analysis
- Classify ad types (product, service, trailer)
- Extract ad metadata (brand, product)
- Measure ad quality/relevance

### 5. Real-Time Processing
- Optimize for live streaming
- Reduce latency
- Online learning for adaptation

---

## Conclusion

This paper presents a practical, scalable, and accurate method for detecting individual advertisements from video streams. By cleverly leveraging audio data for fast segmentation and audiovisual features for accurate classification, the method achieves 97% accuracy while being 6-12× faster than existing approaches. The non-reference design makes it applicable to any streaming service without requiring an ad gallery, enabling real-world user experience measurement and content quality analysis.

The key insight is that **audio provides sufficient information for fast global segmentation**, while **audiovisual features with temporal modeling provide accurate local classification**. This hierarchical approach balances speed and accuracy, making the method practical for large-scale video analysis.
