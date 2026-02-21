# Project Status - NFL Ad Detection

**Last Updated**: 2026-01-25
**Domain**: NFL broadcasts from Fubo
**Phase**: Data Pipeline & Annotation
**Target Completion**: 1-2 months (prototype), then scale to production

---

## Overall Progress: 35%

```
[#################.........................................] 35/100

Infrastructure  [##########] 100%  ✅ Complete (3 repos operational)
Data Acq        [######....] 60%   🔄 In Progress (download done, annotation workflow defined)
Annotation      [#.........] 10%   🔄 In Progress (DaVinci workflow defined, ready to start)
CP Model        [..........] 0%    🔲 Not Started
Ad Model        [..........] 0%    🔲 Not Started
Pipeline        [..........] 0%    🔲 Not Started
Production      [..........] 0%    🔲 Not Started
```

---

## Multi-Repository Progress

This project spans **4 GitHub repositories** working together. See [DATA_PIPELINE.md](DATA_PIPELINE.md) for complete workflow.

| Repository | Purpose | Status | Progress |
|------------|---------|--------|----------|
| **fubo-scraper** | Extract stream URLs from Fubo TV | ✅ Complete | 100% |
| **GetGamesToLocal** | Download games to local MP4 files | ✅ Complete | 100% |
| **CPPreData** | Detect change point candidates | ✅ Setup done | 50% (not run on dataset yet) |
| **ad-game-explainer** (THIS REPO) | Train models & build pipeline | 🔄 In Progress | 30% |

---

## Phase 0: Data Pipeline Infrastructure ✅ COMPLETE

### Multi-Repository Setup (fubo-scraper, GetGamesToLocal, CPPreData)
- [x] **fubo-scraper**: Stream URL extraction from Fubo TV
- [x] **GetGamesToLocal**: MP4 download with master.csv tracking
- [x] **CPPreData**: Change point detection pipeline
- [x] All 3 repos tested and operational

**Status**: 100% complete - all data infrastructure repos are operational

**Documentation**:
- See [DATA_PIPELINE.md](DATA_PIPELINE.md) for complete workflow
- Each repo has excellent README.md with troubleshooting guides

**Data Available**:
- 150+ NFL games from 2025 season accessible via fubo-scraper
- Multiple games successfully downloaded to local storage
- CPPreData tested on 1 game (Cowboys vs Eagles, 240+ CP candidates detected)

---

## Phase 1: Setup & Infrastructure (Week 1)

### Development Environment ✅ COMPLETE
- [x] Project structure created
- [x] Git repository initialized
- [x] Documentation files created (CLAUDE.md, README.md, NFL_PROJECT_PLAN.md)
- [x] Requirements.txt prepared
- [x] Download scripts created (Fubo m3u8 → mp4)
- [x] S3 upload/sync scripts created
- [x] .claude/ directory for session context
- [ ] **TODO**: Install Python dependencies locally
- [ ] **TODO**: Setup AWS (S3 bucket + EC2 g4dn.xlarge)
- [ ] **TODO**: Test download script on 1 game

**Next Commands**:
```bash
# Install dependencies
conda create -n ad-detection python=3.8
conda activate ad-detection
pip install -r requirements.txt

# Setup AWS S3
aws s3 mb s3://nfl-ad-detection --region us-east-1

# Launch EC2 (via AWS console or CLI)
# Instance: g4dn.xlarge, AMI: Deep Learning AMI (Ubuntu 20.04)
```

**Infrastructure Decision**: AWS EC2 g4dn.xlarge (NOT SageMaker - 4x cheaper)

### Week 1: Data Pipeline Setup ✅ COMPLETE
- [x] **COMPLETE**: fubo-scraper operational (extracts stream URLs)
- [x] **COMPLETE**: GetGamesToLocal operational (downloads games)
- [x] **COMPLETE**: CPPreData operational (detects CP candidates)
- [x] **COMPLETE**: Multiple games successfully downloaded
- [x] **COMPLETE**: CPPreData tested on 1 game (240+ CP candidates)
- [ ] **TODO**: Run CPPreData on all downloaded games
- [ ] **TODO**: Select 10-15 games for prototype dataset annotation

**Data Available**:
- 150+ NFL games from 2025 season (~450-525 hours)
- Broadcast quality with embedded commercials
- Multiple networks (CBS, NBC, FOX, ESPN)
- Multiple games downloaded via GetGamesToLocal (check master.csv for count)

**Prototype Needs**:
- 10-15 games (30-45 hours) → 1000-1500 clips
- Diverse networks, game types, season timing

---

## Phase 2: Change Point Detection & Annotation (Week 2-3)

### Step 1: Run CPPreData on Downloaded Games
- [ ] **TODO**: Run CPPreData on all downloaded games from GetGamesToLocal
  ```bash
  cd /Users/hungmuhamath/projects/GitHub/CPPreData/
  uv run process_videos.py --input /Users/hungmuhamath/projects/GitHub/GetGamesToLocal/videos
  ```
- [ ] **TODO**: Verify EDL files and manifest CSVs generated for each game
- [ ] **TODO**: Check audio extraction quality

### Step 2: Manual Annotation with DaVinci Resolve
- [ ] **TODO**: Select 10-15 games for prototype annotation
- [ ] **TODO**: Annotate 3 games manually (test workflow):
  - Import MP4 file into DaVinci Resolve
  - Import EDL file (markers appear on timeline at each silent segment)
  - Review ~200-300 markers per game
  - Identify TRUE segment boundaries (content↔ad transitions, typically ~10-20 per game)
  - Create annotation JSON with segments and change points
  - Save to `data/annotations/{game_id}.json`
- [ ] **TODO**: Annotate remaining 7-12 games
- [ ] **TODO**: Validate annotation quality (random review)

### Step 3: Extract Short Clips for Training
- [ ] **TODO**: Run clip extraction script on annotation JSONs
- [ ] **TODO**: Generate balanced dataset (ad vs content clips, 10-30s each)
- [ ] **TODO**: Verify clip quality and labels
- [ ] **TODO**: Create train/val/test splits (70/15/15)

**Annotation Format**: Using Approach B (Ad Breaks)
- Annotate entire commercial breaks as single "ad" segments (not individual ads)
- Each game: ~10-25 segments (alternating content/ad)
- Simpler and faster than individual ad annotation
- Post-processing will handle splitting merged ads later

**Dataset Targets**:
- Prototype: 10-15 games → 150-250 segments → 1000-1500 short clips
- Production: 30-40 games → 400-800 segments → 3000-5000 short clips

**NFL-Specific Notes**:
- Each game has ~60-80 commercial breaks
- ~15-25 minutes of ads per game
- Clear ad break patterns (timeouts, quarters, 2-min warning)
- CPPreData typically detects 200-300 silence segments per game
- Manual annotation identifies ~10-20 true segment boundaries (content/ad transitions)

**Current Status**:
- ✅ Annotation workflow defined (DaVinci Resolve + JSON format)
- ✅ Annotation format specified (Approach B: ad breaks)
- 1 game processed through CPPreData (Cowboys vs Eagles, 240+ candidates)
- 0 games manually annotated
- 🔲 **NEXT**: Run CPPreData on all downloaded games
- 🔲 **NEXT**: Annotate first 3 games in DaVinci Resolve

---

## Phase 3: Change Point Classifier (Week 4-5)

### Model Development
- [ ] **TODO**: Implement audio preprocessing module (`src/audio_processing.py`)
- [ ] **TODO**: Implement CP classifier model (`src/models/cp_classifier.py`)
- [ ] **TODO**: Implement CP dataset loader (`src/datasets/cp_dataset.py`)
- [ ] **TODO**: Download VoxCeleb2 pre-trained weights
- [ ] **TODO**: Implement training script (`src/train_cp_classifier.py`)

### Training
- [ ] **TODO**: Train for 50 epochs with triplet loss
- [ ] **TODO**: Evaluate on validation set
- [ ] **TODO**: Tune hyperparameters (learning rate, batch size, threshold)
- [ ] **TODO**: Save best model checkpoint

**Target Metrics**:
- AUC: ≥0.90 (NFL should be easier than movies due to clear transitions)
- Detection Rate: ≥90% at 10% FPR

**NFL Advantage**: Clear audio transitions (crowd fade → silence → ads)

**Current Status**: Not started

---

## Phase 4: Ad Classifier (Week 6-7)

### Model Development
- [ ] **TODO**: Implement video preprocessing module (`src/video_processing.py`)
- [ ] **TODO**: Implement audio-only ad classifier (`src/models/ad_classifier.py`)
- [ ] **TODO**: Implement LSTM temporal model
- [ ] **TODO**: Implement ad dataset loader (`src/datasets/ad_dataset.py`)
- [ ] **TODO**: Implement training script (`src/train_ad_classifier.py`)

### Training
- [ ] **TODO**: Train for 50 epochs with cross-entropy loss
- [ ] **TODO**: Evaluate on validation set
- [ ] **TODO**: Tune double-threshold parameters (high=0.3333, low=0.1)
- [ ] **TODO**: Save best model checkpoint

**Target Metrics**:
- Ad Precision: ≥90% (prototype with audio-only)
- Ad Recall: ≥90%
- Upgrade to audiovisual: ≥95% (production)

**NFL Advantage**: Ads have distinct audio patterns (music, voiceover vs crowd, commentary)

**Current Status**: Not started

---

## Phase 5: End-to-End Pipeline (Week 8)

### Integration
- [ ] **TODO**: Implement segmentation pipeline (`src/pipeline/segmentation.py`)
- [ ] **TODO**: Implement classification pipeline (`src/pipeline/classification.py`)
- [ ] **TODO**: Implement end-to-end pipeline (`src/pipeline/end_to_end.py`)
- [ ] **TODO**: Implement evaluation metrics (`src/evaluation/metrics.py`)

### Testing
- [ ] **TODO**: Test on 5-10 long videos (20-60 min)
- [ ] **TODO**: Compute end-to-end metrics (precision, recall, F1)
- [ ] **TODO**: Error analysis and debugging
- [ ] **TODO**: Performance optimization

**Target Metrics**:
- F1 Score: ≥75% (prototype)
- Processing Speed: <5 min for 20-min video
- False Positive Rate: <20%

**Current Status**: Not started

---

## Future Enhancements (Post-Prototype)

### Model Improvements
- [ ] Upgrade to full audiovisual SlowFast network
- [ ] Implement Vision Transformer (ViT) temporal model
- [ ] Add PANNs post-processing
- [ ] Implement aggressive re-segmentation for short ads
- [ ] **CP Classifier: Strong Supervision Improvement**
  - Current approach: Weak supervision (same clip = non-CP, different clips = CP)
  - Improvement: Extract 2s before/after each annotated change_point
  - Benefit: TRUE CP pairs at exact boundaries vs approximate pairs
  - Rationale: NFL annotations include precise CP timestamps from DaVinci Resolve review
  - Expected impact: Higher CP classifier accuracy, better boundary detection

### Dataset Scaling
- [ ] Expand to 30-40 games (12,000+ clips)
- [ ] Include playoff games (different ad patterns)
- [ ] Multi-season data (different advertiser campaigns)

### Multi-Sport Support
- [ ] NBA (similar ad patterns to NFL)
- [ ] MLB (longer games, different flow)
- [ ] NHL (fewer stoppages)

### Production Features
- [ ] REST API for inference (`src/api/app.py`)
- [ ] Docker containerization
- [ ] TorchScript optimization
- [ ] Batch processing support
- [ ] Real-time processing capability

### Deployment
- [ ] Cloud deployment (AWS Lambda, GCP Cloud Run)
- [ ] Edge deployment (TensorRT, ONNX)
- [ ] Monitoring and logging
- [ ] A/B testing framework

---

## Blockers & Risks

### Current Blockers
- None (just starting)

### Identified Risks
1. **m3u8 download reliability**: Fubo streams may have DRM or rate limiting
   - Mitigation: Test download script early, use rate limiting (5s between downloads)
2. **Annotation bottleneck**: Manual annotation is time-consuming
   - Mitigation: NFL has predictable patterns, can semi-automate with quarter detection
3. **EC2 costs**: Training can get expensive if not managed
   - Mitigation: Use spot instances for non-critical work, stop instance when idle
4. **Network variations**: Different networks (CBS, NBC) may have different ad patterns
   - Mitigation: Train on diverse network samples from day 1

**Risk Eliminated**: NFL matches paper's sports domain better than gaming streams would have!

---

## Questions & Decisions

### Open Questions
- [ ] AWS region for S3/EC2? (Recommend us-east-1 for cost)
- [ ] Which 10-15 games to select for prototype? (Need network/game type diversity)
- [ ] Annotation tool preference? (VIA, Label Studio, or custom NFL tool?)
- [ ] Should we build automated quarter detection to speed up annotation?

### Decisions Made
- [x] Target: NFL broadcasts from Fubo (BETTER than original gaming idea!)
- [x] Infrastructure: AWS EC2 g4dn.xlarge + S3 (NOT SageMaker)
- [x] Data pipeline: m3u8 → ffmpeg → mp4 → S3
- [x] Prototype approach: Audio-only classifier, then upgrade to audiovisual
- [x] Timeline: 5-6 weeks for production-ready prototype

---

## Next Steps (This Week)

### Priority 1 (Must Do This Week) - Data Preparation
1. [ ] **Check downloaded game count**:
   ```bash
   cd /Users/hungmuhamath/projects/GitHub/GetGamesToLocal/
   wc -l master.csv
   grep "downloaded" master.csv | wc -l
   ```
2. [ ] **Run CPPreData on all downloaded games**:
   ```bash
   cd /Users/hungmuhamath/projects/GitHub/CPPreData/
   uv run process_videos.py --input /Users/hungmuhamath/projects/GitHub/GetGamesToLocal/videos
   ```
3. [ ] **Verify manifest CSVs generated** (check `manifests/` directory)
4. [ ] **Select 3 games for test annotation** (diverse networks: CBS, NBC, FOX)

### Priority 2 (Should Do This Week) - Start Annotation
5. [ ] **Annotate 1 game manually with DaVinci Resolve** (test workflow):
   - Import MP4 + EDL file into DaVinci Resolve
   - Review markers on timeline (~200-300 markers)
   - Identify true segment boundaries (~10-20 per game)
   - Create annotation JSON with segments and change points
   - Save to `data/annotations/{game_id}.json`
6. [ ] **Document annotation workflow** (time per game, challenges encountered)
7. [ ] **Annotate 2 more test games**

### Priority 3 (Next 1-2 Weeks) - Scale Annotation
8. [ ] **Select 10-15 games for prototype** (diverse networks/types/season timing)
9. [ ] **Annotate remaining 7-12 games**
10. [ ] **Extract short clips (10-30s) from annotations**
11. [ ] **Create train/val/test splits** (70/15/15)
12. [ ] **Install PyTorch environment** (`conda create`, `pip install -r requirements.txt`)

---

## Weekly Progress Log

### Week 1 (2024-01-21)
- ✅ Created project structure
- ✅ Initialized git repository
- ✅ Created documentation (CLAUDE.md, README.md, NFL_PROJECT_PLAN.md)
- ✅ **Pivot to NFL**: Changed from gaming streams to NFL broadcasts (better domain match!)
- ✅ Created .claude/ directory for session context
- ✅ Defined requirements and dependencies

### Weeks 2-4 (Data Infrastructure)
- ✅ Built fubo-scraper (Python + Selenium-Wire)
- ✅ Built GetGamesToLocal (Node.js + FFmpeg)
- ✅ Built CPPreData (Python + Pydub + SciPy)
- ✅ Tested complete pipeline on 1 game
- ✅ Downloaded multiple NFL games to local storage
- ✅ Created DATA_PIPELINE.md documenting complete workflow

### Week 5 (2026-01-27) - Current
- ✅ Reviewed all 3 data repos and documented workflow
- ✅ Updated PROJECT_STATUS.md to reflect actual progress (15% → 35%)
- ✅ Defined DaVinci Resolve annotation workflow
- ✅ Specified annotation JSON format (Approach B: ad breaks)
- ✅ Updated DATA_PIPELINE.md with complete workflow
- 🔄 **Next**: Run CPPreData on all downloaded games
- 🔄 **Next**: Start manual annotation with DaVinci Resolve (3 games)

---

## Resource Tracking

### Time Spent
- Setup & Documentation: 3 hours
- Script development: 1 hour
- **Total**: 4 hours / ~120 hours estimated (NFL should be faster than gaming)

### Costs Incurred
- $0 (not started AWS usage yet)

### Expected Costs
- S3 storage: ~$15/month
- EC2 training (g4dn.xlarge): ~$43 total
- Data transfer: <$5
- **Total Estimated**: $60-80 (much cheaper than SageMaker would have been!)

---

## Success Criteria Checklist

### Prototype Success (8 weeks)
- [ ] Working end-to-end pipeline
- [ ] 75%+ F1 score on test set
- [ ] <5 min processing time for 20-min video
- [ ] Tested on 3+ different games/platforms

### Production Success (3-4 months)
- [ ] 95%+ F1 score on test set
- [ ] <1 min processing time for 20-min video
- [ ] Works across all major platforms
- [ ] REST API deployed
- [ ] Real-time processing capable

---

**Notes**: Update this file weekly to track progress and identify blockers early.
