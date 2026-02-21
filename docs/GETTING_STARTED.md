# Getting Started - NFL Ad Detection

**Welcome!** This guide helps you quickly understand the current project status and next steps.

---

## ✅ What's Complete (35% Overall Progress)

### Infrastructure (100% Complete)
- ✅ **fubo-scraper**: Extracts stream URLs from Fubo TV
- ✅ **GetGamesToLocal**: Downloads games to local MP4 files
- ✅ **CPPreData**: Detects change point candidates (silent segments)
- ✅ **ad-game-explainer**: Project structure and documentation

### Data Acquisition (60% Complete)
- ✅ 150+ NFL games accessible via fubo-scraper
- ✅ Multiple games downloaded to local storage
- ✅ CPPreData tested on 1 game (Cowboys vs Eagles, 240+ silent segments detected)
- ✅ EDL generation working

### Annotation Workflow (10% Complete)
- ✅ DaVinci Resolve workflow defined
- ✅ `createAnnotationJson.py` script created
- ✅ Annotation JSON format specified
- 🔲 **TODO**: Annotate first 3 games

### Documentation
- **CLAUDE.md** - Technical reference (architecture, training, troubleshooting)
- **DATA_PIPELINE.md** - Complete 4-repository workflow
- **ANNOTATION_WORKFLOW.md** - EDL to annotation JSON process
- **TRAINING_APPROACH.md** - CP classifier methodology (strong supervision)
- **PROJECT_STATUS.md** - Progress tracking
- **README.md** - Project overview

---

## 🎯 Current Workflow Overview

```
Fubo TV → fubo-scraper → GetGamesToLocal → CPPreData → DaVinci Resolve → Annotation JSON → Training
```

**4-Repository Pipeline**:

1. **fubo-scraper**: Extract stream URLs (✅ Complete)
2. **GetGamesToLocal**: Download MP4 files (✅ Complete)
3. **CPPreData**: Generate EDL files with silent segment markers (✅ Complete)
4. **ad-game-explainer**: Train models (🔲 Not started)

**Current Stage**: Manual annotation with DaVinci Resolve

---

## 🚀 Next Steps (Priority Order)

### Immediate (This Week)

**1. Run CPPreData on Downloaded Games**
```bash
cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/
uv run process_videos.py --input /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/02-download-games/videos
```
- Expected output: EDL + CSV per game in `manifests/`
- Time: ~5-10 min per game

**2. Annotate First Game in DaVinci Resolve**
1. Import MP4 file
2. Import EDL file (markers appear at ~200-300 silent segments)
3. Review each marker, color-code:
   - **Red (ResolveColorRed)**: TRUE CP (content↔ad boundary)
   - **Other colors**: Non-CP (delete or keep for reference)
4. Export adjusted EDL

**3. Generate Annotation JSON**
```bash
cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/
python createAnnotationJson.py \
  --file_path manifests/{video_name}/{video_name}_marker.edl \
  --use-master-map
```
- Output: `manifests/{video_name}/{video_id}_annotation.json`

**4. Copy Annotation to Training Directory**
```bash
cp CPPreData/manifests/{video_name}/{video_id}_annotation.json \
   ad-game-explainer/data/annotations/
```

### Short Term (Next 2 Weeks)

5. Annotate 2-3 more games (test workflow consistency)
6. Address silent segment annotation gap (for CP classifier training)
7. Extract short clips (10-30s) from annotated segments
8. Annotate remaining 7-12 games for prototype dataset

### Medium Term (Weeks 3-5)

9. Implement CP classifier training
10. Implement Ad classifier training
11. Build end-to-end inference pipeline

---

## 📊 Data Requirements

### For Prototype (10-15 Games)
- **Silent Segments**: ~2000-4500 total annotations
  - ~100-300 TRUE CPs (content↔ad boundaries)
  - ~1800-4200 non-CPs (pauses within content)
- **Short Clips**: 1000-1500 clips (10-30s each)
  - ~500-750 ad clips
  - ~500-750 content clips

### What You Have
- ✅ 150+ NFL games (~450-525 hours)
- ✅ Multiple networks (CBS, NBC, FOX, ESPN)
- ✅ Broadcast quality with embedded commercials
- ✅ **3-4x more data than needed!**

---

## 🔑 Key Concepts

### Two-Level Annotation

**Level 1: Silent Segments** (for CP classifier)
- Label ALL ~200-300 silent segments per game
- Binary: is_CP = 1 (TRUE CP) or 0 (non-CP)
- Used to extract 2s before/after for training pairs

**Level 2: Segments** (for Ad classifier)
- Higher-level: ad breaks vs game content
- Annotate entire commercial breaks (not individual ads)
- Used to extract 10-30s clips for training

### Strong Supervision (Paper's Approach)

**CP Classifier Training**:
- Positive pairs (non-CP): 2s before/after non-CP silent segments
- Negative pairs (CP): 2s before/after TRUE CP boundaries
- Model learns: "Does this silent segment mark a content↔ad transition?"

**Why**: Model at inference compares audio before vs after silent segments, so training must match this structure.

---

## 📁 File Locations

### CPPreData Repository
```
/Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/
├── process_videos.py                    # Detect silent segments
├── createAnnotationJson.py              # EDL → annotation JSON
└── manifests/{video_name}/
    ├── {video_name}.csv                 # All silent segments
    ├── {video_name}_marker.edl          # Original EDL
    ├── TimelineTest.edl                 # Adjusted EDL (from DaVinci)
    └── {video_id}_annotation.json       # Generated annotation
```

### ad-game-explainer Repository
```
/Users/hungmuhamath/projects/GitHub/ad-game-explainer/
├── data/annotations/                    # Copy annotations here
├── data/short_clips/
│   ├── ads/                             # 10-30s ad clips (extracted)
│   └── content/                         # 10-30s content clips (extracted)
├── models/                              # Trained model checkpoints
└── src/                                 # Implementation code
```

---

## 💡 Quick Commands

### Check Downloaded Games
```bash
cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/02-download-games/
wc -l master.csv
grep "downloaded" master.csv | wc -l
```

### Process Single Video
```bash
cd /Users/hungmuhamath/projects/GitHub/ad-game-explainer/pipeline/03-process-videos/
uv run process_videos.py --input /path/to/game.mp4
```

### Generate Annotation from EDL
```bash
python createAnnotationJson.py \
  --file_path manifests/video/video_marker.edl \
  --use-master-map
```

---

## 🎯 Success Criteria

### Prototype (8 weeks target)
- [ ] 10-15 games fully annotated
- [ ] CP classifier trained (AUC ≥0.90)
- [ ] Ad classifier trained (precision/recall ≥90%)
- [ ] End-to-end pipeline working
- [ ] 75-85% F1 score on test set

### Production (3-4 months target)
- [ ] 30-40 games annotated
- [ ] Full audiovisual SlowFast network
- [ ] Post-processing with PANNs
- [ ] 95%+ F1 score
- [ ] <1 min processing time per 20-min video

---

## 🤔 Open Questions / Decisions Needed

1. **Immediate**: Which 3 games to annotate first?
   - Recommend: Different networks, different game types
2. **Short-term**: How to handle silent segment annotations?
   - Current gap: Need is_CP labels for ALL silent segments
3. **Medium-term**: Automation strategy?
   - Claude Workflow Agent for auto-processing?

---

## 🆘 If You Get Stuck

**CPPreData Issues**:
- Check README.md in CPPreData repo
- Verify FFmpeg installed: `ffmpeg -version`
- Check parameter tuning guide

**Annotation Issues**:
- See ANNOTATION_WORKFLOW.md
- Verify EDL color coding (Red = ad boundaries)
- Check createAnnotationJson.py output

**Training Issues**:
- See TRAINING_APPROACH.md
- Check CLAUDE.md troubleshooting section

**General**:
- Read DATA_PIPELINE.md for complete workflow
- Check PROJECT_STATUS.md for current progress
- Ask Claude Code for help (has full context)

---

## 📚 Documentation Map

**Start Here**:
1. This file (GETTING_STARTED.md) - Overview and next steps
2. DATA_PIPELINE.md - Complete 4-repo workflow

**When Annotating**:
3. ANNOTATION_WORKFLOW.md - EDL to JSON process
4. CPPreData/README.md - Silent segment detection

**When Training**:
5. TRAINING_APPROACH.md - CP classifier methodology
6. CLAUDE.md - Complete technical reference
7. Technical_Info/ - Paper analysis and architecture

**For Tracking**:
8. PROJECT_STATUS.md - Progress and weekly updates

---

## 🏈 Why NFL is Perfect for This Project

**Advantages over gaming streams**:
- ✅ Clear ad transitions (crowd noise → silence → ads)
- ✅ Predictable patterns (timeouts, quarters, 2-min warning)
- ✅ Matches paper's sports testing domain (golf)
- ✅ Higher expected accuracy: 95%+ vs 85-90% for gaming
- ✅ Real-world business value (broadcast metrics)

---

**You're ready to start!** Begin by running CPPreData on downloaded games, then annotate your first game in DaVinci Resolve.

Good luck! 🏈
