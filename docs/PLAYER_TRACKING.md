# Player Tracking and Analytics Guide

This guide covers the comprehensive player tracking and analytics features implemented in VolleySense.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Components](#components)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Analytics Dashboard](#analytics-dashboard)

## Overview

VolleySense now includes advanced player tracking and analytics capabilities powered by:

- **YOLO (You Only Look Once)** - Real-time player detection
- **Multi-Object Tracking (MOT)** - Persistent player IDs across frames
- **Pose Estimation** - MediaPipe/YOLO-Pose for player pose analysis
- **Video Overlays** - Bounding boxes, poses, trails, and heatmaps
- **Comprehensive Analytics** - Volleyball-specific statistics and metrics

## Features

### 🎯 Real-Time Player Detection

- YOLOv8/YOLOv11 integration for accurate player detection
- Configurable confidence thresholds
- GPU acceleration support
- Optimized for volleyball court scenarios

### 📊 Multi-Object Tracking

- SORT-based tracking algorithm
- Persistent player IDs throughout video
- Motion prediction with velocity tracking
- Optional Kalman filtering for smooth tracking
- Track history for analytics

### 🏃 Pose Estimation

- Per-player pose estimation
- MediaPipe Pose (33 keypoints) or YOLO-Pose (17 keypoints)
- Real-time skeleton overlay
- Pose-based event detection (jumps, spikes, etc.)

### 🎨 Video Overlays

- **Bounding Boxes**: Color-coded by track ID with confidence scores
- **Pose Skeletons**: Full-body pose visualization
- **Movement Trails**: Player movement history
- **Heatmaps**: Court coverage visualization
- **Statistics**: Real-time per-player stats overlay

### 📈 Comprehensive Analytics

#### Offensive Stats
- Kills, Attacks, Attack Efficiency
- Sets, Assists
- Serves, Aces, Serve Efficiency

#### Defensive Stats
- Digs, Dig Percentage
- Blocks (Solo, Assist, Total)
- Receives

#### Errors
- Attack Errors, Serve Errors
- Dig Errors, Block Errors
- Violations (Net, Rotation, Foot Faults)

#### Advanced Metrics
- Kill Percentage
- Attack Efficiency: `(Kills - Errors - Blocked) / Total Attacks`
- Serve Efficiency: `(Aces - Errors) / Total Serves`
- Overall Player Rating

## Installation

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics>=8.0.0` - YOLO models
- `mediapipe>=0.10.0` - Pose estimation
- `opencv-python>=4.8.0` - Video processing
- `torch>=2.0.0` - Deep learning backend
- `filterpy>=1.4.5` - Kalman filtering (optional)

### GPU Support (Recommended)

For optimal performance, install CUDA-enabled PyTorch:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Process Video with Tracking

```python
from ingest.video_pipeline import VideoProcessingPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    yolo_model="yolov8n",  # Options: yolov8n/s/m/l/x, yolov11n/s/m/l/x
    detection_confidence=0.25,
    enable_pose=True,
    enable_overlays=True,
    enable_heatmap=True,
)

# Initialize pipeline
pipeline = VideoProcessingPipeline(config=config)

# Process video
result = pipeline.process_video(
    video_path="path/to/volleyball.mp4",
    output_path="path/to/output_with_overlays.mp4",
)

# Access results
print(f"Processed {result.processed_frames} frames")
print(f"Tracked {len(result.track_histories)} players")

# Get analytics
analytics = pipeline.get_player_analytics(result)
for track_id, stats in analytics.items():
    print(f"Player {track_id}: {stats}")
```

### 2. Use Individual Components

#### YOLO Detection

```python
from ingest.detector_rt import DetectorRunner
import cv2

# Initialize detector
detector = DetectorRunner(
    model="yolov8n",
    confidence_threshold=0.25,
    device="cuda",  # or "cpu"
)

# Detect in single frame
frame = cv2.imread("frame.jpg")
detections = detector.detect([frame])

# Process entire video
detections = detector.detect_video("video.mp4", stream=True)
```

#### Multi-Object Tracking

```python
from ingest.tracker import PlayerTracker

# Initialize tracker
tracker = PlayerTracker(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
)

# Update with detections
tracks = tracker.update(detections)

# Get track history
history = tracker.get_track_history(track_id=1)
all_histories = tracker.get_all_track_histories()
```

#### Pose Estimation

```python
from ingest.pose import PoseEstimator

# Initialize pose estimator
pose_estimator = PoseEstimator(
    model_type="mediapipe",  # or "yolov8-pose"
    model_complexity=1,
    min_detection_confidence=0.5,
)

# Run pose estimation
poses = pose_estimator.infer(
    frames=[frame],
    bboxes=[[bbox1, bbox2, ...]],
    track_ids=[[1, 2, ...]],
)
```

#### Video Overlays

```python
from viz.overlays import VideoOverlay, OverlayConfig

# Configure overlays
config = OverlayConfig(
    draw_boxes=True,
    draw_pose=True,
    draw_trails=True,
    trail_length=30,
)

# Initialize renderer
overlay = VideoOverlay(config=config)

# Render frame
output_frame = overlay.render_frame(
    frame=input_frame,
    tracks=tracks,
    poses=poses,
)
```

## Components

### 1. DetectorRunner (`ingest/detector_rt.py`)

Real-time YOLO-based player detection.

**Key Methods:**
- `detect(frames)` - Detect objects in frames
- `detect_video(video_path)` - Process entire video
- `get_model_info()` - Get model configuration

**Configuration:**
- `model`: YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- `confidence_threshold`: Detection confidence (0.0-1.0)
- `iou_threshold`: NMS IoU threshold
- `device`: Processing device (cpu, cuda, cuda:0, etc.)
- `classes_filter`: List of class IDs to detect (default: [0] for person)

### 2. PlayerTracker (`ingest/tracker.py`)

SORT-based multi-object tracking for persistent player IDs.

**Key Methods:**
- `update(detections)` - Update tracker with new detections
- `get_track_history(track_id)` - Get position history for track
- `get_all_track_histories()` - Get all track histories
- `reset()` - Reset tracker state

**Configuration:**
- `max_age`: Max frames without detection before track dies
- `min_hits`: Min detections before track is confirmed
- `iou_threshold`: IoU threshold for association
- `use_kalman`: Enable Kalman filtering

### 3. PoseEstimator (`ingest/pose.py`)

Pose estimation for tracked players.

**Supported Models:**
- **MediaPipe Pose**: 33 keypoints, accurate, CPU-friendly
- **YOLOv8-Pose**: 17 COCO keypoints, fast, GPU-optimized

**Key Methods:**
- `infer(frames, bboxes, track_ids)` - Run pose estimation
- `close()` - Release resources

**Keypoint Data:**
- `name`: Keypoint name
- `x`, `y`: Coordinates
- `z`: Depth (if available)
- `visibility`: Visibility score (0.0-1.0)
- `confidence`: Detection confidence

### 4. VideoOverlay (`viz/overlays.py`)

Render visual overlays on video frames.

**Overlay Types:**
- Bounding boxes with track IDs
- Pose skeletons with connections
- Movement trails
- Statistics overlay
- Heatmaps

**Key Methods:**
- `render_frame(frame, tracks, poses, stats)` - Render all overlays
- `render_heatmap_overlay(frame, heatmap)` - Add heatmap overlay
- `reset_trails()` - Clear trail history
- `update_stats(track_id, stats)` - Update player stats

### 5. VideoProcessingPipeline (`ingest/video_pipeline.py`)

Integrated pipeline combining all components.

**Features:**
- End-to-end video processing
- Automatic component initialization
- Progress callbacks
- Analytics extraction
- Heatmap generation

**Key Methods:**
- `process_video(video_path, output_path)` - Process entire video
- `process_frame(frame)` - Process single frame (stateful)
- `get_player_analytics(result)` - Extract player analytics
- `reset()` - Reset pipeline state

### 6. StatsAccumulator (`scoring/stats.py`)

Comprehensive volleyball statistics tracking.

**Key Methods:**
- `ingest(events)` - Process events
- `get_player_summary(team, number)` - Get player stats
- `get_track_summary(track_id)` - Get track stats
- `compute_advanced_metrics(track_id)` - Calculate advanced metrics
- `get_all_player_summaries()` - Get all player summaries
- `export_to_csv_format()` - Export to CSV

**Event Types:**
```python
from scoring.stats import EventType

# Offensive
EventType.SERVE, EventType.SERVE_ACE, EventType.KILL
EventType.ATTACK, EventType.ASSIST, EventType.SET

# Defensive
EventType.DIG, EventType.BLOCK, EventType.RECEIVE

# Errors
EventType.ATTACK_ERROR, EventType.SERVE_ERROR
EventType.NET_VIOLATION, EventType.ROTATION_ERROR
```

## API Reference

### Video Processing Endpoint

**POST** `/_api/analytics/process_video`

Process video with tracking and analytics.

**Parameters:**
- `session_id`: Session ID
- `video`: Video file (multipart/form-data)
- `enable_pose`: Enable pose estimation (default: true)
- `enable_overlays`: Enable video overlays (default: true)
- `yolo_model`: YOLO model name (default: "yolov8n")
- `detection_confidence`: Detection confidence threshold (default: 0.25)

**Response:**
```json
{
  "success": true,
  "video_path": "/assets/session_id/clips/video.mp4",
  "output_path": "/assets/session_id/processed/tracked_video.mp4",
  "heatmap_path": "/assets/session_id/heatmaps/heatmap.png",
  "total_frames": 1500,
  "processed_frames": 1500,
  "total_tracks": 12,
  "analytics": {
    "1": {
      "total_frames": 850,
      "avg_x": 320.5,
      "avg_y": 240.2,
      "movement_distance": 1250.5,
      "avg_speed": 45.2
    }
  }
}
```

### Analytics Stats Endpoint

**GET** `/_api/analytics/stats?session_id={session_id}`

Get player statistics for a session.

**Response:**
```json
{
  "players": [
    {
      "track_id": 1,
      "team": "A",
      "number": "7",
      "kills": 15,
      "attacks": 42,
      "attack_efficiency": 0.286,
      "digs": 8,
      "blocks": 3,
      "rating": 52.5
    }
  ]
}
```

### Export Analytics Endpoint

**POST** `/_api/analytics/export`

Export analytics as CSV.

**Parameters:**
- `session_id`: Session ID

**Response:**
- CSV file download

## Configuration

### Environment Variables

Configure tracking features via environment variables (prefix: `VOLLEYSENSE_`):

```bash
# Player tracking
VOLLEYSENSE_YOLO_MODEL=yolov8n
VOLLEYSENSE_DETECTION_CONFIDENCE=0.25
VOLLEYSENSE_DETECTION_DEVICE=cuda

# Tracking
VOLLEYSENSE_TRACKING_MAX_AGE=30
VOLLEYSENSE_TRACKING_MIN_HITS=3
VOLLEYSENSE_TRACKING_IOU_THRESHOLD=0.3

# Pose estimation
VOLLEYSENSE_ENABLE_POSE_ESTIMATION=true
VOLLEYSENSE_POSE_MODEL_TYPE=mediapipe
VOLLEYSENSE_POSE_MODEL_COMPLEXITY=1
VOLLEYSENSE_POSE_MIN_CONFIDENCE=0.5

# Overlays
VOLLEYSENSE_ENABLE_VIDEO_OVERLAYS=true
VOLLEYSENSE_OVERLAY_DRAW_BOXES=true
VOLLEYSENSE_OVERLAY_DRAW_POSE=true
VOLLEYSENSE_OVERLAY_DRAW_TRAILS=true
VOLLEYSENSE_OVERLAY_TRAIL_LENGTH=30

# Analytics
VOLLEYSENSE_ENABLE_PLAYER_ANALYTICS=true
VOLLEYSENSE_ENABLE_HEATMAP_GENERATION=true
VOLLEYSENSE_HEATMAP_KERNEL_SIZE=25
VOLLEYSENSE_HEATMAP_DECAY_FACTOR=0.95
```

### Programmatic Configuration

```python
from app.settings import Settings

settings = Settings(
    yolo_model="yolov8s",
    detection_confidence=0.3,
    enable_pose_estimation=True,
    pose_model_type="yolov8-pose",
)
```

## Analytics Dashboard

### Accessing the Dashboard

Navigate to the Analytics Dashboard via the sidebar or visit:
```
http://localhost:7860/#analytics
```

### Dashboard Features

#### Summary Cards
- Total Players Tracked
- Total Events
- Average Attack Efficiency
- Average Dig Percentage

#### Player Statistics Table
- Track ID, Team, Number
- Kills, Attacks, Attack Efficiency
- Digs, Blocks, Assists
- Serve Aces, Errors
- Overall Rating

#### Team Comparison
- Side-by-side team statistics
- Total kills, digs, blocks, assists, errors

#### Player Detail View
- Click "Details" on any player
- Comprehensive offensive, defensive, serving stats
- Error breakdown

#### Filters and Sorting
- Filter by team
- Filter by player number
- Filter by stat category
- Sort by any metric

#### Export
- Export all stats to CSV
- Download for external analysis

### Dashboard API Integration

The dashboard uses HTMX and Alpine.js for reactive updates:

```javascript
// Refresh stats
await fetch('/_api/analytics/stats?session_id=SESSION_ID');

// Export stats
await fetch('/_api/analytics/export', {
  method: 'POST',
  body: new FormData({session_id: 'SESSION_ID'})
});
```

## Performance Optimization

### Model Selection

| Model | Speed | Accuracy | GPU Required |
|-------|-------|----------|--------------|
| yolov8n | Fastest | Good | No |
| yolov8s | Fast | Better | Recommended |
| yolov8m | Medium | Great | Yes |
| yolov8l | Slow | Excellent | Yes |
| yolov8x | Slowest | Best | Yes |

### Recommendations

- **CPU-only**: Use `yolov8n` with `mediapipe` pose
- **GPU (4GB)**: Use `yolov8s` with `mediapipe` pose
- **GPU (8GB+)**: Use `yolov8m` with `yolov8-pose`
- **Production**: Use `yolov8l` or `yolov8x` with GPU

### Frame Sampling

Process every Nth frame for faster processing:

```python
config = PipelineConfig(
    process_every_n_frames=2,  # Process every 2nd frame
)
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Use smaller model (yolov8n/s instead of l/x)
- Reduce batch size
- Process fewer frames

**Slow Processing**
- Enable GPU acceleration
- Use lighter models
- Disable pose estimation if not needed
- Process every Nth frame

**Poor Tracking**
- Adjust `detection_confidence` threshold
- Increase `tracking_max_age`
- Adjust `tracking_iou_threshold`

**Pose Estimation Failures**
- Ensure players are clearly visible
- Increase `pose_min_confidence`
- Try different pose model type

## Examples

See the `examples/` directory for complete examples:
- `basic_tracking.py` - Basic player tracking
- `advanced_analytics.py` - Full analytics pipeline
- `custom_overlays.py` - Custom overlay rendering
- `realtime_processing.py` - Real-time video processing

## License

This feature is part of VolleySense and is subject to the project's license.

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
