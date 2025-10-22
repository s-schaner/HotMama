# Video LLM Interaction Module

## Overview

The Video LLM Interaction module adds real-time video Q&A capabilities to HotMama, enabling users to ask natural language questions about video content and receive streaming AI-powered responses. The module integrates vision-language models (VLMs) with computer vision models for comprehensive video analysis.

## Features

### 1. **Real-Time Video Q&A**
- Upload video clips directly in the interface
- Ask questions in natural language (e.g., "Who is the last player that touched the ball?")
- Receive streaming responses from vision-language models
- Supports both LM Studio (local) and Hugging Face (cloud) providers

### 2. **Vision Model Integration**
- **YOLOv8**: Fast object detection for real-time analysis
- **Detectron2**: Advanced instance segmentation
- **Pose Estimation**: Skeletal tracking and analysis
- **None**: Pure LLM analysis without CV preprocessing

### 3. **Video Snapshot Analysis**
- Capture frames from video for detailed analysis
- Apply vision models to captured frames
- Send snapshots to both CV pipeline and LLM for dual analysis
- Save annotated frames for reference

### 4. **Intelligent Video Control**
- Natural language video control commands
- Example: "pause when number 8 touches the ball"
- Client-side playback control based on detected conditions
- Frame-accurate event detection

## Architecture

### Module Structure

```
deploy/gui/app/
├── modules/
│   ├── __init__.py
│   ├── base.py                        # Abstract module base class
│   └── video_llm_interaction.py       # Video LLM module implementation
├── client.py                          # VideoLLMClient for streaming queries
├── controller.py                      # Business logic (query, snapshot, control methods)
├── config.py                          # Configuration with LM Studio + HF support
└── interface.py                       # Tab-based UI with module loading
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                    Video LLM Interaction                    │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Video       │   Vision     │   LLM        │   Video        │
│  Player      │   Model      │   Provider   │   Control      │
│              │   Selector   │   Selector   │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
       │               │              │              │
       └───────────────┴──────────────┴──────────────┘
                       │
              ┌────────▼─────────┐
              │  GuiController   │
              └────────┬─────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
  │ Video   │    │ Vision  │    │ LLM     │
  │ Upload  │    │ Models  │    │ Client  │
  │ Storage │    │ (CV2)   │    │ Stream  │
  └─────────┘    └─────────┘    └─────────┘
```

## Configuration

### LM Studio (Local Inference)

Default configuration for local LLM inference at `127.0.0.1:1234`:

```bash
# .env or environment variables
GUI_LMSTUDIO_BASE_URL=http://127.0.0.1:1234
GUI_LMSTUDIO_API_KEY=lm-studio
GUI_LM_VISION_MODEL=qwen/qwen2.5-vl-7b
GUI_LM_PARSER_MODEL=qwen2.5-3b-instruct
GUI_LLM_PROVIDER=lmstudio
```

**Recommended Models:**
- **Vision Model**: `qwen/qwen2.5-vl-7b` - Multimodal vision-language model
- **Parser Model**: `qwen2.5-3b-instruct` - Text-only instruction model

### Hugging Face (Cloud Inference)

For cloud-based inference with Hugging Face Inference API:

```bash
GUI_HUGGINGFACE_API_URL=https://api-inference.huggingface.co/models/YOUR_MODEL
GUI_HUGGINGFACE_API_KEY=your_hf_api_key_here
GUI_HUGGINGFACE_MODEL=Salesforce/blip-vqa-base
GUI_LLM_PROVIDER=huggingface
```

**Compatible Models:**
- `Salesforce/blip-vqa-base` - Image/video question answering
- `microsoft/git-large-vqav2` - Visual question answering
- Any VQA or vision-language model on HF Inference API

### Switching Providers

Users can switch providers dynamically in the UI via the **LLM Provider** radio button:
- **LM Studio (Local)**: Uses local inference server
- **Hugging Face**: Uses cloud API

## Usage Guide

### 1. Basic Video Query

1. Navigate to the **Video LLM Interaction** tab
2. Upload a video clip (MP4, MOV, MKV, AVI)
3. Select a vision model (optional):
   - `YOLOv8` for object detection
   - `Detectron2` for instance segmentation
   - `Pose Estimation` for skeletal analysis
   - `None` for LLM-only analysis
4. Choose LLM provider (LM Studio or Hugging Face)
5. Enter your query in the **Query** box:
   ```
   Who is the last player that touched the ball?
   ```
6. Click **Send Query**
7. Watch the **Reply** box stream the response in real-time

### 2. Video Snapshot Capture

1. Upload a video
2. Select a vision model
3. (Optional) Enter a query about the frame
4. Click **Capture Snapshot**
5. View the captured frame with optional CV overlays
6. Read the analysis in the **Snapshot Analysis** box

**Use Cases:**
- Capture key moments for detailed analysis
- Apply pose estimation to single frames
- Extract frames with object detection overlays
- Send frames to CV pipeline for processing

### 3. Intelligent Video Control

1. Upload a video
2. Enter a control command in the **Control Command** box:
   ```
   pause when number 8 touches the ball
   ```
3. Click **Execute Control**
4. View the detected frame number and timestamp
5. Use the information to navigate the video

**Supported Commands:**
- `pause when [condition]` - Find and pause at specific event
- `play when [condition]` - Resume at event (future)
- Custom conditions parsed via LLM

## API Reference

### GuiController Methods

#### `query_video_llm(video_path, query, vision_model, llm_provider)`

Stream LLM responses for video queries.

**Parameters:**
- `video_path` (str): Path to video file
- `query` (str): Natural language question
- `vision_model` (str): Vision model to use (`yolov8`, `detectron2`, `pose`, `none`)
- `llm_provider` (str): LLM provider (`lmstudio`, `huggingface`)

**Returns:**
- Generator yielding text chunks

**Example:**
```python
for chunk in controller.query_video_llm(
    video_path="/path/to/video.mp4",
    query="What color jersey is the goalkeeper wearing?",
    vision_model="yolov8",
    llm_provider="lmstudio"
):
    print(chunk, end="", flush=True)
```

#### `capture_video_snapshot(video_path, vision_model, query)`

Capture frame with optional CV overlay and LLM analysis.

**Parameters:**
- `video_path` (str): Path to video file
- `vision_model` (str): Vision model overlay
- `query` (str | None): Optional LLM query

**Returns:**
- Tuple of `(snapshot_path, message)`

**Example:**
```python
snapshot_path, message = controller.capture_video_snapshot(
    video_path="/path/to/video.mp4",
    vision_model="pose",
    query="Describe the player's posture"
)
print(f"Saved to: {snapshot_path}")
print(message)
```

#### `execute_video_control(video_path, command)`

Execute intelligent video control command.

**Parameters:**
- `video_path` (str): Path to video file
- `command` (str): Natural language control command

**Returns:**
- Result message with frame/timestamp

**Example:**
```python
result = controller.execute_video_control(
    video_path="/path/to/video.mp4",
    command="pause when player 10 enters the frame"
)
print(result)  # "✅ Condition met at frame 450 (timestamp: 15.00s)"
```

## Implementation Details

### Frame Extraction

The `VideoLLMClient` samples 5 frames uniformly across the video for LLM analysis:

```python
# Sample 5 frames
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = [int(i * total_frames / 5) for i in range(5)]
```

Frames are resized to `512x384` and encoded as base64 JPEG for efficiency.

### Streaming Protocol

**LM Studio:**
- Uses Server-Sent Events (SSE) via `/v1/chat/completions` endpoint
- Streams delta content chunks: `data: {"choices": [{"delta": {"content": "..."}}]}`

**Hugging Face:**
- Uses Inference API streaming endpoint
- Extracts `token.text` or `generated_text` from response

### Vision Model Placeholders

Current implementation includes **placeholder** vision model integration:

- `_apply_pose_skeleton(frame)`: Adds text overlay (future: integrate MediaPipe/OpenPose)
- `_apply_object_detection(frame, model_type)`: Adds text overlay (future: integrate YOLO/Detectron2)
- `_find_condition_frame(video_path, condition)`: Returns middle frame (future: real detection)

**Roadmap:**
- Integrate YOLOv8 via `ultralytics` package
- Add Detectron2 via `detectron2` package
- Implement MediaPipe for pose estimation
- Add real-time object tracking

## Extending the Module

### Adding New Vision Models

1. Update `vision_model_selector` choices in `video_llm_interaction.py`:
   ```python
   vision_model_selector = gr.Dropdown(
       label="Vision Model",
       choices=[
           # ... existing choices ...
           ("Custom Model - Description", "custom_model"),
       ],
       value="yolov8",
   )
   ```

2. Add handler in `GuiController._apply_custom_model()`:
   ```python
   def _apply_custom_model(self, frame):
       # Your model implementation
       return processed_frame
   ```

3. Update `capture_video_snapshot()` to call your handler:
   ```python
   elif vision_model == "custom_model":
       frame = self._apply_custom_model(frame)
       message = "✅ Snapshot captured with custom model."
   ```

### Adding New LLM Providers

1. Update `config.py` with provider settings:
   ```python
   custom_provider_url: str | None = Field(
       default=None, alias="GUI_CUSTOM_PROVIDER_URL"
   )
   ```

2. Extend `VideoLLMClient._query_custom_provider_stream()`:
   ```python
   def _query_custom_provider_stream(self, query: str, frames_b64: list[str]):
       # Your provider implementation
       yield from response_stream
   ```

3. Update `query_video_stream()` to support new provider:
   ```python
   elif self._provider == "custom_provider":
       yield from self._query_custom_provider_stream(query, frames_data)
   ```

## Troubleshooting

### Issue: "OpenCV is required but not installed"

**Solution:** Install OpenCV in the GUI container:
```bash
pip install opencv-python-headless
```

Or add to `deploy/gui/requirements.gui.txt`:
```
opencv-python-headless==4.9.0.80
```

### Issue: "LM Studio is not configured"

**Solution:** Ensure LM Studio is running and set the base URL:
```bash
export GUI_LMSTUDIO_BASE_URL=http://127.0.0.1:1234
```

Or update `.env`:
```
GUI_LMSTUDIO_BASE_URL=http://127.0.0.1:1234
```

### Issue: Streaming response is slow

**Solution:**
1. Reduce number of frames analyzed (default is 5)
2. Lower frame resolution in `_extract_frames()`
3. Use a faster vision model (e.g., `qwen2.5-3b-instruct` instead of `qwen2.5-vl-7b`)
4. Increase `GUI_REQUEST_TIMEOUT` for longer videos

### Issue: "Hugging Face request failed"

**Solution:**
1. Verify API key: `echo $GUI_HUGGINGFACE_API_KEY`
2. Check model availability on HF Inference API
3. Ensure model supports vision inputs (VQA/VLM models)
4. Review API rate limits

## Performance Considerations

### Frame Sampling

- **Default**: 5 frames sampled uniformly
- **Trade-off**: More frames = better context, but slower inference
- **Recommendation**: 3-5 frames for most use cases

### Frame Resolution

- **Default**: 512x384 (resized from source)
- **Trade-off**: Higher resolution = better detail, but larger payload
- **Recommendation**: 512x384 for balanced performance

### Model Selection

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `qwen2.5-3b-instruct` | Fast | Good | Text-only, simple queries |
| `qwen/qwen2.5-vl-7b` | Medium | Excellent | Vision + text, detailed analysis |
| `qwen2.5-72b-instruct` | Slow | Best | Complex reasoning (if available) |

### Caching

Future enhancement: Cache extracted frames for repeated queries on same video.

## Future Enhancements

### Planned Features

1. **Real Vision Model Integration**
   - YOLOv8 object detection
   - Detectron2 instance segmentation
   - MediaPipe pose estimation

2. **Enhanced Video Control**
   - Real-time event detection
   - Multi-condition tracking
   - Seek to timestamp

3. **Multi-Frame Analysis**
   - Temporal reasoning across frames
   - Motion detection
   - Action recognition

4. **Batch Processing**
   - Queue multiple videos for analysis
   - Export results to CSV/JSON
   - Batch snapshot extraction

5. **Advanced UI Features**
   - Timeline visualization with events
   - Side-by-side comparison
   - Annotation tools

## Contributing

To contribute to the Video LLM Interaction module:

1. Follow the modular architecture in `deploy/gui/app/modules/`
2. Extend `GuiModule` base class for new modules
3. Add tests in `tests/unit/gui/test_video_llm_module.py`
4. Update this documentation with changes

## License

Same as HotMama project license.

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review logs in `deploy/gui/app/` for errors

---

**Version**: 0.2.0
**Last Updated**: 2025-10-22
**Author**: HotMama Team
