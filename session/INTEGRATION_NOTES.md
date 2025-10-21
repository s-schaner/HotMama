# Session Module Integration Notes

## Quick Start Integration

This guide shows how to integrate the Session Management module into your VolleySense workflows, plugins, and custom agents.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Basic Integration](#basic-integration)
3. [Plugin Integration](#plugin-integration)
4. [UI Integration](#ui-integration)
5. [Event Workflow](#event-workflow)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Dependencies

The Session module requires:

- Python 3.10+
- FastAPI
- Pydantic v2
- SQLite3 (built-in)

All dependencies are included in the main VolleySense installation.

### Settings

Ensure your `.env` file or environment has:

```bash
VOLLEYSENSE_SESSIONS_DIR=/path/to/sessions
# Optional: custom database URL
VOLLEYSENSE_DB_URL=sqlite:///path/to/volleysense.db
```

---

## Basic Integration

### Getting the SessionService

```python
from app.settings import get_settings
from session.service import SessionService

# In your module/plugin
settings = get_settings()
db_url = settings.db_url or f"sqlite:///{settings.sessions_dir / 'volleysense.db'}"
service = SessionService(db_url, settings.sessions_dir)
```

### Dependency Injection (FastAPI)

```python
from fastapi import Depends
from session.api import get_session_service

@app.get("/my-endpoint")
async def my_endpoint(service: SessionService = Depends(get_session_service)):
    sessions = service.list_sessions()
    return {"sessions": sessions}
```

---

## Plugin Integration

### Using Session Context in Plugins

Your plugin can read and write session data via the standard hooks:

```python
# plugins/my_plugin/plugin.py

from typing import Dict, List
from session.service import SessionService
from session.schemas import EventIn

class MyAnalysisPlugin:
    name = "my_analysis"
    version = "1.0.0"
    provides_ui = False

    def __init__(self):
        self.service: SessionService = None

    def on_register(self, ctx) -> None:
        """Called when plugin is registered."""
        self.service = ctx.session_service

    def on_clip_ingested(
        self,
        session_id: str,
        clip_id: str,
        clip_path: str
    ) -> dict:
        """
        Called after a video clip is added to a session.

        Return events to be added to the session.
        """
        # Perform your analysis
        detected_events = self.analyze_clip(clip_path)

        # Format as EventIn objects
        events = [
            EventIn(
                t_sec=evt["time"],
                event_type=evt["type"],
                actor_team=evt.get("team"),
                actor_number=evt.get("number"),
                payload=evt.get("metadata", {})
            )
            for evt in detected_events
        ]

        return {
            "success": True,
            "events": events,
            "metadata": {"analyzed_by": self.name}
        }

    def analyze_clip(self, clip_path: str) -> List[dict]:
        """Your custom analysis logic."""
        # Example: detect serves
        return [
            {"time": 5.0, "type": "serve", "team": "A", "number": "10"},
            {"time": 12.5, "type": "attack", "team": "B", "number": "7"},
        ]
```

### Registering Your Plugin

```python
# In your plugin's __init__.py

from plugins.my_plugin.plugin import MyAnalysisPlugin

def register_plugin():
    return MyAnalysisPlugin()
```

---

## UI Integration

### Adding Session UI to Your Templates

Include the session dashboard in your page:

```html
<!-- In your template -->
<div
  hx-get="/_api/sessions/ui/dashboard"
  hx-trigger="load"
  hx-swap="innerHTML">
  Loading sessions...
</div>
```

### Creating Sessions from Your UI

```html
<button
  class="btn btn-primary"
  hx-post="/_api/sessions"
  hx-vals='{"title": "My Session", "venue": "Court 1"}'
  hx-swap="none"
  hx-on::after-request="showToast('Session created!', 'success')">
  Create Session
</button>
```

### Linking to Session Editor

```html
<a
  hx-get="/_api/sessions/ui/editor/{{ session.id }}"
  hx-target="#modal-content"
  hx-swap="innerHTML"
  class="btn btn-sm">
  View Session
</a>
```

---

## Event Workflow

### Complete Event Processing Flow

Here's how events flow through the system:

```
1. Video Upload
   ↓
2. Clip Ingestion (SessionService.add_clip)
   ↓
3. Plugin Hooks (on_clip_ingested)
   ↓
4. Event Detection (your analysis)
   ↓
5. Event Submission (SessionService.add_events)
   ↓
6. Automatic Rollup Computation
   ↓
7. UI Update (HTMX)
```

### Example: Full Workflow

```python
from pathlib import Path
from session.service import SessionService
from session.schemas import SessionCreate, ClipCreate, EventIn

# Step 1: Initialize service
service = SessionService(db_url, sessions_dir)

# Step 2: Create or load session
session_id = service.create_session(
    SessionCreate(title="Training Session", venue="Gym A")
)

# Step 3: Add video clip
clip_id = service.add_clip(
    session_id,
    ClipCreate(
        path="uploads/training_rally.mp4",
        duration_sec=45.0,
        meta={"quality": "1080p"}
    )
)

# Step 4: Analyze and extract events
# (This would be done by your analysis module/plugin)
events = [
    EventIn(
        t_sec=2.0,
        event_type="serve",
        actor_team="A",
        actor_number="12",
        payload={"serve_type": "jump"}
    ),
    EventIn(
        t_sec=5.5,
        event_type="attack",
        actor_team="B",
        actor_number="8",
        payload={"attack_zone": "4"}
    ),
]

# Step 5: Submit events
service.add_events(session_id, clip_id, events)

# Step 6: Get computed rollup
rollup = service.get_rollup(session_id)

# Step 7: Export or use data
export_result = service.export_csv(session_id, "exports/")
print(f"Exported to: {export_result['events']}")
```

---

## Common Patterns

### Pattern 1: Session-Scoped Analysis

Execute analysis within a specific session context:

```python
class SessionAnalyzer:
    def __init__(self, service: SessionService, session_id: str):
        self.service = service
        self.session_id = session_id
        self.session_data = service.load_session(session_id)

    def analyze(self):
        """Run analysis on all clips in the session."""
        for clip in self.session_data["clips"]:
            events = self.process_clip(clip)
            self.service.add_events(
                self.session_id,
                clip.id,
                events
            )

    def get_summary(self):
        """Get analysis summary."""
        return self.session_data["rollup_summary"]
```

### Pattern 2: Batch Session Processing

Process multiple sessions:

```python
def process_all_sessions(service: SessionService):
    """Process all sessions in batches."""
    offset = 0
    batch_size = 10

    while True:
        sessions = service.list_sessions(limit=batch_size, offset=offset)
        if not sessions:
            break

        for session in sessions:
            try:
                process_session(service, session.id)
            except Exception as e:
                logger.error(f"Failed to process {session.id}: {e}")

        offset += batch_size
```

### Pattern 3: Real-Time Updates

Use HTMX for real-time session updates:

```html
<!-- Auto-refresh session stats every 5 seconds -->
<div
  hx-get="/_api/sessions/{{ session_id }}"
  hx-trigger="every 5s"
  hx-swap="outerHTML">
  {{ session.title }} - {{ rollup_summary.total_events }} events
</div>
```

### Pattern 4: Artifact Pipeline

Generate and attach artifacts:

```python
def generate_heatmap_artifact(
    service: SessionService,
    session_id: str,
    clip_id: str
):
    """Generate and attach heatmap artifact."""
    # Load session data
    session_data = service.load_session(session_id)

    # Generate heatmap (your logic here)
    heatmap_path = generate_heatmap(session_data, clip_id)

    # Attach to session
    artifact_id = service.attach_artifact(
        session_id=session_id,
        artifact_type="heatmap",
        file_path=str(heatmap_path),
        metadata={
            "clip_id": clip_id,
            "generated_at": datetime.utcnow().isoformat()
        }
    )

    return artifact_id
```

### Pattern 5: Cross-Session Analytics

Compare multiple sessions:

```python
from collections import defaultdict

def compare_sessions(service: SessionService, session_ids: List[str]):
    """Compare statistics across multiple sessions."""
    comparison = defaultdict(lambda: defaultdict(int))

    for session_id in session_ids:
        rollup = service.get_rollup(session_id)

        for entry in rollup:
            key = f"{entry.team_side}-{entry.number}"
            comparison[key][entry.event_type] += entry.count

    return dict(comparison)
```

---

## Troubleshooting

### Issue: NotFoundError when loading session

**Cause**: Session ID doesn't exist or was deleted.

**Solution**:
```python
from app.errors import NotFoundError

try:
    session = service.load_session(session_id)
except NotFoundError:
    # Handle missing session
    print(f"Session {session_id} not found")
    # Create new session or redirect
```

### Issue: Database locked errors

**Cause**: SQLite doesn't support concurrent writes well.

**Solution**:
```python
# Use transactions properly
with service._connection() as conn:
    # Do all operations within transaction
    conn.execute(...)
    conn.execute(...)
    # Automatic commit on success
```

Or consider PostgreSQL for production:
```bash
VOLLEYSENSE_DB_URL=postgresql://user:pass@localhost/volleysense
```

### Issue: Rollup not updating

**Cause**: Rollup is computed automatically on `add_events()`.

**Solution**:
```python
# Force rollup recomputation
service.rebuild_rollup(session_id)
```

### Issue: Templates not found

**Cause**: Template paths are relative to `webapp/templates/`.

**Solution**:
Ensure templates are in:
```
webapp/templates/session/
├── dashboard.html
├── editor.html
└── _create_form.html
```

### Issue: Session directory permissions

**Cause**: Insufficient permissions on `sessions_dir`.

**Solution**:
```bash
chmod -R 755 /path/to/sessions
chown -R user:user /path/to/sessions
```

---

## Advanced Topics

### Custom Event Types

Define your own event types:

```python
# In your analysis module
CUSTOM_EVENTS = {
    "custom_rotation": "Player rotation detected",
    "timeout_called": "Team timeout",
    "substitution": "Player substitution",
}

# Use in events
event = EventIn(
    t_sec=30.0,
    event_type="custom_rotation",
    payload={"from_position": "MB", "to_position": "RS"}
)
```

### Extending Session Metadata

Store custom metadata in the `meta` field:

```python
session = SessionCreate(
    title="Match",
    meta={
        "tournament": "Nationals 2024",
        "round": "Quarterfinals",
        "weather": "Indoor",
        "court_type": "hardwood",
        "referee": "John Doe",
        "custom_analytics": {
            "ai_model_version": "2.1.0",
            "confidence_threshold": 0.85
        }
    }
)
```

### Database Migrations

For schema changes, consider using Alembic:

```bash
pip install alembic
alembic init migrations
# Edit alembic.ini with your database URL
alembic revision --autogenerate -m "Add custom field"
alembic upgrade head
```

---

## Performance Tips

1. **Use pagination**: Always paginate session lists in UI
2. **Batch operations**: Group database operations when possible
3. **Lazy loading**: Load session details only when needed
4. **Index optimization**: Add indexes for frequently queried fields
5. **Backup cleanup**: Regular cleanup prevents disk bloat

---

## Security Considerations

1. **Input validation**: All inputs validated via Pydantic
2. **SQL injection**: Parameterized queries prevent injection
3. **File uploads**: Validate file types and sizes
4. **Session IDs**: Use UUIDs to prevent enumeration
5. **Backups**: Ensure backup directory has restricted permissions

---

## Testing Your Integration

```python
# tests/test_my_integration.py

from pathlib import Path
import pytest
from session.service import SessionService
from session.schemas import SessionCreate

def make_service(tmp_path: Path) -> SessionService:
    db_url = f"sqlite:///{tmp_path/'test.db'}"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return SessionService(db_url, sessions_dir)

def test_my_plugin_integration(tmp_path):
    service = make_service(tmp_path)
    session_id = service.create_session(
        SessionCreate(title="Test")
    )

    # Test your integration
    result = my_plugin_function(service, session_id)

    assert result["success"]
    assert len(result["events"]) > 0
```

---

## Getting Help

- **Documentation**: See `README.md` for API reference
- **Tests**: Check `tests/test_session_crud.py` for examples
- **Source**: Review `session/service.py` for implementation details
- **Issues**: Report bugs via GitHub issues

---

## Migration Guide

### From Legacy Session System

If migrating from an older session system:

1. **Export existing data** to JSON
2. **Create new sessions** via API
3. **Import events** using `add_events()`
4. **Verify rollup** statistics
5. **Update UI** to use new endpoints

Example migration script:

```python
def migrate_legacy_session(legacy_data: dict, service: SessionService):
    # Create new session
    session_id = service.create_session(
        SessionCreate(
            title=legacy_data["name"],
            venue=legacy_data.get("location"),
            meta=legacy_data.get("metadata", {})
        )
    )

    # Migrate clips
    for clip in legacy_data["clips"]:
        clip_id = service.add_clip(
            session_id,
            ClipCreate(
                path=clip["file"],
                duration_sec=clip["duration"]
            )
        )

        # Migrate events
        events = convert_legacy_events(clip["events"])
        service.add_events(session_id, clip_id, events)

    return session_id
```

---

## Best Practices Checklist

- [ ] Use `SessionService` instead of direct database access
- [ ] Enable backups when deleting sessions
- [ ] Implement pagination for session lists
- [ ] Validate inputs with Pydantic schemas
- [ ] Handle `NotFoundError` exceptions
- [ ] Use transactions for multi-step operations
- [ ] Clean up temp files and old backups
- [ ] Test integrations with temporary databases
- [ ] Document custom event types
- [ ] Version your metadata schemas

---

## Changelog

### v1.0.0 (Initial Release)
- Full CRUD operations
- Pagination and search
- Backup and recovery
- Artifact management
- Rollup computation
- UI templates with HTMX
- Comprehensive test suite
