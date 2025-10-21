# Session Management Module

## Overview

The **Session Management Module** is VolleySense's stateful anchor — the sandbox where all analysis, video clips, statistics, and visualizations converge and persist. Every agent and module reads from and writes to the session context, making it the foundation for continuity, cross-module communication, and long-term analytics.

## Architecture

```
session/
├── api.py           # FastAPI router for RESTful CRUD operations
├── service.py       # Business logic and database operations
├── store.py         # SQLite database layer
├── schemas.py       # Pydantic models for validation
├── ops.py           # Rollup computation and indexing utilities
├── README.md        # This file
└── INTEGRATION_NOTES.md  # Integration guide
```

## Core Concepts

### Session

A **Session** represents a single volleyball match or training session. It includes:

- **Metadata**: Title, venue, date, notes
- **Teams**: Two teams (A and B) with rosters
- **Clips**: Video clips associated with the session
- **Events**: Time-stamped events (serves, attacks, digs, etc.)
- **Statistics**: Aggregated rollup data for analysis
- **Artifacts**: Generated files (heatmaps, exports, etc.)

### Data Model

```python
Session {
    id: UUID
    created_utc: timestamp
    title: str
    venue: str (optional)
    meta: dict (arbitrary metadata)
}

Team {
    side: "A" | "B"
    name: str (optional)
    roster: List[Player]
}

Clip {
    id: UUID
    path: str
    duration_sec: float
    meta: dict
}

Event {
    t_sec: float
    event_type: str
    actor_team: str (optional)
    actor_number: str (optional)
    actor_resolved_name: str (optional)
    payload: dict
}
```

## Features

### 1. CRUD Operations

Full Create, Read, Update, Delete operations for sessions:

- **Create**: Initialize new sessions with metadata
- **Read**: Load complete session details including all related data
- **Update**: Modify session metadata
- **Delete**: Remove sessions with automatic backup

### 2. Historical Session Management

- **Pagination**: List sessions with configurable limits and offsets
- **Search**: Full-text search across titles, venues, and IDs
- **Filtering**: Filter by date, teams, event types

### 3. Resilience & Backups

- **Automatic Snapshots**: Point-in-time backups on save
- **Backup on Delete**: Sessions backed up before deletion
- **Snapshot Cleanup**: Automatic pruning of old backups (keeps latest 10)
- **Recovery**: Restore sessions from snapshots

### 4. Statistics & Rollups

- **Event Aggregation**: Automatic computation of statistics
- **Player Stats**: Per-player event counts and metrics
- **Team Stats**: Team-level performance summaries
- **Advanced Metrics**: Attack efficiency, serve performance, etc.

### 5. Artifact Management

Attach and manage generated files:

- Heatmaps
- CSV exports
- Video overlays
- Analysis reports

## API Endpoints

### Session CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/_api/sessions` | Create new session |
| GET    | `/_api/sessions` | List sessions (with pagination/search) |
| GET    | `/_api/sessions/{id}` | Get session details |
| PATCH  | `/_api/sessions/{id}` | Update session metadata |
| DELETE | `/_api/sessions/{id}` | Delete session |

### Artifacts & Exports

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/_api/sessions/{id}/artifacts` | Attach artifact to session |
| POST   | `/_api/sessions/{id}/export` | Export session to CSV |

### UI Endpoints (HTMX)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/_api/sessions/ui/dashboard` | Session dashboard partial |
| GET    | `/_api/sessions/ui/editor/{id}` | Session editor partial |
| GET    | `/_api/sessions/ui/create` | Create session form |

## Usage Examples

### Creating a Session

**Via API (JSON):**

```python
import requests

response = requests.post(
    "http://localhost:7860/_api/sessions",
    json={
        "title": "Championship Final 2024",
        "venue": "Main Arena",
        "meta": {
            "division": "A",
            "season": "2024"
        }
    }
)

session_id = response.json()["session_id"]
```

**Via Service Layer:**

```python
from session.service import SessionService
from session.schemas import SessionCreate

service = SessionService(db_url, sessions_dir)
session_id = service.create_session(
    SessionCreate(
        title="Championship Final 2024",
        venue="Main Arena",
        meta={"division": "A"}
    )
)
```

### Loading a Session

```python
session_data = service.load_session(session_id)

# Access session details
print(session_data["session"].title)
print(session_data["clips"])
print(session_data["teams"])
print(session_data["rollup_summary"])
```

### Adding Events

```python
from session.schemas import ClipCreate, EventIn

# Add a video clip
clip_id = service.add_clip(
    session_id,
    ClipCreate(path="match.mp4", duration_sec=120.0)
)

# Add events
events = [
    EventIn(
        t_sec=5.0,
        event_type="serve",
        actor_team="A",
        actor_number="10"
    ),
    EventIn(
        t_sec=8.5,
        event_type="attack",
        actor_team="B",
        actor_number="7"
    ),
]

service.add_events(session_id, clip_id, events)
```

### Searching Sessions

```python
# Search by title
sessions = service.list_sessions(search="Championship")

# Pagination
page_1 = service.list_sessions(limit=10, offset=0)
page_2 = service.list_sessions(limit=10, offset=10)
```

### Exporting Data

```python
# Export to CSV
export_dir = "/path/to/exports"
result = service.export_csv(session_id, export_dir)

print(result["events"])  # Path to events CSV
print(result["rollup"])  # Path to rollup CSV
```

### Attaching Artifacts

```python
# Attach a heatmap
artifact_id = service.attach_artifact(
    session_id=session_id,
    artifact_type="heatmap",
    file_path="/path/to/heatmap.png",
    metadata={"description": "Ball trajectory heatmap"}
)

# List artifacts
artifacts = service.get_session_artifacts(session_id)
```

## Database Schema

The session module uses SQLite with the following tables:

### sessions
- `id` (TEXT, PRIMARY KEY)
- `created_utc` (TIMESTAMP)
- `title` (TEXT)
- `venue` (TEXT)
- `meta` (JSON)

### teams
- `id` (INTEGER, PRIMARY KEY)
- `session_id` (TEXT, FK → sessions)
- `side` (TEXT: "A" or "B")
- `name` (TEXT)

### roster
- `id` (INTEGER, PRIMARY KEY)
- `team_id` (INTEGER, FK → teams)
- `number` (TEXT)
- `player_name` (TEXT)

### clips
- `id` (TEXT, PRIMARY KEY)
- `session_id` (TEXT, FK → sessions)
- `path` (TEXT)
- `duration_sec` (REAL)
- `meta` (JSON)

### events
- `id` (INTEGER, PRIMARY KEY)
- `session_id` (TEXT, FK → sessions)
- `clip_id` (TEXT, FK → clips)
- `t_sec` (REAL)
- `event_type` (TEXT)
- `actor_team` (TEXT)
- `actor_number` (TEXT)
- `actor_resolved_name` (TEXT)
- `payload` (JSON)

### stats_rollup
- `id` (INTEGER, PRIMARY KEY)
- `session_id` (TEXT, FK → sessions)
- `team_side` (TEXT)
- `number` (TEXT)
- `event_type` (TEXT)
- `count` (INTEGER)

## Backup & Recovery

### Automatic Backups

Backups are stored in `sessions/_backups/` with the format:

```
{session_id}_{timestamp}.json
```

### Creating a Snapshot

```python
# Manual snapshot
service.save_session(session_id, auto_backup=True)
```

### Restoring from Backup

```python
from session.ops import SessionBackup

backup_manager = SessionBackup(sessions_dir / "_backups")

# List snapshots
snapshots = backup_manager.list_snapshots(session_id)

# Restore
snapshot_path = Path(snapshots[0]["path"])
restored_data = backup_manager.restore_snapshot(snapshot_path)
```

## Rollup & Statistics

### Built-in Rollup Computation

The system automatically computes rollup statistics when events are added:

```python
rollup = service.get_rollup(session_id)

for entry in rollup:
    print(f"{entry.team_side}-{entry.number}: {entry.event_type} x{entry.count}")
```

### Advanced Metrics

Use the `RollupComputer` for advanced analytics:

```python
from session.ops import RollupComputer

computer = RollupComputer()

# Get advanced metrics
metrics = computer.compute_advanced_metrics(events, rosters)

print(metrics["attack_efficiency"])
print(metrics["serve_performance"])
```

## Testing

Run comprehensive tests:

```bash
pytest tests/test_session_crud.py -v
```

Test coverage includes:
- Session creation and validation
- Pagination and search
- CRUD operations
- Backup and recovery
- Artifact management
- Rollup computation

## Best Practices

1. **Always use SessionService**: Don't access the database directly
2. **Enable backups**: Keep `backup=True` when deleting sessions
3. **Pagination**: Use pagination for large session lists
4. **Validation**: Let Pydantic schemas handle validation
5. **Error handling**: Catch `NotFoundError` for missing sessions
6. **Transactions**: All database operations use automatic transactions

## Configuration

Session management respects these settings:

```python
# In app/settings.py
sessions_dir: Path = Path.cwd() / "sessions"
db_url: str | None = None  # Defaults to SQLite in sessions_dir
```

Environment variables:

```bash
VOLLEYSENSE_SESSIONS_DIR=/path/to/sessions
VOLLEYSENSE_DB_URL=sqlite:///path/to/db.sqlite
```

## Performance Considerations

- **SQLite Limitations**: Suitable for single-instance deployments
- **File I/O**: Large sessions may have filesystem overhead
- **Backups**: Automatic cleanup prevents disk bloat
- **Indexes**: Consider adding indexes for large deployments
- **Pagination**: Always use pagination for UI listings

## Future Enhancements

- [ ] PostgreSQL support for multi-instance deployments
- [ ] Real-time session synchronization
- [ ] Session templates and presets
- [ ] Advanced filtering and querying
- [ ] Session comparison and diff views
- [ ] Collaborative session editing

## Support

For issues, questions, or contributions, see:
- Integration guide: `INTEGRATION_NOTES.md`
- Tests: `tests/test_session_crud.py`
- API reference: `api.py`

## License

Part of the VolleySense project.
