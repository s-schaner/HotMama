"""
Comprehensive tests for session CRUD operations.

Tests cover:
- Session creation and validation
- Session listing with pagination and search
- Session loading and detail retrieval
- Session updates
- Session deletion with backups
- Artifact attachment
- Export functionality
- Resilience and error handling
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from app.errors import NotFoundError
from session.schemas import ClipCreate, EventIn, RosterEntry, SessionCreate
from session.service import SessionService


def make_service(tmp_path: Path) -> SessionService:
    """Helper to create a SessionService with temp database."""
    db_url = f"sqlite:///{tmp_path/'test.db'}"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return SessionService(db_url, sessions_dir)


class TestSessionCreation:
    """Tests for session creation functionality."""

    def test_create_basic_session(self, tmp_path: Path) -> None:
        """Test creating a basic session with minimal data."""
        service = make_service(tmp_path)
        session_id = service.create_session(SessionCreate(title="Test Match"))

        assert session_id is not None
        assert len(session_id) == 36  # UUID length

        # Verify session directory was created
        session_dir = tmp_path / "sessions" / session_id
        assert session_dir.exists()

    def test_create_session_with_full_metadata(self, tmp_path: Path) -> None:
        """Test creating a session with all metadata fields."""
        service = make_service(tmp_path)
        session_data = SessionCreate(
            title="Championship Final",
            venue="Main Arena",
            meta={"division": "A", "season": "2024"},
        )
        session_id = service.create_session(session_data)

        # Load and verify
        loaded = service.load_session(session_id)
        assert loaded["session"].title == "Championship Final"
        assert loaded["session"].venue == "Main Arena"
        assert loaded["session"].meta["division"] == "A"

    def test_create_session_initializes_teams(self, tmp_path: Path) -> None:
        """Test that creating a session initializes both teams."""
        service = make_service(tmp_path)
        session_id = service.create_session(SessionCreate(title="Test"))

        loaded = service.load_session(session_id)
        assert "A" in loaded["teams"]
        assert "B" in loaded["teams"]


class TestSessionListing:
    """Tests for session listing and search functionality."""

    def test_list_empty_sessions(self, tmp_path: Path) -> None:
        """Test listing when no sessions exist."""
        service = make_service(tmp_path)
        sessions = service.list_sessions()
        assert sessions == []

    def test_list_multiple_sessions(self, tmp_path: Path) -> None:
        """Test listing multiple sessions."""
        service = make_service(tmp_path)

        # Create multiple sessions
        ids = []
        for i in range(5):
            session_id = service.create_session(SessionCreate(title=f"Session {i}"))
            ids.append(session_id)

        sessions = service.list_sessions()
        assert len(sessions) == 5

        # Verify they're ordered by creation time (newest first)
        assert sessions[0].title == "Session 4"

    def test_list_sessions_with_pagination(self, tmp_path: Path) -> None:
        """Test pagination of session listing."""
        service = make_service(tmp_path)

        # Create 10 sessions
        for i in range(10):
            service.create_session(SessionCreate(title=f"Session {i}"))

        # Test pagination
        first_page = service.list_sessions(limit=3, offset=0)
        assert len(first_page) == 3

        second_page = service.list_sessions(limit=3, offset=3)
        assert len(second_page) == 3

        # Verify no overlap
        first_ids = {s.id for s in first_page}
        second_ids = {s.id for s in second_page}
        assert first_ids.isdisjoint(second_ids)

    def test_search_sessions_by_title(self, tmp_path: Path) -> None:
        """Test searching sessions by title."""
        service = make_service(tmp_path)

        service.create_session(SessionCreate(title="Championship Finals"))
        service.create_session(SessionCreate(title="Friendly Match"))
        service.create_session(SessionCreate(title="Championship Semifinals"))

        results = service.list_sessions(search="Championship")
        assert len(results) == 2
        assert all("Championship" in s.title for s in results)

    def test_search_sessions_by_venue(self, tmp_path: Path) -> None:
        """Test searching sessions by venue."""
        service = make_service(tmp_path)

        service.create_session(SessionCreate(title="Match 1", venue="Arena A"))
        service.create_session(SessionCreate(title="Match 2", venue="Arena B"))
        service.create_session(SessionCreate(title="Match 3", venue="Arena A"))

        results = service.list_sessions(search="Arena A")
        assert len(results) == 2


class TestSessionLoading:
    """Tests for loading session details."""

    def test_load_session_with_all_data(self, tmp_path: Path) -> None:
        """Test loading a session returns all related data."""
        service = make_service(tmp_path)

        # Create session with full data
        session_id = service.create_session(
            SessionCreate(title="Full Test", venue="Test Arena")
        )

        # Add roster
        service.set_roster(
            session_id,
            "A",
            [RosterEntry(number="10", player_name="Alice")],
        )

        # Add clip
        clip_id = service.add_clip(
            session_id,
            ClipCreate(path="test.mp4", duration_sec=30.0),
        )

        # Add events
        service.add_events(
            session_id,
            clip_id,
            [EventIn(t_sec=5.0, event_type="serve", actor_team="A", actor_number="10")],
        )

        # Load and verify
        loaded = service.load_session(session_id)

        assert loaded["session"].title == "Full Test"
        assert len(loaded["clips"]) == 1
        assert len(loaded["rosters"]["A"]) == 1
        assert loaded["event_count"] == 1
        assert loaded["rollup_summary"]["total_events"] == 1

    def test_load_nonexistent_session_raises_error(self, tmp_path: Path) -> None:
        """Test loading a nonexistent session raises NotFoundError."""
        service = make_service(tmp_path)

        with pytest.raises(NotFoundError):
            service.load_session("nonexistent-id")


class TestSessionUpdates:
    """Tests for updating session metadata."""

    def test_update_session_title(self, tmp_path: Path) -> None:
        """Test updating session title."""
        service = make_service(tmp_path)
        session_id = service.create_session(SessionCreate(title="Old Title"))

        service.update_session(session_id, {"title": "New Title"})

        loaded = service.load_session(session_id)
        assert loaded["session"].title == "New Title"

    def test_update_session_venue(self, tmp_path: Path) -> None:
        """Test updating session venue."""
        service = make_service(tmp_path)
        session_id = service.create_session(
            SessionCreate(title="Test", venue="Old Venue")
        )

        service.update_session(session_id, {"venue": "New Venue"})

        loaded = service.load_session(session_id)
        assert loaded["session"].venue == "New Venue"

    def test_update_session_metadata(self, tmp_path: Path) -> None:
        """Test updating session metadata."""
        service = make_service(tmp_path)
        session_id = service.create_session(
            SessionCreate(title="Test", meta={"old": "value"})
        )

        service.update_session(
            session_id,
            {"meta": {"new": "data", "updated": True}},
        )

        loaded = service.load_session(session_id)
        assert loaded["session"].meta["new"] == "data"
        assert loaded["session"].meta["updated"] is True

    def test_update_nonexistent_session_raises_error(self, tmp_path: Path) -> None:
        """Test updating a nonexistent session raises NotFoundError."""
        service = make_service(tmp_path)

        with pytest.raises(NotFoundError):
            service.update_session("nonexistent-id", {"title": "New"})


class TestSessionDeletion:
    """Tests for session deletion functionality."""

    def test_delete_session_removes_all_data(self, tmp_path: Path) -> None:
        """Test that deleting a session removes all associated data."""
        service = make_service(tmp_path)

        # Create session with data
        session_id = service.create_session(SessionCreate(title="To Delete"))
        clip_id = service.add_clip(
            session_id,
            ClipCreate(path="test.mp4", duration_sec=10.0),
        )
        service.add_events(
            session_id,
            clip_id,
            [EventIn(t_sec=1.0, event_type="serve")],
        )

        # Delete session
        service.delete_session(session_id, backup=False)

        # Verify it's gone
        with pytest.raises(NotFoundError):
            service.load_session(session_id)

        # Verify directory is deleted
        session_dir = tmp_path / "sessions" / session_id
        assert not session_dir.exists()

    def test_delete_session_creates_backup(self, tmp_path: Path) -> None:
        """Test that deleting a session creates a backup."""
        service = make_service(tmp_path)

        session_id = service.create_session(SessionCreate(title="Backup Test"))

        # Delete with backup
        service.delete_session(session_id, backup=True)

        # Verify backup was created
        backup_dir = tmp_path / "sessions" / "_backups"
        assert backup_dir.exists()

        # Check for backup file
        backups = list(backup_dir.glob(f"{session_id}_*.json"))
        assert len(backups) > 0

    def test_delete_nonexistent_session_raises_error(self, tmp_path: Path) -> None:
        """Test deleting a nonexistent session raises NotFoundError."""
        service = make_service(tmp_path)

        with pytest.raises(NotFoundError):
            service.delete_session("nonexistent-id")


class TestArtifactManagement:
    """Tests for artifact attachment functionality."""

    def test_attach_artifact_to_session(self, tmp_path: Path) -> None:
        """Test attaching an artifact file to a session."""
        service = make_service(tmp_path)
        session_id = service.create_session(SessionCreate(title="Artifact Test"))

        # Create a test file
        test_file = tmp_path / "test_heatmap.png"
        test_file.write_text("fake image data")

        # Attach artifact
        artifact_id = service.attach_artifact(
            session_id=session_id,
            artifact_type="heatmap",
            file_path=str(test_file),
            metadata={"description": "Test heatmap"},
        )

        assert artifact_id is not None

        # Verify artifact was copied
        artifacts_dir = tmp_path / "sessions" / session_id / "artifacts"
        assert artifacts_dir.exists()

        # Check artifact files
        artifact_files = list(artifacts_dir.glob("heatmap_*"))
        assert len(artifact_files) > 0

    def test_get_session_artifacts(self, tmp_path: Path) -> None:
        """Test retrieving all artifacts for a session."""
        service = make_service(tmp_path)
        session_id = service.create_session(SessionCreate(title="Test"))

        # Create and attach multiple artifacts
        for i in range(3):
            test_file = tmp_path / f"artifact_{i}.txt"
            test_file.write_text(f"artifact {i}")

            service.attach_artifact(
                session_id=session_id,
                artifact_type="test",
                file_path=str(test_file),
            )

        # Get artifacts
        artifacts = service.get_session_artifacts(session_id)
        assert len(artifacts) == 3

    def test_attach_artifact_to_nonexistent_session_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test attaching artifact to nonexistent session raises error."""
        service = make_service(tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(NotFoundError):
            service.attach_artifact(
                session_id="nonexistent-id",
                artifact_type="test",
                file_path=str(test_file),
            )


class TestSessionExport:
    """Tests for session export functionality."""

    def test_export_session_to_csv(self, tmp_path: Path) -> None:
        """Test exporting session data to CSV files."""
        service = make_service(tmp_path)

        # Create session with events
        session_id = service.create_session(SessionCreate(title="Export Test"))
        clip_id = service.add_clip(
            session_id,
            ClipCreate(path="test.mp4", duration_sec=20.0),
        )
        service.add_events(
            session_id,
            clip_id,
            [
                EventIn(
                    t_sec=1.0, event_type="serve", actor_team="A", actor_number="10"
                ),
                EventIn(
                    t_sec=5.0, event_type="attack", actor_team="B", actor_number="7"
                ),
            ],
        )

        # Export
        export_dir = tmp_path / "exports"
        result = service.export_csv(session_id, str(export_dir))

        # Verify export files
        assert Path(result["events"]).exists()
        assert Path(result["rollup"]).exists()

        # Check content
        events_csv = Path(result["events"]).read_text()
        assert "serve" in events_csv
        assert "attack" in events_csv


class TestSessionBackupResilience:
    """Tests for backup and resilience features."""

    def test_save_session_creates_snapshot(self, tmp_path: Path) -> None:
        """Test that save_session creates a backup snapshot."""
        service = make_service(tmp_path)

        session_id = service.create_session(SessionCreate(title="Snapshot Test"))

        # Save session (create snapshot)
        service.save_session(session_id, auto_backup=True)

        # Verify snapshot was created
        backup_dir = tmp_path / "sessions" / "_backups"
        snapshots = list(backup_dir.glob(f"{session_id}_*.json"))
        assert len(snapshots) == 1

    def test_save_session_cleanup_old_snapshots(self, tmp_path: Path) -> None:
        """Test that old snapshots are cleaned up."""
        service = make_service(tmp_path)

        session_id = service.create_session(SessionCreate(title="Cleanup Test"))

        # Create multiple snapshots
        for _ in range(15):
            service.save_session(session_id, auto_backup=True)

        # Check that only 10 most recent are kept
        backup_dir = tmp_path / "sessions" / "_backups"
        snapshots = list(backup_dir.glob(f"{session_id}_*.json"))
        assert len(snapshots) <= 10


class TestRollupComputation:
    """Tests for statistics rollup functionality."""

    def test_rollup_summary_with_events(self, tmp_path: Path) -> None:
        """Test rollup summary computation from events."""
        service = make_service(tmp_path)

        session_id = service.create_session(SessionCreate(title="Rollup Test"))
        clip_id = service.add_clip(
            session_id,
            ClipCreate(path="test.mp4", duration_sec=30.0),
        )

        # Add events
        events = [
            EventIn(t_sec=1.0, event_type="serve", actor_team="A", actor_number="10"),
            EventIn(t_sec=2.0, event_type="serve", actor_team="A", actor_number="10"),
            EventIn(t_sec=3.0, event_type="attack", actor_team="B", actor_number="7"),
        ]
        service.add_events(session_id, clip_id, events)

        # Load and check rollup
        loaded = service.load_session(session_id)
        rollup = loaded["rollup_summary"]

        assert rollup["total_events"] == 3
        assert rollup["event_types"]["serve"] == 2
        assert rollup["event_types"]["attack"] == 1
        assert rollup["teams"]["A"]["events"] == 2
        assert rollup["teams"]["B"]["events"] == 1
