from deploy.gui.app.storage import StorageManager


def test_save_upload(tmp_path) -> None:
    upload_dir = tmp_path / "uploads"
    download_dir = tmp_path / "downloads"
    storage = StorageManager(upload_dir, download_dir)

    source = tmp_path / "example.mp4"
    source.write_bytes(b"video-bytes")

    stored = storage.save_upload(str(source))
    assert stored.parent == upload_dir
    assert stored.suffix == ".mp4"
    assert stored.read_bytes() == b"video-bytes"


def test_prepare_artifact(tmp_path) -> None:
    upload_dir = tmp_path / "uploads"
    download_dir = tmp_path / "downloads"
    storage = StorageManager(upload_dir, download_dir)

    artifact = tmp_path / "result.json"
    artifact.write_text("{}")

    prepared = storage.prepare_artifact(str(artifact))
    assert prepared.parent == download_dir
    assert prepared.read_text() == "{}"
