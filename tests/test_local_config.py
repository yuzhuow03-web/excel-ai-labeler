import json

from local_config import clear_settings, load_settings, protect_secret, save_settings, unprotect_secret


def test_windows_secret_round_trip_does_not_store_plaintext():
    secret = "unit-test-secret-value"
    encrypted = protect_secret(secret)
    assert encrypted
    assert secret not in encrypted
    assert unprotect_secret(encrypted) == secret


def test_settings_round_trip_with_encrypted_api_key(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(
        {"model": "test-model", "system_prompt": "test prompt"},
        "sk-test-key",
        True,
        path,
    )
    raw = path.read_text(encoding="utf-8")
    assert "sk-test-key" not in raw
    loaded = load_settings(path)
    assert loaded["model"] == "test-model"
    assert loaded["api_key"] == "sk-test-key"
    assert loaded["remember_api_key"] is True
    assert clear_settings(path) is True
    assert not path.exists()


def test_settings_without_remembering_key_contains_no_secret(tmp_path):
    path = tmp_path / "settings.json"
    save_settings({"model": "test-model"}, "sk-do-not-save", False, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_key_protected"] == ""
    assert "api_key" not in load_settings(path)
