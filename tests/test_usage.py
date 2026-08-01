from shared.usage import UsageTracker


def test_records_and_persists_across_instances(tmp_path):
    path = tmp_path / "usage.json"

    tracker = UsageTracker(path=path)
    tracker.record_nim_request()
    tracker.record_nim_request()

    reloaded = UsageTracker(path=path)
    assert reloaded._data["nim_requests"] == 2


def test_warns_once_per_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr("shared.usage.NIM_REQUEST_CAP", 10)
    path = tmp_path / "usage.json"
    tracker = UsageTracker(path=path)

    warnings = [tracker.record_nim_request() for _ in range(8)]
    assert warnings[-1] is not None
    assert "80%" in warnings[-1]
    assert all(w is None for w in warnings[:-1])

    # ratio still under the next threshold, so no repeat warning
    assert tracker.record_nim_request() is None

    warning_at_cap = tracker.record_nim_request()
    assert warning_at_cap is not None
    assert "100%" in warning_at_cap


def test_resets_on_new_month(monkeypatch, tmp_path):
    path = tmp_path / "usage.json"
    monkeypatch.setattr("shared.usage._current_month", lambda: "2026-01")
    tracker = UsageTracker(path=path)
    tracker.record_nim_request()

    monkeypatch.setattr("shared.usage._current_month", lambda: "2026-02")
    fresh = UsageTracker(path=path)
    assert fresh._data["nim_requests"] == 0
