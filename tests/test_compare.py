from temporal_classifier.compare import format_comparison


def test_format_comparison_reports_both_scores_and_the_delta():
    report = format_comparison(telemetry_only_f1=0.60, vision_enriched_f1=0.75)

    assert "0.60" in report
    assert "0.75" in report
    assert "+0.15" in report or "0.15" in report
