from temporal_classifier.compare import format_comparison


def test_format_comparison_reports_both_scores_and_the_delta():
    report = format_comparison(telemetry_only_f1=0.60, vision_enriched_f1=0.75)

    assert "0.60" in report
    assert "0.75" in report
    assert "+0.15" in report or "0.15" in report


def test_format_comparison_reports_the_pooled_vision_arm_when_present():
    report = format_comparison(
        telemetry_only_f1=0.60, vision_enriched_f1=0.75, vision_pooled_f1=0.71
    )

    assert "0.60" in report and "0.75" in report and "0.71" in report
    assert "+0.15" in report
    assert "+0.11" in report


def test_format_comparison_omits_the_pooled_arm_when_not_run():
    report = format_comparison(telemetry_only_f1=0.60, vision_enriched_f1=0.75)

    assert "0.60" in report and "0.75" in report
    assert "pooled" not in report.lower()


def test_format_comparison_signs_a_negative_delta():
    report = format_comparison(telemetry_only_f1=0.75, vision_enriched_f1=0.60)

    assert "-0.15" in report
