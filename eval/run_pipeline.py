"""CLI entrypoint for the LLM-as-Judge evaluation pipeline (Problem 2)."""
from __future__ import annotations

import argparse
import json
import sys
import traceback


def _load_suite(path: str):
    from eval.pipeline.schemas import TestSuite
    if path.endswith((".yaml", ".yml")):
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    return TestSuite.model_validate(data)


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge Evaluation Pipeline (Problem 2)",
    )
    parser.add_argument(
        "--suite", "-s",
        default="eval/suites/sample_suite.yaml",
        help="Path to test suite (JSON or YAML)",
    )
    parser.add_argument(
        "--report", "-r",
        default=None,
        help="Output report path (overrides env)",
    )
    args = parser.parse_args()

    from eval.pipeline.config import pipeline_settings
    if args.report:
        pipeline_settings.report_path = args.report

    suite = _load_suite(args.suite)
    print(f"Loaded suite: {suite.name} ({len(suite.cases)} cases)")
    print(f"Config A: {suite.config_a}")
    print(f"Config B: {suite.config_b}")
    print(f"Judge: {pipeline_settings.judge_provider}/{pipeline_settings.judge_model}")
    print()

    from eval.pipeline.logger import AuditLogger
    from eval.pipeline.pairwise import evaluate_case
    from eval.pipeline.validation import run_validation
    from eval.pipeline.report import (
        compute_bias_metrics, build_suite_report, write_report, print_summary,
    )

    audit_logger = AuditLogger()
    results = []

    for i, case in enumerate(suite.cases, 1):
        try:
            print(f"  [{i}/{len(suite.cases)}] Evaluating: {case.id}...")
            result = evaluate_case(case, audit_logger)
            results.append(result)
            print(f"    -> Winner: {result.final_winner}  (bias: {result.position_bias_detected})")
        except Exception as exc:
            print(f"    -> ERROR: {exc}")
            traceback.print_exc()

    print("\nRunning judge validation (adversarial probes)...")
    validation = run_validation(results, audit_logger)

    bias_metrics = compute_bias_metrics(results, validation)
    report = build_suite_report(suite, results, bias_metrics, validation, audit_logger)

    write_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()
