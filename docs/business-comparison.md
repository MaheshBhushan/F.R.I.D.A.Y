# Business-task comparison

`friday compare` records measured executions of three fixed synthetic business
tasks: invoice checking, meeting action extraction and support-ticket triage.
It compares assistant results with manual results using the same prompts and
rubrics. SQLite retains the submitted outputs, evidence notes, model identifiers,
costs, quality checks and recording timestamps. The command works offline and
imports operator-recorded results. It does not execute prompts or call a model.

## Collect evidence

```bash
uv run friday compare tasks > /tmp/business-tasks.json
```

Use each task's prompt and output fields for both methods. Start a stopwatch
before reading the prompt and stop after checking the final answer. Include
reading, execution, retries and checking in both times. Record the assistant's
actual result without fixing incorrect answers. Mark a failed execution as
`failed`; use `null` for output if it produced no usable JSON object. Retain any
partial JSON object to score its correct fields. Describe failures in limitations.

For manual execution, solve the task without an assistant. Keep the expected
answers hidden during both executions. Use independent participants or alternate
the order across trials to reduce practice effects. Give a matching manual and
assistant execution the same `task_id` and `trial_id`. Use a new trial ID for each
repeat. Run all selected tasks and retain unsuccessful attempts.

Create one JSON evidence file per execution. This unmeasured template requires
you to fill in real data before recording:

```json
{
  "suite_version": 1,
  "task_id": "invoice-check",
  "trial_id": "YOUR_TRIAL_ID",
  "method": "assistant",
  "status": "completed",
  "elapsed_seconds": null,
  "output": null,
  "model": "ACTUAL_MODEL_IDENTIFIER",
  "model_cost_usd": null,
  "evidence": "Describe the execution, timer, date, retries and checking procedure.",
  "limitations": []
}
```

Replace `elapsed_seconds` with the stopwatch measurement and `output` with the
returned JSON object. Set `method` to `manual` and `model` to `null` for a manual
execution. Manual executions leave `model_cost_usd` null; the report labels it
not applicable. The tool excludes human labour costs.

For assistant executions, leave `model_cost_usd` null when the charge is unknown.
To record a known charge, provide a nonnegative USD number and a nonempty
`cost_evidence` string. Cite the provider usage/billing record or document the
token counts, applicable rates, cache charges and calculation for the entire
attempt, including retries. Zero requires evidence too. The tool has no built-in
price assumptions and does not infer cost from latency or model name. Evidence
notes must not contain credentials or sensitive business data.

```bash
uv run friday compare record /tmp/manual-observation.json
uv run friday compare record /tmp/assistant-observation.json
uv run friday compare report
uv run friday compare report --json
```

The default database is `~/.friday/business-comparison.sqlite3` (under
`FRIDAY_HOME` when configured). Use `--db /path/to/evidence.sqlite3` on `record`
and `report` to select another file. Duplicate task/trial/method records fail
without replacing evidence. Reports include every recorded trial in the current
suite version and show tasks with no measurements. Reporting does not create a
missing database.

Fixture changes require a new suite version. Keep existing versions' evidence
separate so that different tasks or rubrics never form a comparison pair.

## Interpret the report

Each expected top-level field supplies one exact quality check. A further check
rejects additional top-level fields. Nested objects, arrays, owners and dates
must match exactly; numeric equivalents such as `220` and `220.0` match. The
quality ratio is the number of passed checks divided by the number of checks.
An execution completes the task only if its status is `completed` and all checks
pass. Each task has its own check count. Ratios across tasks should not be treated
as equivalent difficulty.

The report presents quality, completion, elapsed seconds, model cost and
limitations per execution. It calculates assistant minus manual latency only
within the same task and trial when both executions pass every check. A negative
difference means the assistant attempt took less recorded time. Missing manual
results, failed executions and unknown costs remain explicit. Reports do not
pool models or discard failed attempts.

These fixtures measure a narrow set of structured tasks. Exact checks cannot
judge writing quality, tool reliability, employee acceptance or enterprise
deployment readiness. Timings and costs are self-reported. Small samples,
practice effects, participant differences and model changes limit conclusions.
The report excludes voice overhead unless the operator includes it in the timer.
Observed latency differences do not establish productivity gains. No measurements
or business gains ship with this feature; the automated tests use synthetic data.
