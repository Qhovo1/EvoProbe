# Harbor Agent Evaluation

This directory contains an example Harbor job configuration. It does not include the Harbor framework, benchmark tasks, or run outputs.

The experiment used Harbor 0.5.0, which requires Python 3.12 or newer, and Docker. Install Harbor separately from EvoProbe.

Before running, set the model and provider fields in `agent-evaluation.yaml` and configure the provider credentials as required by Harbor. From the EvoProbe repository root, run:

```bash
harbor run -c harbor/agent-evaluation.yaml -p /path/to/task
```

The configured `jobs_dir` determines where Harbor writes run outputs.
