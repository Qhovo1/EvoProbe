# EvoProbe

This repository contains the core workflows for task generation, diagnostic evaluation, feedback-pattern extraction, and feedback-driven regeneration. The benchmark data and experiment outputs are not included. The instructions below describe how to reproduce the workflows.

## Requirements

- Python 3.10 or newer for the EvoProbe core and Mini-SWE-agent.
- The workflow uses Mini-SWE-agent 2.3.0.
- `requirements.txt` lists the core's Python package dependencies. Install them in the environment used to run EvoProbe:

  ```bash
  python -m pip install -r requirements.txt
  ```

- Docker for containerized challenge execution.
- Harbor is a separate tool and is not installed by `requirements.txt`. The Harbor version used for the experiment is 0.5.0 and requires Python 3.12 or newer; install/run it separately. `harbor/agent-evaluation.yaml` is an example job configuration.
- Access to the model APIs configured in `core/config.py`

The checked-in API key and API base fields are blank. Set the relevant `API_KEY`/`API_BASE` or `DEEPSEEK_API_KEY`/`DEEPSEEK_API_BASE` values in `core/config.py` for the selected model; do not commit credentials.

## Data paths

By default, `DATA_ROOT` in `core/config.py` is the parent directory of the EvoProbe repository. The expected workspace layout is:

```text
<workspace>/
├── EvoProbe/
│   ├── core/
│   ├── harbor/
│   │   ├── agent-evaluation.yaml
│   │   └── README.md
│   ├── README.md
│   └── requirements.txt
├── EvoProbeBench/
│   ├── cybench/
│   └── feedback/
└── question/
```

`EvoProbeBench/cybench/<category>/` contains reference tasks, `EvoProbeBench/feedback/<category>/` contains feedback inputs, and generated tasks are written under `question/`. These datasets are external to this repository. If they are stored elsewhere, update `DATA_ROOT` or the individual directory constants in `core/config.py`.

## Feedback-pattern extraction inputs

Before running `extract_patterns.py`, prepare these JSON files in `EvoProbeBench/feedback/<category>/`:

- `<category>_feedback.json`: a skeleton with `success_patterns` and `failure_patterns` lists. The success list has an empty `success_pattern` and `stage`; failure rows have `failure_category`, `failure_subtype`, and `frequency` filled in, with `failure_pattern` and `stage` left empty.
- `<category>_summary_digest.json`: a digest with a `tasks` list containing `agents`. Each agent should include `success` and `trajectory_summaries`; failed agents should also include `failure_category`, `failure_subtype`, and `failure_analysis.evidence`.

The digest must be prepared before extraction. The script writes `<category>_extract_raw.json` and merges the extracted patterns into the feedback file.

## Workflow overview

Question generation and feedback-driven regeneration use Mini-SWE-agent to create Harbor-format tasks. Solver-agent runs are launched separately with Harbor in Docker; `harbor/agent-evaluation.yaml` is an example job configuration, not the Harbor framework itself. The core `evaluate.py` and `extract_patterns.py` scripts provide diagnostic evaluation and feedback-pattern extraction; pattern extraction expects prepared feedback and digest inputs.

## Workflow entry points

Run commands from the EvoProbe repository root. Replace the example model with a name listed in `core/config.py`.

Task generation:

```bash
python core/generate_question.py --category crypto --model deepseek-v4-pro
```

Diagnostic evaluation of an existing task:

```bash
python core/evaluate.py --task-dir /path/to/task --judge-model deepseek-v4-pro
```

Feedback-driven regeneration:

```bash
python core/generate_question_feedback.py --category crypto --model deepseek-v4-pro
```

Feedback-pattern extraction (requires the category summary digest and feedback files):

```bash
python core/extract_patterns.py --category crypto --model deepseek-v4-pro
```

Generated tasks are placed in a timestamped directory under `question/`; evaluation files are written under the selected task directory. Feedback-driven regeneration uses the category feedback file when it is available. Run any entry point with `--help` to see its options.