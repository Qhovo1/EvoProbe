#!/usr/bin/env python3
"""mini-swe-agent CLI integration module.
Invokes mini-swe-agent through its CLI, matching the Harbor execution path.
"""
import os
import sys
import time
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


def _normalize_content(raw_content: Any) -> str:
    """Normalize message content which may be a string, list of parts, or None."""
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for part in raw_content:
            if isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(raw_content)


def _add_observation_to_last_agent_step(steps: list, content: str, message_index: int) -> None:
    """Add observation content to the most recent agent step."""
    if steps and steps[-1]["source"] == "agent":
        prev_step = steps[-1]
        if "observation" in prev_step and prev_step["observation"].get("results"):
            prev_step["observation"]["results"].append({"content": content})
        else:
            prev_step["observation"] = {"results": [{"content": content}]}


def _build_step_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    prompt_tokens_details: dict,
    completion_tokens_details: dict,
    total_cost_usd: float,
    total_completion_tokens: int,
) -> dict | None:
    """Build metrics for an individual step."""
    if prompt_tokens == 0 and completion_tokens == 0:
        return None

    step_cost = None
    if total_cost_usd > 0 and total_completion_tokens > 0 and completion_tokens > 0:
        step_cost = (completion_tokens / total_completion_tokens) * total_cost_usd

    extra_metrics = {}
    if prompt_tokens_details:
        extra_metrics["prompt_tokens_details"] = prompt_tokens_details
    if completion_tokens_details:
        extra_metrics["completion_tokens_details"] = completion_tokens_details

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens if cached_tokens > 0 else None,
        "cost_usd": step_cost if step_cost and step_cost > 0 else None,
        "extra": extra_metrics if extra_metrics else None,
    }


def _parse_tool_calls(message: dict, content: str, step_id: int) -> tuple[list | None, str | None]:
    """Parse tool calls from an assistant message."""
    message_tool_calls = message.get("tool_calls")
    if not message_tool_calls:
        return None, content if content else None

    tool_calls = []
    for tc in message_tool_calls:
        tc_id = tc.get("id", f"call_{step_id}_{len(tool_calls) + 1}")
        func = tc.get("function") or {}
        func_name = func.get("name", "bash")
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                arguments = {"command": raw_args}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {"command": str(raw_args)}
        tool_calls.append({
            "tool_call_id": tc_id,
            "function_name": func_name,
            "arguments": arguments,
        })

    reasoning = content if content else None
    return tool_calls if tool_calls else None, reasoning


def convert_mini_swe_agent_to_atif(mini_swe_agent_trajectory: dict, session_id: str) -> dict:
    """Convert mini-swe-agent trajectory to ATIF format."""
    info = mini_swe_agent_trajectory.get("info") or {}
    config = info.get("config") or {}
    model_config = config.get("model") or {}
    agent_config = config.get("agent") or {}

    model_name = model_config.get("model_name") or "unknown"
    mini_version = info.get("mini_version") or "unknown"
    trajectory_format = mini_swe_agent_trajectory.get("trajectory_format", "unknown")
    messages = mini_swe_agent_trajectory.get("messages") or []

    steps = []
    step_id = 1

    # Track cumulative tokens
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    total_cost_usd = (info.get("model_stats") or {}).get("instance_cost") or 0.0

    # First pass: count total completion tokens
    for message in messages:
        extra = message.get("extra") or {}
        response_data = extra.get("response") or {}
        usage = response_data.get("usage") or {}
        total_completion_tokens += usage.get("completion_tokens") or 0

    # Process messages
    for i, message in enumerate(messages):
        role = message.get("role")
        content = _normalize_content(message.get("content"))
        extra = message.get("extra") or {}

        response_data = extra.get("response") or {}
        usage = response_data.get("usage") or {}

        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        prompt_tokens_details = usage.get("prompt_tokens_details") or {}
        completion_tokens_details = usage.get("completion_tokens_details") or {}
        cached_tokens = prompt_tokens_details.get("cached_tokens") or 0
        reasoning_tokens = completion_tokens_details.get("reasoning_tokens") or 0

        total_prompt_tokens += prompt_tokens
        total_cached_tokens += cached_tokens
        total_reasoning_tokens += reasoning_tokens

        if role == "system":
            steps.append({
                "step_id": step_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "system",
                "message": content,
            })
            step_id += 1

        elif role == "user":
            if i == 1:
                steps.append({
                    "step_id": step_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "user",
                    "message": content,
                })
                step_id += 1
            else:
                _add_observation_to_last_agent_step(steps, content, i)

        elif role == "tool":
            _add_observation_to_last_agent_step(steps, content, i)

        elif role == "assistant":
            tool_calls, reasoning = _parse_tool_calls(message, content, step_id)
            metrics = _build_step_metrics(
                prompt_tokens, completion_tokens, cached_tokens,
                prompt_tokens_details, completion_tokens_details,
                total_cost_usd, total_completion_tokens
            )

            step = {
                "step_id": step_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "agent",
                "model_name": model_name,
                "message": content,
            }
            if reasoning:
                step["reasoning_content"] = reasoning
            if tool_calls:
                step["tool_calls"] = tool_calls
            if metrics:
                step["metrics"] = metrics
            steps.append(step)
            step_id += 1

    # Build final metrics
    final_extra = {}
    if total_reasoning_tokens > 0:
        final_extra["total_reasoning_tokens"] = total_reasoning_tokens

    final_metrics = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cached_tokens": total_cached_tokens if total_cached_tokens > 0 else None,
        "total_cost_usd": total_cost_usd if total_cost_usd > 0 else None,
        "extra": final_extra if final_extra else None,
    }

    return {
        "schema_version": "ATIF-v1.2",
        "session_id": session_id,
        "agent": {
            "name": "mini-swe-agent",
            "version": mini_version,
            "model_name": model_name,
            "extra": {
                "original_format": trajectory_format,
                "agent_config": agent_config,
            },
        },
        "steps": steps,
        "final_metrics": final_metrics,
        "notes": "Converted from mini-swe-agent trajectory format to ATIF",
    }


def call_mini_swe_agent(
    instruction: str,
    model: str,
    working_dir: str = ".",
    api_key: str = "",
    api_base: str = "",
    timeout: int = 5400,
    output_prefix: str = "",
) -> dict:
    """
    Run a task using the mini-swe-agent CLI.

    Args:
        instruction: Task instruction.
        model: Model name (for example, "claude-sonnet-4-6").
        working_dir: Working directory.
        api_key: API key
        api_base: API base URL
        timeout: Timeout in seconds.
        output_prefix: Output filename prefix to prevent collisions during concurrent runs; empty by default.

    Returns:
        A dictionary containing success, messages, tool_calls, and metrics.
    """
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"mini-swe-agent CLI Caller")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"API Base: {api_base}")
    print(f"Working Dir: {working_dir}")
    print(f"{'='*60}\n")

    # Prepare environment variables.
    env = os.environ.copy()
    env["MSWEA_CONFIGURED"] = "true"  # Disable interactive setup.
    env["MSWEA_COST_TRACKING"] = "ignore_errors"  # Ignore unknown model pricing.
    env["MSWEA_BASH_TIMEOUT"] = "600"  # Bash command timeout: 600 seconds (10 minutes).

    # Set the API key.
    api_base_lower = (api_base or "").lower()

    # Add a provider prefix to the model name for LiteLLM when needed.
    litellm_model_name = model

    if "yunyi" in api_base_lower or "rdzhvip" in api_base_lower:
        # Yunyi relay using the Anthropic format.
        env["ANTHROPIC_API_KEY"] = api_key
        if api_base:
            env["ANTHROPIC_BASE_URL"] = api_base
        # Clear other provider variables to prevent provider misidentification in litellm_model.py.
        env.pop("DEEPSEEK_BASE_URL", None)
        env.pop("DEEPSEEK_API_KEY", None)
        env.pop("OPENAI_API_BASE", None)
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_API_KEY", None)
        # Add anthropic/ prefix if not present
        if not model.startswith("anthropic/"):
            litellm_model_name = f"anthropic/{model}"
    elif "anthropic" in api_base_lower or "claude" in api_base_lower:
        # Official Anthropic API.
        env["ANTHROPIC_API_KEY"] = api_key
        if api_base:
            env["ANTHROPIC_BASE_URL"] = api_base
        # Clear other provider variables to prevent provider misidentification in litellm_model.py.
        env.pop("DEEPSEEK_BASE_URL", None)
        env.pop("DEEPSEEK_API_KEY", None)
        env.pop("OPENAI_API_BASE", None)
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_API_KEY", None)
    elif "deepseek" in api_base_lower:
        # Official DeepSeek API; use DEEPSEEK_* environment variables.
        env["DEEPSEEK_API_KEY"] = api_key
        if api_base:
            env["DEEPSEEK_BASE_URL"] = api_base
        # Clear ANTHROPIC_BASE_URL so litellm_model.py does not mistake this for Anthropic.
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_API_KEY", None)
        if not model.startswith("deepseek/"):
            litellm_model_name = f"deepseek/{model}"
    elif "dashscope" in api_base_lower or "aliyuncs" in api_base_lower:
        # Alibaba Cloud DashScope (Qwen).
        env["DASHSCOPE_API_KEY"] = api_key
        # Clear ANTHROPIC_BASE_URL to prevent LiteLLM from misidentifying the provider.
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_API_KEY", None)
        if not model.startswith("dashscope/"):
            litellm_model_name = f"dashscope/{model}"
    elif "openai" in api_base_lower:
        env["OPENAI_API_KEY"] = api_key
        if api_base:
            env["OPENAI_API_BASE"] = api_base
        # Clear ANTHROPIC_BASE_URL to prevent LiteLLM from misidentifying the provider.
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_API_KEY", None)
    elif api_base:
        # Use OpenAI-compatible mode for custom API endpoints.
        env["OPENAI_API_KEY"] = api_key
        base_url = api_base if api_base.endswith('/v1') else f"{api_base}/v1"
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_BASE"] = base_url
        # Clear other provider variables to prevent provider misidentification in litellm_model.py.
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("DEEPSEEK_BASE_URL", None)
        env.pop("DEEPSEEK_API_KEY", None)
        if not model.startswith("openai/"):
            litellm_model_name = f"openai/{model}"
    else:
        # Default provider setup.
        env["ANTHROPIC_API_KEY"] = api_key
        if api_base:
            env["ANTHROPIC_BASE_URL"] = api_base

    print(f"LiteLLM model name: {litellm_model_name}\n")

    # Prepare output paths.
    working_path = Path(working_dir)

    # Include output_prefix in filenames when set to support concurrent runs.
    if output_prefix:
        trajectory_path = working_path / f"{output_prefix}_trajectory.json"
        trajectory_atif_path = working_path / f"{output_prefix}_trajectory_atif.json"
    else:
        # Use default filenames for backward compatibility.
        trajectory_path = working_path / "trajectory.json"
        trajectory_atif_path = working_path / "trajectory_atif.json"

    # Build the CLI command.
    # Use the current Python environment, which must have mini-swe-agent installed.
    python_executable = sys.executable
    cmd = [
        python_executable,
        "-m", "minisweagent",
        "--yolo",  # Skip confirmation; use InteractiveAgent by default, matching Harbor.
        "--cost-limit=30",  # Set the cost limit to $20, matching mini_swe_caller.py.
        f"--model={litellm_model_name}",
        f"--task={instruction}",  # Pass directly; subprocess handles argument quoting.
        f"--output={trajectory_path}",  # Write the raw trajectory to trajectory.json.
        "--exit-immediately",  # Exit as soon as the task completes.
    ]

    print(f"Executing: {' '.join(cmd[:4])} ...")
    print(f"Task length: {len(instruction)} chars")
    print(f"\nStarting agent execution...\n")

    try:
        # Run the command from the working directory.
        # Use stdin=subprocess.DEVNULL to prevent EOFError.
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            env=env,
            stdin=subprocess.DEVNULL,  # Prevent interactive input from causing EOFError.
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        execution_time = time.time() - start_time

        # Read metrics from the raw trajectory.json file.
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cost = 0.0

        if trajectory_path.exists():
            try:
                trajectory_data = json.loads(trajectory_path.read_text())

                # Extract the cost.
                total_cost = ((trajectory_data.get("info") or {}).get("model_stats") or {}).get(
                    "instance_cost"
                ) or 0.0

                # Sum token usage across messages.
                for msg in trajectory_data.get("messages") or []:
                    usage = ((msg.get("extra") or {}).get("response") or {}).get("usage") or {}

                    prompt_tokens_details = usage.get("prompt_tokens_details") or {}
                    total_cache_read_tokens += prompt_tokens_details.get("cached_tokens") or 0

                    total_input_tokens += usage.get("prompt_tokens") or 0
                    total_output_tokens += usage.get("completion_tokens") or 0

                print(f"\n✓ Token usage (from trajectory.json):")
                print(f"  Input: {total_input_tokens:,}")
                print(f"  Output: {total_output_tokens:,}")
                print(f"  Cache: {total_cache_read_tokens:,}")
                print(f"  Cost: ${total_cost:.4f}")

                # Convert the trajectory to ATIF format.
                print(f"\nConverting to ATIF format...")
                session_id = str(uuid.uuid4())
                atif_trajectory = convert_mini_swe_agent_to_atif(trajectory_data, session_id)

                # Save the ATIF-formatted trajectory.
                trajectory_atif_path.write_text(json.dumps(atif_trajectory, indent=2))

                # Measure the size reduction.
                raw_size = len(trajectory_path.read_text())
                atif_size = len(trajectory_atif_path.read_text())
                raw_messages = len(trajectory_data.get("messages", []))
                atif_steps = len(atif_trajectory.get("steps", []))

                print(f"✓ ATIF conversion complete:")
                print(f"  Original: {raw_messages} messages, {raw_size/1024:.1f}KB (trajectory.json)")
                print(f"  ATIF: {atif_steps} steps, {atif_size/1024:.1f}KB (trajectory_atif.json)")
                print(f"  Reduction: {raw_size/atif_size:.1f}x")

            except Exception as e:
                print(f"\n⚠️ Failed to convert the trajectory: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠️ Trajectory file was not generated: {trajectory_path}")

        # Check the execution result.
        if result.returncode == 0:
            print(f"\n✓ mini-swe-agent execution completed")
            success = True
        else:
            print(f"\n✗ mini-swe-agent failed with exit code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            success = False

        # Extract messages from the raw trajectory.
        messages = []
        if trajectory_path.exists():
            try:
                trajectory_data = json.loads(trajectory_path.read_text())
                messages = trajectory_data.get("messages") or []
            except:
                pass

        # Build the result in the format expected by the generation scripts.
        return {
            "success": success,
            "messages": messages,
            "tool_calls": [],  # Tool calls are not extracted in CLI mode.
            "metrics": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cache_read_tokens": total_cache_read_tokens,
                "cost_usd": total_cost,
                "duration_sec": execution_time,
                "n_calls": len([m for m in messages if m.get("role") == "assistant"]),
            },
            "stdout": result.stdout,
            "stderr": result.stderr,
            "config_dir": str(Path(working_dir)),
        }

    except subprocess.TimeoutExpired:
        print(f"\n✗ Timeout after {timeout} seconds")
        return {
            "success": False,
            "messages": [],
            "tool_calls": [],
            "metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cache_read_tokens": 0,
                "cost_usd": 0.0,
                "duration_sec": time.time() - start_time,
                "n_calls": 0,
            },
            "stdout": "",
            "stderr": "Timeout",
            "config_dir": "",
            "error": "Timeout",
        }

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "messages": [],
            "tool_calls": [],
            "metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cache_read_tokens": 0,
                "cost_usd": 0.0,
                "duration_sec": time.time() - start_time,
                "n_calls": 0,
            },
            "stdout": "",
            "stderr": str(e),
            "config_dir": "",
            "error": str(e),
        }


if __name__ == "__main__":
    # Test.
    sys.path.insert(0, str(Path(__file__).parent))
    import config
    from models import get_model_config

    model_name = "qwen3.7-plus"  # Test a Qwen model through the configured API endpoint.
    model_config = get_model_config(model_name)

    print(f"Loaded model configuration for {model_name}.")


    test_dir = Path(__file__).parent / "test_mini_swe_cli"
    test_dir.mkdir(exist_ok=True)

    result = call_mini_swe_agent(
        instruction="Create a file called hello.txt with content 'Hello World', then run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        model=model_config["name"],  # Pass the original name; call_mini_swe_agent adds the provider prefix.
        api_key=model_config["api_key"],
        api_base=model_config.get("api_base", ""),
        working_dir=str(test_dir),
        timeout=120,
    )

    print(f"\n{'='*60}")
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Messages: {len(result['messages'])}")
    print(f"Metrics: {result['metrics']}")
    print(f"{'='*60}")
