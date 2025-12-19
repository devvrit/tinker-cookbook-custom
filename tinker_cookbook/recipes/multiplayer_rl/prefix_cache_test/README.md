# Prefix Cache Test Recipe

A simple environment designed to test whether prefix caching is working in your inference engine.

## Quick Start

```bash
python -m tinker_cookbook.recipes.multiplayer_rl.prefix_cache_test.train
```

## What is Prefix Caching?

Prefix caching is an optimization technique where the key-value (KV) cache for common prefixes is reused across different requests. When multiple prompts share the same system prompt or prefix, the inference engine computes the KV cache once and reuses it for subsequent requests.

**Benefits:**
- Reduced computation time for requests with shared prefixes
- Lower time-to-first-token (TTFT)
- Improved throughput for batched requests

## How This Recipe Tests Prefix Caching

This recipe is specifically designed to make prefix caching benefits observable:

1. **Long Shared System Prompt**: All environments share a ~2KB system prompt, maximizing the benefit of caching the prefix.

2. **Simple Task**: The task is to simply echo a word back (e.g., "Echo: apple"). This minimizes variance in generation time so caching effects are more visible.

3. **High Parallelism**: Default configuration uses:
   - `batch_size=16`: Multiple different prompts per batch
   - `group_size=8`: Multiple environments per prompt (all sharing exact same prefix)

4. **Timing Logs**: Debug-level logging tracks per-step timing to help identify caching effects.

## Verifying Prefix Caching is Working

### Method 1: Check Inference Server Logs

If using vLLM, look for these metrics in the server logs:
- `prefix_cache_hit_rate` - Should be > 0 if caching is working
- `gpu_cache_usage_perc` - Shows cache utilization

### Method 2: Monitor Time-to-First-Token (TTFT)

Compare TTFT for:
- First request (cache miss) - Higher latency
- Subsequent requests with same prefix (cache hit) - Lower latency

### Method 3: Throughput Comparison

Run the same workload with and without prefix caching enabled:

```bash
# With prefix caching (vLLM default for most recent versions)
python -m tinker_cookbook.recipes.multiplayer_rl.prefix_cache_test.train

# Without prefix caching (if your server supports disabling it)
# Set enable_prefix_caching=False in vLLM config
```

## Configuration Options

This recipe uses `chz` for CLI args, so use `key=value` syntax (not `--key value`):

```bash
# With wandb logging
python -m tinker_cookbook.recipes.multiplayer_rl.prefix_cache_test.train \
    wandb_project=prefix-caching

# With multiple parameters
python -m tinker_cookbook.recipes.multiplayer_rl.prefix_cache_test.train \
    batch_size=32 \
    group_size=16 \
    max_tokens=32 \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    wandb_project=prefix-caching \
    wandb_name=my-experiment
```

## Expected Behavior

If prefix caching is working correctly:
1. The system prompt KV cache is computed once per unique prompt
2. All environments in a group (same `word_to_echo`) reuse the cached prefix
3. You should see improved throughput compared to no caching

## The Task

Simple echo task:
```
[System]: <~2KB system prompt about prefix caching test>
[User]: Please echo the following word: apple
[Assistant]: Echo: apple
[User]: Correct! Echo matched.
```

Reward: 1.0 for correct echo, 0.0 for incorrect, -1.0 for format error.

## Debugging

Enable debug logging to see timing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show per-environment, per-step timing that can help identify if caching is working.

