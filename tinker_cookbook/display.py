import io

import tinker
from termcolor import colored

from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils.format_colorized import format_colorized


def _truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text showing start and end with escaped newlines."""
    text_escaped = text.replace('\n', '\\n')
    if len(text) > max_len:
        start = text[:100].replace('\n', '\\n')
        end = text[-50:].replace('\n', '\\n')
        return f"{start}...{end}"
    return text_escaped


def to_ints(chunk: tinker.ModelInputChunk, tokenizer: Tokenizer):
    if isinstance(chunk, tinker.EncodedTextChunk):
        return chunk.tokens
    else:
        (at_token,) = tokenizer.encode("@", add_special_tokens=False)
        return [at_token] * chunk.length


def colorize_example(datum: tinker.Datum, tokenizer: Tokenizer, key: str = "weights"):
    int_tokens = [
        token for chunk in datum.model_input.chunks for token in to_ints(chunk, tokenizer)
    ] + [datum.loss_fn_inputs["target_tokens"].tolist()[-1]]
    weights = [0.0] + datum.loss_fn_inputs[key].tolist()
    return format_colorized(int_tokens, weights, tokenizer)

def format_text(text: str, preview: bool = False) -> str:
    return _truncate_text(text) if preview else text

def format_trajectory(
    trajectory: Trajectory,
    tokenizer: Tokenizer,
    preview: bool = False,
    label: str = "",
) -> str:
    """Format a trajectory for display.
    
    Args:
        trajectory: The trajectory to format
        tokenizer: Tokenizer for decoding tokens
        preview: If True, truncate observation and action text.
                 If False, show full text.
        label: Optional label prefix (e.g., "Group 0, Traj 1")
        
    Returns:
        Formatted string representation of the trajectory
    """
    buf = io.StringIO()

    def colorize(s: str):
        return colored(s, "green", attrs=["bold"])

    def bprint(s: str):
        print(s, file=buf)
    

    header = f"{'=' * 60} {label}" if label else "=" * 60
    bprint(header)
    for i, transition in enumerate(trajectory.transitions):
        obs_text = tokenizer.decode(transition.ob.to_ints())
        action_text = tokenizer.decode(transition.ac.tokens, skip_special_tokens=True)
        
        bprint(f"------ Transition {i} ------")
        bprint(f"{colorize('Observation')}: {format_text(obs_text, preview)}")
        bprint(f"{colorize('Action')}: {format_text(action_text, preview)}")
        bprint(f"{colorize('Reward')}: {transition.reward}")
        bprint(f"{colorize('Episode done')}: {transition.episode_done}")
        bprint(f"{colorize('Metrics')}: {transition.metrics}")
        bprint("-" * 60)
    bprint("=" * 60)
    
    return buf.getvalue()
