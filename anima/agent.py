"""The fusion, phase 1: Claude as the scientist / cortex, the fly connectome as the organ.

    python -m anima.agent "How does the fly turn a looming object into an escape? Prove it by experiment."

Claude gets six tools over a live whole-CNS simulation (BrainLab) and is asked to reason
about mechanism the way a systems neuroscientist would: look up types, trace anatomy,
then *test* it by stimulation with a scrambled-wiring control. Needs ANTHROPIC_API_KEY
(or an `ant auth login` profile). Every tool call and result is echoed so the run is auditable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")                      # ANTHROPIC_API_KEY lives here, never in code
sys.path.insert(0, str(ROOT))
from anima.brainlab import BrainLab  # noqa: E402

MODEL = "claude-opus-5"

SYSTEM = """You are the cortex of a bio-digital system: a language model wired to a live simulation of the complete central nervous system of a male fruit fly (MaleCNS v1.0 connectome: 166,700 neurons, 25 million connections, leaky integrate-and-fire dynamics, weights from electron microscopy). Nothing in the fly brain is trained; you can only stimulate its neurons and read what they do.

Work like a systems neuroscientist:
1. Anatomy first: search_types and describe_type to find candidate populations; trace_paths to find routes between them.
2. Then physiology: stimulate a population and read which descending neurons (the brain's motor commands to the body) change. Watch the specific types your anatomy predicted.
3. Always run the scrambled-wiring control (set_scrambled True, repeat, set_scrambled False) before claiming a circuit is real. Degree-preserving scrambling keeps every statistic but destroys who-connects-to-whom.
4. Report mechanism with numbers: which types, which side, input fractions, firing-rate changes vs baseline, and what the control showed. Say what the model cannot tell you (no plasticity, hand-calibrated drive, sensory relays bypassed).

Cell-type names follow the Drosophila literature (e.g. LC4, LPLC2 looming detectors; DNp01 giant fiber; DNa02 steering; MDN backward walking; KC Kenyon cells; MBON mushroom body output neurons). Sides are "L" and "R". A stimulation of 50 steps is one second of biological time and takes about one second of wall-clock, so experiments are cheap; run several rather than guess."""


def build_tools(lab: BrainLab):
    def dump(x) -> str:
        return json.dumps(x, indent=1)

    @beta_tool
    def overview() -> str:
        """Dataset, model, neuron counts by superclass, and how many descending-neuron types exist."""
        return dump(lab.overview())

    @beta_tool
    def search_types(query: str, limit: int = 25) -> str:
        """Find cell types whose name contains `query` (case-insensitive), with neuron counts and superclass.

        Args:
            query: Name fragment, e.g. "LC4", "DNp", "MBON", "KC".
            limit: Maximum number of types to return.
        """
        return dump(lab.search_types(query, limit))

    @beta_tool
    def describe_type(cell_type: str, top: int = 10) -> str:
        """Neuron count by side, superclass, excitatory/inhibitory sign, and the strongest upstream and downstream cell types by input fraction.

        Args:
            cell_type: Exact cell type name, e.g. "DNp01".
            top: How many partners to list in each direction.
        """
        return dump(lab.describe_type(cell_type, top))

    @beta_tool
    def trace_paths(source_type: str, target_type: str, max_hops: int = 3, top: int = 5) -> str:
        """Strongest anatomical routes from one cell type to another through the real connectome (beam search over type-level connectivity).

        Args:
            source_type: Presynaptic cell type, e.g. "LC4".
            target_type: Postsynaptic cell type, e.g. "DNp01".
            max_hops: Longest route to consider (1 = direct synapses only).
            top: Number of routes to return.
        """
        return dump(lab.trace_paths(source_type, target_type, max_hops, top))

    @beta_tool
    def stimulate(types: list[str], side: str | None = None, amount: float = 0.8, steps: int = 50,
                  watch: list[str] | None = None) -> str:
        """Drive the named cell types every step and compare firing rates to a baseline run with the same random seed. Reports watched types by side and the descending-neuron types whose firing changed most.

        Args:
            types: Cell types (or a superclass such as "visual_projection") to stimulate together.
            side: "L" or "R" to stimulate one side only; omit for both.
            amount: Voltage added per 20 ms step, 0 to 1 (threshold is 1.0; 0.8 is a strong drive).
            steps: Duration in 20 ms steps (50 = 1 s).
            watch: Cell types whose rates to report explicitly, e.g. ["DNp01", "DNa02"].
        """
        return dump(lab.stimulate(types, side, amount, steps, watch))

    @beta_tool
    def set_scrambled(enabled: bool) -> str:
        """Switch stimulate() to a degree-preserving scrambled wiring (control), or back to the real connectome.

        Args:
            enabled: True for scrambled control wiring, False for the real connectome.
        """
        return dump(lab.set_scrambled(enabled))

    return [overview, search_types, describe_type, trace_paths, stimulate, set_scrambled]


def run(question: str, model: str = MODEL, effort: str = "high", lab: BrainLab | None = None) -> str:
    lab = lab or BrainLab()
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": effort},
        tools=build_tools(lab),
        messages=[{"role": "user", "content": question}],
    )
    final_text = ""
    for message in runner:
        for block in message.content:
            if block.type == "thinking" and block.thinking:
                print(f"\n[thinking] {block.thinking[:600]}", file=sys.stderr)
            elif block.type == "tool_use":
                print(f"\n>> {block.name}({json.dumps(block.input)})", file=sys.stderr)
            elif block.type == "text":
                final_text = block.text
                print(f"\n{block.text}")
        if message.stop_reason == "refusal":
            print("model refused; see stop_details", file=sys.stderr)
            break
        tool_response = runner.generate_tool_call_response()
        if tool_response is not None:
            for r in tool_response["content"]:
                content = r["content"] if isinstance(r["content"], str) else json.dumps(r["content"])
                print(f"<< {content[:1500]}{'...' if len(content) > 1500 else ''}", file=sys.stderr)
    return final_text


def main() -> None:
    p = argparse.ArgumentParser(description="Ask Claude to investigate the fly connectome by experiment.")
    p.add_argument("question", nargs="?", default="How does the fly turn a looming object on its left into an escape? "
                                                  "Find the circuit, test it by stimulation, and run the scrambled control.")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    a = p.parse_args()
    run(a.question, a.model, a.effort)


if __name__ == "__main__":
    main()
