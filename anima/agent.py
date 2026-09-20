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


# ---- human-readable rendering of tool results (what a judge sees) -------------------------------

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console(highlight=False)


def render_result(name: str, data) -> None:
    if not isinstance(data, dict):
        console.print(f"  [dim]{str(data)[:300]}[/]")
        return
    if "error" in data:
        console.print(f"  [red]{data['error']}[/]  did you mean: {', '.join(data.get('did_you_mean', [])[:6])}")
        return
    if name == "overview":
        console.print(f"  {data['neurons']:,} neurons, {data['connections']:,} connections, {data['cell_types']:,} cell types, "
                      f"{data['descending_neuron_types']} descending-neuron types (motor commands)")
    elif name == "search_types":
        hits = ", ".join(f"{t['type']} ({t['neurons']})" for t in data["types"][:10])
        console.print(f"  {data['matches']} types match '{data['query']}': {hits}")
    elif name == "describe_type":
        down = ", ".join(f"{d['type']} {100*d['input_fraction']:.0f}%{d['net_sign']}" for d in data["strongest_downstream"][:5])
        up = ", ".join(f"{d['type']} {100*d['input_fraction']:.0f}%{d['net_sign']}" for d in data["strongest_upstream"][:5])
        sides = "/".join(f"{k}:{v}" for k, v in data["sides"].items())
        console.print(f"  [bold]{data['type']}[/]: {data['neurons']} neurons ({sides}), {data['sign']}, {'/'.join(data['superclass'])}")
        console.print(f"    -> strongest targets (share of their input): {down}")
        console.print(f"    <- strongest inputs (share of its input):    {up}")
    elif name == "trace_paths":
        d = data["direct_input_fraction"]
        console.print(f"  [bold]{data['from']} -> {data['to']}[/]: direct synapses supply {100*d:.1f}% of {data['to']}'s input ({data['direct_sign']})"
                      if d > 0 else f"  [bold]{data['from']} -> {data['to']}[/]: no direct synapses")
        for p in data["paths"][:4]:
            console.print(f"    {p['route']}   strength {p['score']:.4f}")
        if not data["paths"]:
            console.print("    no route found within the search budget")
    elif name == "stimulate":
        s = data["stimulated"]
        scr = data["wiring"].startswith("scrambled")
        colour = "red" if scr else "green"
        title = (f"[{colour}]{data['wiring'].upper()}[/]   drive {', '.join(s['types'])} side={s['side'] or 'both'} "
                 f"({s['neurons']} neurons, {s['duration_s']} s)")
        t = Table(title=title, title_justify="left", show_edge=False, pad_edge=False, header_style="bold")
        for col in ("neuron", "side", "baseline Hz", "stimulated Hz", "change"):
            t.add_column(col, justify="right" if "Hz" in col or col == "change" else "left")

        def row(name, side, v):
            dlt = v["delta_hz"]
            style = "bold green" if dlt >= 5 else ("bold red" if dlt <= -5 else "dim")
            t.add_row(name, side, f"{v['baseline_hz']:.0f}", f"{v['stimulated_hz']:.0f}", f"[{style}]{dlt:+.0f}[/]")

        for name_, sides in data["watched"].items():
            for side, v in sides.items():
                row(name_, side, v)
        if data["watched"] and data["most_changed_descending_neurons"]:
            t.add_section()
        for c in data["most_changed_descending_neurons"][:6]:
            if c["type"] not in data["watched"]:
                row(f"{c['type']}  (most changed)", c["side"], c)
        console.print(t)
        wb = data["whole_brain"]
        console.print(f"  whole brain: {wb['neurons_with_delta_over_5hz']:,} of 166,700 neurons changed by >5 Hz; "
                      f"mean rate {wb['baseline_mean_hz']:.2f} -> {wb['stimulated_mean_hz']:.2f} Hz")
    elif name == "set_scrambled":
        scr = data["wiring"].startswith("scrambled")
        console.print(Panel(f"[bold]{'CONTROL: wiring scrambled. Same neurons, same degrees, same signs. Who-connects-to-whom is random.' if scr else 'Real connectome restored.'}[/]",
                            style="red" if scr else "green", expand=False))
    else:
        console.print(f"  [dim]{json.dumps(data)[:400]}[/]")


def run(question: str, model: str = MODEL, effort: str = "high", lab: BrainLab | None = None) -> str:
    console.print(Panel(f"[bold]{question}[/]\n[dim]model {model} · effort {effort} · brain: MaleCNS v1.0, 166,700 neurons, live[/]",
                        title="anima-zero", expand=False))
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
    calls = 0
    for message in runner:
        pending = []
        for block in message.content:
            if block.type == "thinking" and block.thinking:
                console.print(f"[dim italic]{block.thinking[:400].strip()}[/]")
            elif block.type == "tool_use":
                calls += 1
                args = ", ".join(f"{k}={json.dumps(v)}" for k, v in block.input.items())
                console.print(f"\n[bold cyan]{calls:>2}. {block.name}[/]([dim]{args}[/])")
                pending.append(block.name)
            elif block.type == "text" and block.text.strip():
                final_text = block.text
                console.print()
                console.print(Markdown(block.text))
        if message.stop_reason == "refusal":
            console.print("[red]model refused; see stop_details[/]")
            break
        tool_response = runner.generate_tool_call_response()
        if tool_response is not None:
            for name, r in zip(pending, tool_response["content"]):
                raw = r["content"] if isinstance(r["content"], str) else json.dumps(r["content"])
                try:
                    render_result(name, json.loads(raw))
                except json.JSONDecodeError:
                    console.print(f"  [dim]{raw[:300]}[/]")
    console.print(f"\n[dim]{calls} tool calls[/]")
    return final_text


def main() -> None:
    p = argparse.ArgumentParser(description="Ask Claude to investigate the fly connectome by experiment.")
    p.add_argument("question", nargs="?", default="How does the fly escape a looming object on its left? Find the circuit, "
                                                  "prove it by stimulation with the scrambled control, and tell me what the "
                                                  "model cannot show. Be efficient: at most 8 tool calls.")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    a = p.parse_args()
    run(a.question, a.model, a.effort)


if __name__ == "__main__":
    main()
