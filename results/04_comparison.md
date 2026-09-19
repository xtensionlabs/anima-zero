# Same question, same six tools, same brain: Fable 5.1 vs Opus 4.8

Question given to both (2026-09-20, unedited transcripts in `04_courtship_*.log`):

> A male fruit fly starts singing courtship song when he detects a female. Find the circuit from
> sensory input to the song command descending neuron, identify the central courtship hub, test
> the pathway by stimulation with the scrambled-wiring control, and state clearly what the model
> cannot tell you. Finish within 20 tool calls.

| | claude-opus-4-8 | claude-fable-5-1 |
|---|---|---|
| Tool calls | 20 | 19 |
| Found hub -> command -> song premotor (pC1 -> pIP10 -> TN1a) | yes | yes |
| Ran the scrambled control | yes, on 1 stimulation | yes, on 2 stimulations |
| Sensory input to the hub | Followed the textbook pheromone route (cVA -> ORN_DA1 -> DA1_lPN -> pC1). Anatomy traced; **physiology failed** (driving DA1_lPN did not move pIP10). Reported honestly as "not functionally engaged". | Read the connectome instead of the textbook: the hub's dominant input (12%) is an ascending leg-contact neuron, AN08B074. **Tested it: +15 Hz on the hub, song premotor neurons +6 to +8 Hz, control abolished it (+0.3 Hz).** Also tested the visual (LC10a) and olfactory (Or47b) routes and showed they are weak. |
| Extra structure found | pIP10 outputs match TN1a song neurons | pC1 recurrent self-excitation (substrate for persistent courtship state); AN08B074 bypass collaterals straight to the VNC song circuit; brain-to-VNC positive feedback via pMP2 |
| Graded its own evidence | limits listed | Yes: flagged pIP10 as the weakest, least control-separated link (+3 Hz real vs +2 Hz scrambled) and noticed baseline-rate differences between runs |
| Model limits flagged | plasticity, state, behaviour, coarse drive | the same plus predicted transmitter signs, type-level lumping, single seed, beam-search misses |

## Honest reading

Both models completed the task with controls. The previous model confirmed the textbook circuit and
correctly reported that the textbook sensory arm does not work in this model. Fable 5.1 treated the
connectome as the primary source, found the sensory input the data actually supports, proved it
physiologically with a control, and ranked the strength of each link in its own chain. That is the
difference between confirming a known answer and discovering one from the wiring.

Both ran concurrently on the same laptop, about 15 minutes each of wall clock (shared CPU).
