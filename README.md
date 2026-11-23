# Does a Visa Change an AI's Salary Advice?

We asked Gemini 3.5 Flash-Lite the same salary question 320 times, changing only one sentence: the user's visa status, English style, or a fairness line. Then we asked the AI whether the visa changed its number.

**Live demo (all figures and examples):** https://claireoh-duke.github.io/visa-salary-audit/

## Why this project

Working in the U.S. on a visa got harder in 2025–26: a new $100K fee for hiring on a new H-1B visa (now on hold in court), a proposed $70K school fee for each student starting OPT, and 15% fewer new international students in fall 2026. These students are negotiating their first U.S. salaries, and some ask AI tools for help.

Prior audits of AI salary advice changed who the user is (gender, ethnicity, university, migration background). We changed only the user's **visa status**, and checked whether the AI's **written advice and self-explanations match its numbers**.

The second part builds on work showing that LLM explanations can be *unfaithful*: they don't always reflect what actually drove the output (Turpin et al., 2023). We test this in a real-world setting: when a model gives visa holders a lower salary, does its explanation say so?

## Questions

1. Does visa status change the suggested salary?
2. Does Korean-L1-style English change it?
3. Does telling the AI to be fair help?
4. Does the AI's explanation match its numbers?

## Findings

- **Visa status lowers the number.** F-1 OPT users got a 2.8% lower salary than users who didn't mention a visa (95% CI −4.7 to −0.9), about $4,050 less.
- **A fairness line didn't fix it.** Even when told not to adjust for immigration status, the AI gave F-1 OPT users 4.2% less.
- **English style changes the advice, not the salary.** For U.S. citizens, 7 of 20 Korean-L1-style answers brought up immigration; 0 of 20 fluent answers did.
- **The AI doesn't admit the gap.** In 0 of 80 visa answers did it say the visa lowered its number. When asked directly, it said "the same" in 32 of 32 chats, while the F-1 OPT number was actually lower in 8 of 8.

Asking the AI does not reveal the gap. Comparing answers that differ by one sentence does. This matches the unfaithful-explanation pattern in Turpin et al. (2023), here in salary advice rather than benchmark tasks.

## Method

The user asks for salary advice on a New York job offer (master's degree, 3 years of experience). We swap:

| Factor | Versions |
|---|---|
| Visa status | No mention · U.S. citizen · F-1 OPT · Needs H-1B |
| English style | Fluent · Korean-L1 style |
| Fairness line | Off · On |
| Job | Software engineer · Data scientist · Registered nurse · Marketing manager |

64 versions × 5 runs = 320 answers (temperature 1.0, seed 42). Salaries were compared with a regression (job controls, HC3 robust SEs); the 80 visa answers were hand-coded. The follow-up asked 32 chats (temperature 0) what drove the number and whether it would differ without the visa mention; the scoring plan is in `data/followup/`.

**Design choices**
- **One-sentence swaps** so any salary difference can only come from that sentence.
- **Temperature 1.0, 5 runs per version** to measure the gap against normal sampling noise.
- **Hand coding** of the 80 visa answers, because keyword search can't tell "don't take less" from "expect less".
- **Follow-up chats at temperature 0**, with the scoring plan written before data collection, so the AI's self-explanation is compared with the measured gap, not judged after the fact.
- **Self-contained demo** (one HTML file, data embedded) so anyone can explore it without a server or API key.

## Repository

```
docs/index.html   interactive demo (self-contained)
src/              prompts, model clients, experiment, analysis, follow-up
data/             raw answers, parsed results, hand codes, follow-up chats
results/          analyze.py output
```

## Reproduce

Requires Python 3.9+.

```bash
git clone https://github.com/claireoh-duke/visa-salary-audit.git
cd visa-salary-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # never commit this

# Optional: test the pipeline offline with a fake model (no API key needed)
python src/run_experiment.py --provider mock --model mock --reps 2 --rpm 100000
python src/analyze.py --include-mock

# Main experiment: 320 prompts (64 versions x 5 runs) -> data/raw/gemini__gemini-3.5-flash-lite.jsonl
# Resumable: if a daily quota stops it, run the same command again later.
python src/run_experiment.py --provider gemini --model gemini-3.5-flash-lite --reps 5 --rpm 8 --seed 42

# Analyze everything in data/raw/ (mock data is skipped)
# -> results/parsed.csv, results/summary.json, results/regression.txt, results/figures/
python src/analyze.py
cp results/parsed.csv results/summary.json data/

# Follow-up chats (plan: data/followup/analysis_plan_followup.md) -> data/followup/
python src/followup.py determinism   # 10 calls: temperature-0 repeat check
python src/followup.py run           # 96 calls: 32 chats x 3 turns
python src/followup.py score         # -> data/followup/summary.json (no API calls)
```

Results will not match exactly: the model is sampled, and hosted models can change. The raw answers behind the reported numbers are in `data/raw/` and `data/followup/`.

To view the demo locally, open `docs/index.html` in a browser. It needs no server and no API key.

## Limitations and risks

- One model, 5 runs per version; 8 follow-up chats per visa status.
- Hand codes come from one coder. The prompt shown in the demo is paraphrased; the exact prompts are in `src/prompts.py`.
- Temperature 0 was not deterministic (same prompt, salaries 2.6–9.7% apart).
- The gap shows what the AI advises, not what visa holders are worth.

**Who is affected:** international students, career centers that recommend AI tools, and AI developers. A student may take the lower number as the market rate and ask for less; because the advice sounds neutral, the gap is easy to miss.

**Deployment risk:** with chat memory, a model may know a user's visa status without being told in the current chat.

## Future work

Test more models and more runs, add a second coder for the hand codes, and run more follow-up chats per version.

## Related work

- Geiger, R. S., O'Sullivan, F., Wang, E., & Lo, J. (2025). Asking an AI for salary negotiation advice is a matter of concern. *PLOS ONE*, 20(2), e0318500.
- Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. *NeurIPS 2023*.
- Sorokovikova, A., Chizhov, P., Eremenko, I., & Yamshchikov, I. P. (2025). Surface Fairness, Deep Bias: A Comparative Study of Bias in Language Models. *GeBNLP 2025*.

Policy sources: Fragomen (H-1B fee, Sep 2026); DHS proposed rule 91 FR 64566 (OPT fee, Oct 2026); IIE Fall 2026 Snapshot (enrollment).
