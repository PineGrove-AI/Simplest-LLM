# The simplest possible LLM

A tiny next-word predictor that fits in one file of plain numbers, wrapped in a
small interactive web app. Built to help people *see* what an LLM actually is:
a table of weights, and one line of math (`logits = x @ W + b`) that turns input
words into probabilities over the next word.

Everything runs in the browser. There's no backend and no Python.

## What's in here

| Path | What it is |
|---|---|
| `web/src/model.json` | **The brain.** A plain JSON file of numbers: a vocabulary, a weight matrix `W`, and a bias `b`. This is the whole model. |
| `web/src/model.ts` | Loads the model and does completion (one word at a time). Also exposes helpers the UI uses to illustrate each step. |
| `web/src/trainer.ts` | Learns `W` and `b` from your sentences, right in the browser, using plain gradient descent. |
| `web/src/App.tsx` | The UI: prompt box, one-word-at-a-time completion, a "How it works" accordion with live examples, and a "Train your own" panel. |
| `web/src/App.css` / `index.css` | Styling. |

## How it works (the five ideas)

1. **Vocabulary** – every word gets an index.
2. **Flatten** – the last 4 words become a long vector with 1s where those words sit.
3. **Matrix multiply** – `logits = x @ W + b`. One operation. That's all the "thinking."
4. **Softmax** – turn logits into probabilities that sum to 1.
5. **Sample** – pick a word by its probability, append it, slide the window, repeat.

That's the entire system. No layers, no attention, no magic. It's bad on purpose
so you can trace exactly what happens at each step.

## Run it

```bash
cd web
pnpm install    # once
pnpm dev        # -> http://localhost:5173
```

For sharing, `pnpm build` produces a static `dist/` you can drop on any static
host (Netlify, GitHub Pages, S3). No server needed: the model is just bundled JSON.

## Try these demo moves

- **Open the left panel** to see the whole model as plain JSON; it updates when you retrain.
- **Complete one word at a time** and watch the sentence grow, with the probability distribution shown for each step.
- **Retrain on your own text** in the "Train your own" panel: new vocab, new weights, new behaviour, all in the browser.

## Why it's still an "LLM"

Real LLMs do the same core thing — predict the next token from previous tokens
using learned weights — just with many more parameters, attention layers, and
far more data. This is that idea stripped to its bones.
