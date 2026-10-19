// The entire "brain" of our tiny LLM. By default it's loaded from a plain
// JSON file (the bundled model), but you can swap in any other model at
// runtime via setModel() -- e.g. one just trained in the browser.
//
// Deliberately simple: no framework, no magic. Just numbers and one line of
// math (logits = x @ W + b) to predict the next word.

import rawModel from "./model.json";
import type { TrainedModel } from "./trainer";

export interface Model {
  vocab: string[];
  context_size: number;
  // Row-major weight table of length (context_size * vocab.length).
  // Row r corresponds to (position p, word w) where r = p * V + w.
  W: number[][];
  b: number[];
}

// The active model lives in a mutable slot so we can hot-swap it after training.
let model: Model = rawModel as unknown as Model;

/** Replace the active model (e.g. with a freshly trained one). */
export function setModel(m: Model): void {
  model = m;
}

/** The currently active model. */
export function getModel(): Model {
  return model;
}

/** Turn raw scores into probabilities that sum to 1. */
export function softmax(x: number[]): number[] {
  const max = Math.max(...x);
  const exps = x.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Build the input vector for the given tokens under the active model. */
function flatten(tokens: string[]): number[] {
  const V = model.vocab.length;
  const index = new Map(model.vocab.map((w, i) => [w, i]));
  const x = new Array(model.context_size * V).fill(0);
  tokens.forEach((tok, pos) => {
    const i = index.get(tok);
    if (i !== undefined) x[pos * V + i] = 1;
  });
  return x;
}

/**
 * Public version of flatten: turn tokens into the long one-hot input vector
 * the model actually consumes. Exposed so the UI can show what "flatten"
 * really produces for the current prompt.
 */
export function toInputVector(tokens: string[]): number[] {
  return flatten(tokens);
}

/**
 * Compute the next-word distribution for a prompt: raw logits and the
 * softmax probabilities, both aligned with the vocabulary order. Used by the
 * UI to illustrate the matrix-multiply, softmax and sample steps live.
 */
export function nextWordDistribution(prompt: string[]): { logits: number[]; probs: number[] } {
  let tokens = prompt.slice(-model.context_size);
  while (tokens.length < model.context_size) tokens = ["pad", ...tokens];
  const x = flatten(tokens);
  const lg = logits(x);
  return { logits: lg, probs: softmax(lg) };
}

/** THE forward pass. One operation. This is all the model does. */
function logits(x: number[]): number[] {
  const V = model.vocab.length;
  const out = new Array(V).fill(0);
  for (let o = 0; o < V; o++) {
    let acc = model.b[o];
    for (let i = 0; i < x.length; i++) acc += model.W[i][o] * x[i];
    out[o] = acc;
  }
  return out;
}

export interface Step {
  token: string;          // the word chosen this step
  probs: { word: string; p: number }[]; // full distribution over the next word
  top: { word: string; p: number }[];   // top few, sorted
}

/**
 * Generate exactly ONE next word from a context (the words so far), using the
 * active model. Returns the chosen word and its probability distribution, so
 * the UI can show what the model "thought". Used for one-word-at-a-time
 * completion: call it repeatedly, feeding back the growing context each time.
 */
export function nextWord(context: string[]): Step {
  const vocab = model.vocab;
  const V = vocab.length;
  const CONTEXT_SIZE = model.context_size;

  let tokens = context.slice(-CONTEXT_SIZE);
  while (tokens.length < CONTEXT_SIZE) tokens = ["pad", ...tokens];

  const x = flatten(tokens);
  const l = logits(x);
  const probs = softmax(l);

  const dist = vocab
    .map((word, i) => ({ word, p: probs[i] }))
    .sort((a, b) => b.p - a.p);

  // Sample a word by its probability.
  let r = Math.random();
  let chosen = 0;
  for (let i = 0; i < V; i++) {
    r -= probs[i];
    if (r <= 0) { chosen = i; break; }
    chosen = i;
  }

  return { token: vocab[chosen], probs: dist, top: dist.slice(0, 5) };
}

/**
 * Generate a full completion of `n` words from a prompt using the active
 * model. Returns just the generated words (no per-step detail), for use in
 * batch comparisons like the Test bench.
 *
 * When `greedy` is true it always picks the most probable word, so completions
 * are deterministic: the only thing that changes between runs is the weights,
 * which makes before/after comparisons clean.
 */
export function completePrompt(prompt: string[], n: number, greedy = false): string[] {
  let context = [...prompt];
  const out: string[] = [];
  const vocab = model.vocab;
  const V = vocab.length;
  for (let i = 0; i < n; i++) {
    let tokens = context.slice(-model.context_size);
    while (tokens.length < model.context_size) tokens = ["pad", ...tokens];
    const x = flatten(tokens);
    const probs = softmax(logits(x));
    let chosen: number;
    if (greedy) {
      chosen = 0;
      for (let k = 1; k < V; k++) if (probs[k] > probs[chosen]) chosen = k;
    } else {
      let r = Math.random();
      chosen = 0;
      for (let k = 0; k < V; k++) {
        r -= probs[k];
        if (r <= 0) { chosen = k; break; }
        chosen = k;
      }
    }
    const token = vocab[chosen];
    out.push(token);
    context = [...context, token];
  }
  return out;
}

/**
 * Generate `n` words from a prompt using the active model, recording the
 * probability distribution at every step so the UI can illustrate what the
 * model "thought".
 */
export function complete(prompt: string[], n: number): { text: string[]; steps: Step[] } {
  const vocab = model.vocab;
  const V = vocab.length;
  const CONTEXT_SIZE = model.context_size;

  let tokens = prompt.slice(-CONTEXT_SIZE);
  while (tokens.length < CONTEXT_SIZE) tokens = ["pad", ...tokens];

  const text: string[] = [];
  const steps: Step[] = [];

  for (let s = 0; s < n; s++) {
    const x = flatten(tokens);
    const l = logits(x);
    const probs = softmax(l);

    // Full distribution, sorted descending, for illustration.
    const dist = vocab
      .map((word, i) => ({ word, p: probs[i] }))
      .sort((a, b) => b.p - a.p);

    // Sample a word by its probability.
    let r = Math.random();
    let chosen = 0;
    for (let i = 0; i < V; i++) {
      r -= probs[i];
      if (r <= 0) { chosen = i; break; }
      chosen = i;
    }
    const token = vocab[chosen];
    text.push(token);
    steps.push({ token, probs: dist, top: dist.slice(0, 5) });
    tokens = [...tokens.slice(1), token];
  }

  return { text, steps };
}

/**
 * Render the active model as pretty-printed JSON -- the "file of math".
 * The weight table is shown in full; the UI collapses it for readability.
 */
export function modelAsJson(): string {
  return JSON.stringify(getModel(), null, 2);
}

// Re-export the trainer's type so consumers can import it from one place.
export type { TrainedModel };
