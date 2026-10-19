// In-browser trainer. Ported 1:1 from train.py.
//
// Reads sentences, learns a single matrix W and bias b that predict the next
// word from the previous `contextSize` words, using plain gradient descent.
// No framework, no NumPy: just arrays and loops, so you can read every line.

export interface TrainedModel {
  vocab: string[];
  context_size: number;
  // Row-major weight table of length (context_size * vocab.length).
  W: number[][];
  b: number[];
}

export interface TrainOptions {
  contextSize?: number;   // how many previous words to look at (default 4)
  epochs?: number;        // how many times to walk over the data (default 200)
  learningRate?: number;  // size of each correction step (default 0.5)
  maxExamples?: number;   // cap on training examples, to keep big inputs fast (default 4000)
  onProgress?: (epoch: number, avgLoss: number) => void;
}

/** Yield control back to the browser so it can repaint between epochs. */
function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** Turn raw scores into probabilities that sum to 1. */
function softmax(x: number[]): number[] {
  const max = Math.max(...x);
  const exps = x.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Standard-normal random via Box-Muller (replaces np.random.randn). */
function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Train a model from raw text. Returns the full model (vocab, W, b) ready to
 * be used for completion. This is the "file of math" being created live.
 *
 * Async on purpose: it yields to the browser between epochs so the UI stays
 * responsive (progress bar animates, page doesn't freeze) even on large input.
 */
export async function train(text: string, opts: TrainOptions = {}): Promise<TrainedModel> {
  const CONTEXT_SIZE = opts.contextSize ?? 4;
  const EPOCHS = opts.epochs ?? 200;
  const LEARNING_RATE = opts.learningRate ?? 0.5;
  const MAX_EXAMPLES = opts.maxExamples ?? 4000;

  // --- Build vocabulary: "pad" first, then every new word in order of appearance.
  const vocab: string[] = ["pad"];
  const seen = new Set<string>(["pad"]);
  for (const line of text.split("\n")) {
    for (const w of line.trim().toLowerCase().split(/\s+/).filter(Boolean)) {
      if (!seen.has(w)) { seen.add(w); vocab.push(w); }
    }
  }
  const V = vocab.length;
  const index = new Map(vocab.map((w, i) => [w, i]));

  // --- Build examples: (previous CONTEXT_SIZE words) -> (next word).
  const X: number[][] = [];
  const y: number[] = [];
  for (const line of text.split("\n")) {
    const toks = line.trim().toLowerCase().split(/\s+/).filter(Boolean)
      .map((w) => index.get(w)!);
    for (let t = 0; t < toks.length; t++) {
      let ctx = toks.slice(Math.max(0, t - CONTEXT_SIZE), t);
      while (ctx.length < CONTEXT_SIZE) ctx = [0, ...ctx]; // pad the front
      X.push(ctx);
      y.push(toks[t]);
    }
  }
  // Cap the work so a big paste stays fast. If there are more examples than
  // the cap, use a random sample of them (still representative of the text).
  if (X.length > MAX_EXAMPLES) {
    const sampled: number[] = [];
    while (sampled.length < MAX_EXAMPLES) {
      sampled.push(Math.floor(Math.random() * X.length));
    }
    for (let i = X.length - 1; i > 0; i--) {
      const j = sampled[i];
      [X[i], X[j]] = [X[j], X[i]];
      [y[i], y[j]] = [y[j], y[i]];
    }
    X.splice(MAX_EXAMPLES);
    y.splice(MAX_EXAMPLES);
  }

  // --- The model: W is a table, one row per (position, word) pair.
  const D = CONTEXT_SIZE * V;
  const W: number[][] = Array.from({ length: D }, () =>
    new Array(V).fill(0).map(() => randn() * 0.1),
  );
  const b = new Array(V).fill(0);

  // --- Gradient descent.
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    let totalLoss = 0;
    for (let e = 0; e < X.length; e++) {
      const xi = X[e], yi = y[e];

      // Flatten position+word into a one-hot input vector.
      const x = new Array(D).fill(0);
      xi.forEach((tok, pos) => { x[pos * V + tok] = 1; });

      // THE forward pass: logits = x @ W + b
      const logits = new Array(V).fill(0);
      for (let o = 0; o < V; o++) {
        let acc = b[o];
        for (let i = 0; i < D; i++) acc += W[i][o] * x[i];
        logits[o] = acc;
      }
      const probs = softmax(logits);

      // How wrong we were (cross-entropy).
      totalLoss += -Math.log(probs[yi] + 1e-9);

      // Gradient of cross-entropy wrt logits.
      const grad = probs.slice();
      grad[yi] -= 1;

      // Nudge the weights: W -= lr * outer(x, grad);  b -= lr * grad.
      for (let i = 0; i < D; i++) {
        if (x[i] === 0) continue;
        for (let o = 0; o < V; o++) W[i][o] -= LEARNING_RATE * x[i] * grad[o];
      }
      for (let o = 0; o < V; o++) b[o] -= LEARNING_RATE * grad[o];
    }
    opts.onProgress?.(epoch, totalLoss / X.length);
    await yieldToBrowser();
  }

  return { vocab, context_size: CONTEXT_SIZE, W, b };
}
