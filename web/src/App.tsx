import { Fragment, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  setModel,
  getModel,
  toInputVector,
  nextWordDistribution,
  nextWord,
  completePrompt,
} from "./model";
import { train } from "./trainer";
import JsonViewer from "@andypf/json-viewer/dist/esm/react/JsonViewer";
import "./App.css";

const DEFAULT_TEXT = `the cat sat on the mat
the dog sat on the mat
the cat sat on a mat
the dog sat on a mat
the cat is on the mat
the dog is on the mat
a cat sat on the mat
a dog sat on the mat
the cat sat in the house
the dog sat in the house
the cat is in the house
the dog is in the house
a cat sat in the house
a dog sat in the house`;

// A single probability bar for one candidate word.
function ProbBar({ word, p }: { word: string; p: number }) {
  const pct = Math.max(0, Math.min(100, p * 100));
  return (
    <div className="prob-row">
      <span className="prob-word">{word}</span>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="prob-num">{(p * 100).toFixed(1)}%</span>
    </div>
  );
}

// One expandable step in the "How it works" accordion.
function Step({
  n,
  title,
  open,
  onToggle,
  children,
}: {
  n: number;
  title: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <div className={`step ${open ? "open" : ""}`}>
      <button className="step-head" onClick={onToggle}>
        <span className="step-num">{n}</span>
        <span className="step-title">{title}</span>
        <span className="step-caret">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="step-body">{children}</div>}
    </div>
  );
}

// Renders the live model as a nice, collapsible JSON tree.
// Starts with the top-level keys visible but the big arrays (W, b, vocab)
// collapsed, so the file is readable without dumping hundreds of numbers.
function ModelJson({ model }: { model: unknown }) {
  return (
    <div className="json-viewer-wrap">
      <JsonViewer
        data={model}
        expanded={1}
        theme="default-dark"
        showToolbar={false}
        showDataTypes={false}
        showSize={true}
        showCopy={true}
        expandIconType="arrow"
        indent={2}
      />
    </div>
  );
}

export default function App() {
  const [promptText, setPromptText] = useState("the cat sat on");
  // Words generated so far (one added per click of Complete).
  const [generated, setGenerated] = useState<string[]>([]);
  // The probability distribution from the most recent step.
  const [lastStep, setLastStep] = useState<import("./model").Step | null>(null);

  // Training state.
  const [trainText, setTrainText] = useState(DEFAULT_TEXT);
  const [epochs, setEpochs] = useState(200);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState<{
    epoch: number;
    loss: number;
  } | null>(null);
  const [trained, setTrained] = useState(false);
  const [showEpochInfo, setShowEpochInfo] = useState(false);
  const [showLossInfo, setShowLossInfo] = useState(false);

  // Left panel: show the live model.json.
  const [showModel, setShowModel] = useState(true);

  // Active tab: explore | train | test.
  const [tab, setTab] = useState<"explore" | "train" | "test">("explore");

  // Test bench state.
  const [testPrompts, setTestPrompts] = useState(
    "the cat\nthe dog sat\na cat is on\nthe dog",
  );
  const [testLength, setTestLength] = useState(5);
  // The "before" snapshot: completions captured the first time we run, so they
  // stay fixed while the user retrains and compares against the new output.
  const [beforeResults, setBeforeResults] = useState<string[][] | null>(null);

  // Which "How it works" step is expanded (null = all closed).
  const [open, setOpen] = useState<number | null>(null);

  const prompt = useMemo(
    () => promptText.toLowerCase().split(/\s+/).filter(Boolean),
    [promptText],
  );

  const model = getModel();

  // The flattened input vector for the current prompt (for the flatten demo).
  const inputVec = useMemo(() => {
    let toks = prompt.slice(-model.context_size);
    while (toks.length < model.context_size) toks = ["pad", ...toks];
    return toInputVector(toks);
  }, [prompt, model]);

  // Live logits + probabilities for the current prompt (for steps 3-5).
  const dist = useMemo(() => nextWordDistribution(prompt), [prompt, model]);
  const index = useMemo(
    () => Object.fromEntries(model.vocab.map((w, i) => [w, i])),
    [model],
  );
  const logitPreview = useMemo(
    () =>
      model.vocab
        .map((word, i) => ({ word, val: dist.logits[i] }))
        .sort((a, b) => b.val - a.val)
        .slice(0, 5),
    [dist, model],
  );
  const probPreview = useMemo(
    () =>
      model.vocab
        .map((word, i) => ({ word, p: dist.probs[i] }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 5),
    [dist, model],
  );
  // A single deterministic "sample" for the demo: the most likely word.
  const top = probPreview[0];
  // The Sample step reflects the live completion when there is one (the last
  // word actually picked and its real distribution); otherwise it falls back
  // to the first word after the raw prompt.
  const hasCompletion = generated.length > 0 && lastStep !== null;
  const sampledWord = hasCompletion ? lastStep!.token : (top?.word ?? "");
  const sampledP = hasCompletion
    ? (lastStep!.probs.find((p) => p.word === lastStep!.token)?.p ?? 0)
    : (top?.p ?? 0);
  const sampleDist = hasCompletion ? lastStep!.probs : probPreview;

  // Add exactly one word to the running completion. The context is the prompt
  // plus everything generated so far, so each click continues from where the
  // last one left off.
  const run = () => {
    const context = [...prompt, ...generated];
    const step = nextWord(context);
    setGenerated((g) => [...g, step.token]);
    setLastStep(step);
  };

  // Reset the running completion (e.g. when the prompt or model changes).
  const reset = () => {
    setGenerated([]);
    setLastStep(null);
  };

  // Test bench: generate completions for every test prompt using the current
  // model. On the first run we also store these as the "before" snapshot so
  // later runs (after retraining) can be compared against them.
  const parseTestPrompts = () =>
    testPrompts
      .split("\n")
      .map((p) => p.trim())
      .filter(Boolean);

  // The "after" completions, computed on demand when Generate is clicked.
  const [afterResults, setAfterResults] = useState<string[][] | null>(null);

  const runTest = () => {
    const results = parseTestPrompts().map((p) =>
      completePrompt(p.split(/\s+/), testLength, true),
    );
    setAfterResults(results);
    if (!beforeResults) setBeforeResults(results); // freeze baseline on first run
  };

  const doTrain = async () => {
    setTraining(true);
    setTrained(false);
    // train() is async and yields between epochs, so the UI stays responsive
    // (progress bar animates) even on large input.
    const m = await train(trainText, {
      epochs,
      onProgress: (epoch, loss) => setProgress({ epoch, loss }),
    });
    setModel(m);
    setTrained(true);
    setTraining(false);
    reset();
  };

  return (
    <div className="layout">
      {/* --- Collapsible left panel: the file of math --- */}
      <aside className={`side ${showModel ? "open" : "closed"}`}>
        <button className="side-toggle" onClick={() => setShowModel((s) => !s)}>
          {showModel ? "◀" : "▶"}{" "}
          <span className="toggle-label">model.json</span>
        </button>
        {showModel && (
          <div className="side-body">
            <p className="side-note">
              The model as plain JSON. Updates when you retrain.
            </p>
            <div className="legend">
              <div className="legend-row">
                <code>vocab</code>
                <span>
                  the {model.vocab.length} words the model knows, in index
                  order.
                </span>
              </div>
              <div className="legend-row">
                <code>context_size</code>
                <span>
                  how many previous words ({model.context_size}) feed each
                  prediction.
                </span>
              </div>
              <div className="legend-row">
                <code>W</code>
                <span>
                  the weights: {model.W.length * model.vocab.length} numbers
                  arranged as {model.context_size}×{model.vocab.length} rows by{" "}
                  {model.vocab.length} columns. Each row is one &ldquo;word at a
                  position&rdquo;; its {model.vocab.length} values are how much
                  that input pushes the score for each possible next word.
                </span>
              </div>
              <div className="legend-row">
                <code>b</code>
                <span>
                  a starting bias: one number per word, added to its score
                  before softmax. It's a context-free baseline, so some words
                  begin ahead of others even when the input gives no strong
                  signal.
                </span>
              </div>
            </div>
            <p className="side-note" style={{ marginTop: 10 }}>
              That's it: every &ldquo;belief&rdquo; the model has lives in these
              numbers. Change them and the output changes. Training just adjusts
              them.
            </p>
            <ModelJson model={model} />
          </div>
        )}
      </aside>

      <div className="app">
        <header className="hero">
          <h1>The simplest possible LLM</h1>
          <p className="sub">
            A vocabulary, learned weights and one line of math producing
            completions.
          </p>
        </header>

        {/* --- Tabs --- */}
        <nav className="tabs">
          <button
            className={`tab ${tab === "explore" ? "active" : ""}`}
            onClick={() => setTab("explore")}
          >
            Explore
          </button>
          <button
            className={`tab ${tab === "train" ? "active" : ""}`}
            onClick={() => setTab("train")}
          >
            Train
          </button>
          <button
            className={`tab ${tab === "test" ? "active" : ""}`}
            onClick={() => setTab("test")}
          >
            Test
          </button>
        </nav>

        {tab === "explore" && (
          <>
            {/* --- Controls --- */}
            <section className="card controls">
              <label className="prompt-label">
                Prompt
                <input
                  value={promptText}
                  onChange={(e) => {
                    setPromptText(e.target.value);
                    reset();
                  }}
                  onKeyDown={(e) => e.key === "Enter" && run()}
                />
              </label>
              <button onClick={run}>Complete one word →</button>
              {generated.length > 0 && (
                <button className="secondary" onClick={reset}>
                  Reset
                </button>
              )}
            </section>

            {/* --- Output --- */}
            {generated.length > 0 && (
              <section className="card output">
                <h2>Completion</h2>
                <p className="generated">
                  <span className="prompt-echo">{prompt.join(" ")}</span>{" "}
                  {generated.map((w, i) => (
                    <span
                      key={i}
                      className={`gen-token ${i === generated.length - 1 ? "latest" : ""}`}
                    >
                      {w}
                    </span>
                  ))}
                </p>

                {lastStep && (
                  <>
                    <h3>
                      What the model thought before choosing &ldquo;
                      {lastStep.token}
                      &rdquo;
                    </h3>
                    <div className="probs">
                      {lastStep.probs.map(({ word, p }) => (
                        <ProbBar key={word} word={word} p={p} />
                      ))}
                    </div>
                    <p className="note">
                      These are the actual probabilities the model computed over
                      its whole vocabulary. It sampled one word from this
                      distribution. Click &ldquo;Complete one word&rdquo; again
                      to continue.
                    </p>
                  </>
                )}
              </section>
            )}

            {/* --- How it works (accordion) --- */}
            <section className="card">
              <h2>How it works</h2>
              <p className="note" style={{ marginTop: 0 }}>
                Click any step to see what it does, with a live example from
                your prompt.
              </p>

              <div className="accordion">
                <Step
                  n={1}
                  title="Vocabulary"
                  open={open === 1}
                  onToggle={() => setOpen(open === 1 ? null : 1)}
                >
                  Based on its training data, the model knows a range of words.
                  Every word gets a number (an index). The model never sees the
                  word <code>{model.vocab[1]}</code>; it sees{" "}
                  <code>{index[model.vocab[1]]}</code>. This model currently
                  knows {model.vocab.length} words:
                  <div className="vocab-chips">
                    {model.vocab.map((w, i) => (
                      <span key={w} className="chip">
                        <b>{i}</b> {w}
                      </span>
                    ))}
                  </div>
                </Step>

                <Step
                  n={2}
                  title="Flatten"
                  open={open === 2}
                  onToggle={() => setOpen(open === 2 ? null : 2)}
                >
                  The model only understands numbers, not words, so we turn the
                  last {model.context_size} words into one long vector of 0s and
                  1s.
                  <p className="note" style={{ marginTop: 8 }}>
                    The vector has to be able to describe
                    <i> any </i> input, and an input is always &ldquo;the last{" "}
                    {model.context_size} words.&rdquo; For each word we need to
                    record two things:
                    <b> which word </b> it was ({model.vocab.length}{" "}
                    possibilities) and <b> where </b> it sat (
                    {model.context_size} positions). That's {model.context_size}{" "}
                    positions × {model.vocab.length} words ={" "}
                    {model.context_size * model.vocab.length} slots, one per
                    question &ldquo;is word X at position Y?&rdquo; A slot is 1
                    when true, 0 otherwise. Encoding position matters because
                    word order changes meaning (&ldquo;dog bit man&rdquo; vs
                    &ldquo;man bit dog&rdquo;).
                  </p>
                  <p className="demo-label">
                    For your prompt, the flattened input looks like this:
                  </p>
                  <div
                    className="flatten-table"
                    style={{
                      gridTemplateColumns: `auto repeat(${model.vocab.length}, minmax(0, 1fr))`,
                      gridTemplateRows: `repeat(${model.context_size}, auto) auto`,
                    }}
                  >
                    <div className="corner" />
                    {model.vocab.map((w) => (
                      <div key={w} className="col-label">
                        {w}
                      </div>
                    ))}
                    {Array.from({ length: model.context_size }).map(
                      (_, pos) => (
                        <Fragment key={pos}>
                          <div className="row-label">pos {pos}</div>
                          {Array.from({ length: model.vocab.length }).map(
                            (_, w) => {
                              const i = pos * model.vocab.length + w;
                              return (
                                <div
                                  key={w}
                                  className={`cell ${inputVec[i] ? "on" : "off"}`}
                                  title={`pos ${pos}: ${model.vocab[w]}`}
                                >
                                  {inputVec[i]}
                                </div>
                              );
                            },
                          )}
                        </Fragment>
                      ),
                    )}
                  </div>
                  <p className="note">
                    Your prompt as one {model.context_size * model.vocab.length}
                    -slot vector.
                  </p>
                </Step>

                <Step
                  n={3}
                  title="Matrix multiply"
                  open={open === 3}
                  onToggle={() => setOpen(open === 3 ? null : 3)}
                >
                  To predict the next word, the model runs one operation:{" "}
                  <code>logits = x @ W + b</code>. The flattened input{" "}
                  <code>x</code> is multiplied by the weight table{" "}
                  <code>W</code>, giving one raw score (a &ldquo;logit&rdquo;)
                  for every word in the vocabulary. That's all the
                  &ldquo;thinking.&rdquo;
                  <p className="demo-label">
                    The logits it computed for your prompt (top few):
                  </p>
                  <div className="logit-list">
                    {logitPreview.map(({ word, val }) => (
                      <div key={word} className="logit-row">
                        <span className="logit-word">{word}</span>
                        <span
                          className={`logit-val ${val >= 0 ? "pos" : "neg"}`}
                        >
                          {val.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="note">
                    Higher logit = the model thinks that word is more likely
                    next. These raw scores aren't probabilities yet.
                  </p>
                </Step>

                <Step
                  n={4}
                  title="Softmax"
                  open={open === 4}
                  onToggle={() => setOpen(open === 4 ? null : 4)}
                >
                  To create a distribution of words to pick from, the raw logits
                  are turned into probabilities that sum to 1, using{" "}
                  <code>e^logit</code> normalized across all words. Now each
                  word has a share of the total.
                  <p className="demo-label">
                    After softmax, the same top words become:
                  </p>
                  <div className="prob-mini">
                    {probPreview.map(({ word, p }) => (
                      <div key={word} className="prob-row">
                        <span className="prob-word">{word}</span>
                        <div className="prob-track">
                          <div
                            className="prob-fill"
                            style={{ width: `${Math.min(100, p * 100)}%` }}
                          />
                        </div>
                        <span className="prob-num">
                          {(p * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="note">
                    These add up to 100%. They're the chances the model assigns
                    to each possible next word.
                  </p>
                </Step>

                <Step
                  n={5}
                  title="Sample"
                  open={open === 5}
                  onToggle={() => setOpen(open === 5 ? null : 5)}
                >
                  Pick one word at random, weighted by its probability. Append
                  it to the prompt, slide the window forward by one word, and
                  repeat. That loop is the whole completion.
                  <p className="demo-label">
                    {hasCompletion
                      ? `Latest roll (after “${prompt.join(" ")} ${generated.slice(0, -1).join(" ")}”) picked:`
                      : "One roll of the dice for your prompt picked:"}
                  </p>
                  <div className="sample-result">
                    <span className="gen-token">{sampledWord}</span>
                    <span className="sample-p">
                      with {(sampledP * 100).toFixed(1)}% chance
                    </span>
                  </div>
                  <div className="prob-mini">
                    {sampleDist.slice(0, 5).map(({ word, p }) => (
                      <div key={word} className="prob-row">
                        <span className="prob-word">{word}</span>
                        <div className="prob-track">
                          <div className="prob-fill" style={{ width: `${Math.min(100, p * 100)}%` }} />
                        </div>
                        <span className="prob-num">{(p * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                  <p className="note">
                    Run &ldquo;Complete one word&rdquo; above and this updates
                    with each new word.
                  </p>
                </Step>
              </div>
            </section>
          </>
        )}

        {tab === "train" && (
          <>
            {/* --- Train your own --- */}
            <section className="card trainer">
              <h2>Train your own</h2>
              <p className="note">
                Type some sentences, hit Train, and watch a brand-new weight
                table get learned right here in your browser. Then use the test
                tab to try it out!
              </p>
              <textarea
                value={trainText}
                onChange={(e) => setTrainText(e.target.value)}
                rows={6}
                placeholder="one sentence per line…"
              />
              <div className="train-controls">
                <label>
                  <span className="field-title">
                    Epochs
                    <button
                      className="info-btn"
                      onClick={() => setShowEpochInfo((s) => !s)}
                      aria-label="What are epochs?"
                    >
                      <span>?</span>
                    </button>
                  </span>
                  <input
                    type="number"
                    min={10}
                    max={2000}
                    step={10}
                    value={epochs}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                  />
                </label>
                <button onClick={doTrain} disabled={training}>
                  {training ? "Training…" : trained ? "Retrain" : "Train"}
                </button>
              </div>

              {showEpochInfo && (
                <div className="epoch-popup">
                  <b>What are epochs?</b> One epoch is one full pass through all
                  your sentences. During each pass the model predicts every
                  word, measures how wrong it was (the <i>loss</i>), and nudges
                  its weights a little to do better next time. More epochs =
                  more passes = more learning, but with diminishing returns:
                  eventually the loss stops dropping and extra epochs just
                  overfit the tiny dataset. Watch the loss fall as the epochs
                  tick by.
                  <button
                    className="popup-close"
                    onClick={() => setShowEpochInfo(false)}
                  >
                    Got it
                  </button>
                </div>
              )}

              {training && progress && (
                <div className="train-progress">
                  <div className="track">
                    <div
                      className="fill"
                      style={{
                        width: `${((progress.epoch + 1) / epochs) * 100}%`,
                      }}
                    />
                  </div>
                  <span>
                    epoch {progress.epoch + 1}/{epochs} ·
                    <button
                      className="info-btn inline"
                      onClick={() => setShowLossInfo((s) => !s)}
                      aria-label="What is loss?"
                    >
                      <span>?</span>
                    </button>
                    loss {progress.loss.toFixed(4)}
                  </span>
                </div>
              )}

              {showLossInfo && (
                <div className="epoch-popup">
                  <b>What is loss?</b> Loss is the model's &ldquo;how wrong am
                  I? &rdquo; score: a single number measuring how far off its
                  predictions are from the correct next words. It's basically
                  the negative log of the probability it assigned to the right
                  word, averaged over all of them. High at first (weights are
                  random), then it falls as each epoch nudges the weights so
                  correct words get higher probability. The model trains by
                  making this number go down.
                  <button
                    className="popup-close"
                    onClick={() => setShowLossInfo(false)}
                  >
                    Got it
                  </button>
                </div>
              )}

              {trained && !training && (
                <p className="trained-note">
                  ✓ Trained a new model with {getModel().vocab.length} words.
                  Try completing above.
                </p>
              )}
            </section>
          </>
        )}

        {tab === "test" && (
          <section className="card test">
            <h2>Test bench</h2>
            <p className="note">
              Paste some training text about a topic in the Train tab, then come
              back here and compare how the model completes these prompts before
              vs after it learned your text.
            </p>

            <label className="test-label">
              Test prompts (one per line)
              <textarea
                value={testPrompts}
                onChange={(e) => setTestPrompts(e.target.value)}
                rows={4}
              />
            </label>
            <div className="test-controls">
              <label>
                Words per prompt
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={testLength}
                  onChange={(e) => setTestLength(Number(e.target.value))}
                />
              </label>
              <button onClick={runTest}>Generate completions</button>
            </div>

            {afterResults && (
              <table className="test-table">
                <thead>
                  <tr>
                    <th>Prompt</th>
                    <th>Before</th>
                    <th>After (current model)</th>
                  </tr>
                </thead>
                <tbody>
                  {parseTestPrompts().map((p, i) => (
                    <tr key={i}>
                      <td className="test-prompt">{p}</td>
                      <td className="test-before">
                        {beforeResults ? beforeResults[i].join(" ") : "—"}
                      </td>
                      <td className="test-after">
                        {afterResults[i].join(" ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {beforeResults && (
              <p className="note">
                The <b>Before</b> column is frozen from your first run; the{" "}
                <b>After</b> column reflects the current model. Retrain on new
                text and watch the After column change while Before stays put.
              </p>
            )}
          </section>
        )}

        <footer className="foot">
          Default model trained on 14 sentences about cats, dogs, mats and
          houses. See <code>model.json</code> for the full weight table.
        </footer>
      </div>
    </div>
  );
}
