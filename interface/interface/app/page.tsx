"use client";

import { FormEvent, useMemo, useState } from "react";

type PredictResponse = {
  engine_id: number;
  predicted_rul: number;
  response: string;
  cost_analysis?: {
    predicted_rul: number;
    planning_horizon: number;
    probability_of_failure: number;
    cost_if_maintain_now: number;
    expected_cost_if_wait: number;
    estimated_savings_if_maintain_now: number;
    recommendation: "maintain_now" | "wait_and_monitor";
  };
  reasoning?: string;
  reasoning_error?: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

const INITIAL_VALUES: Record<string, string> = {
  engine_id: "1",
  cycle: "1",
  setting1: "0.0",
  setting2: "0.0",
  setting3: "100.0",
  sensor1: "518.67",
  sensor2: "642.12",
  sensor3: "1589.7",
  sensor4: "1400.6",
  sensor5: "14.62",
  sensor6: "21.61",
  sensor7: "553.9",
  sensor8: "2388.0",
  sensor9: "9046.19",
  sensor10: "1.3",
  sensor11: "47.47",
  sensor12: "521.66",
  sensor13: "2388.02",
  sensor14: "8138.62",
  sensor15: "8.4195",
  sensor16: "0.03",
  sensor17: "392.0",
  sensor18: "2388.0",
  sensor19: "100.0",
  sensor20: "39.06",
  sensor21: "23.419",
};

const FIELD_ORDER = [
  "engine_id",
  "cycle",
  "setting1",
  "setting2",
  "setting3",
  ...Array.from({ length: 21 }, (_, i) => `sensor${i + 1}`),
];

export default function Home() {
  const [formValues, setFormValues] =
    useState<Record<string, string>>(INITIAL_VALUES);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [includeReasoning, setIncludeReasoning] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const predictorUrl =
    process.env.NEXT_PUBLIC_PREDICTOR_URL ?? "http://127.0.0.1:8000/predict";
  const chatUrl = predictorUrl.replace(/\/predict$/, "/chat");

  const groupedFields = useMemo(
    () => ({
      core: ["engine_id", "cycle"],
      settings: ["setting1", "setting2", "setting3"],
      sensors: Array.from({ length: 21 }, (_, i) => `sensor${i + 1}`),
    }),
    [],
  );

  function updateField(name: string, value: string) {
    setFormValues((prev) => ({ ...prev, [name]: value }));
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payloadRow: Record<string, number> = {};
      for (const key of FIELD_ORDER) {
        const parsed = Number(formValues[key]);
        if (Number.isNaN(parsed)) {
          throw new Error(`Invalid number in ${key}`);
        }
        payloadRow[key] = parsed;
      }

      const response = await fetch(predictorUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sequence: [payloadRow],
          include_reasoning: includeReasoning,
        }),
      });

      const data = (await response.json()) as
        | PredictResponse
        | { error: string };
      if (!response.ok) {
        const message = "error" in data ? data.error : "Prediction failed";
        throw new Error(message);
      }

      setResult(data as PredictResponse);
    } catch (submitError) {
      setError(
        submitError instanceof Error ? submitError.message : "Unexpected error",
      );
    } finally {
      setLoading(false);
    }
  }

  async function onAskQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = chatInput.trim();
    if (!question) {
      return;
    }

    setChatLoading(true);
    setChatMessages((prev) => [...prev, { role: "user", content: question }]);
    setChatInput("");

    try {
      const response = await fetch(chatUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          reference_context: result ?? undefined,
        }),
      });

      const data = (await response.json()) as
        | { answer: string }
        | { error: string };

      if (!response.ok) {
        const message = "error" in data ? data.error : "Chat request failed";
        throw new Error(message);
      }

      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: (data as { answer: string }).answer },
      ]);
    } catch (chatError) {
      const message = chatError instanceof Error ? chatError.message : "Unexpected chat error";
      setChatMessages((prev) => [...prev, { role: "assistant", content: `Error: ${message}` }]);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_10%_10%,#f6d365_0%,transparent_40%),radial-gradient(circle_at_90%_20%,#fda085_0%,transparent_35%),linear-gradient(165deg,#081120_0%,#0b1f3b_60%,#102844_100%)] px-4 py-8 text-slate-100 md:px-10">
      <main className="mx-auto grid w-full max-w-7xl gap-6 lg:grid-cols-[1.6fr_1fr]">
        <section className="rounded-3xl border border-white/20 bg-white/10 p-5 shadow-2xl backdrop-blur-md md:p-8">
          <p className="mb-2 text-xs uppercase tracking-[0.3em] text-amber-200">
            CMAPSS Deployment Console
          </p>
          <h1 className="text-3xl font-bold leading-tight text-white md:text-4xl">
            Sensor Input to RUL Prediction
          </h1>
          <p className="mt-3 max-w-3xl text-sm text-slate-200 md:text-base">
            Enter one cycle of engine settings and sensor values. The interface
            calls your deployed model API and returns predicted Remaining Useful
            Life instantly.
          </p>

          <form onSubmit={onSubmit} className="mt-8 space-y-8">
            <FieldGroup title="Core">
              {groupedFields.core.map((name) => (
                <NumberField
                  key={name}
                  name={name}
                  value={formValues[name]}
                  onChange={updateField}
                />
              ))}
            </FieldGroup>

            <FieldGroup title="Operating Settings">
              {groupedFields.settings.map((name) => (
                <NumberField
                  key={name}
                  name={name}
                  value={formValues[name]}
                  onChange={updateField}
                />
              ))}
            </FieldGroup>

            <FieldGroup title="Sensors 1-21">
              {groupedFields.sensors.map((name) => (
                <NumberField
                  key={name}
                  name={name}
                  value={formValues[name]}
                  onChange={updateField}
                />
              ))}
            </FieldGroup>

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-2xl bg-linear-to-r from-amber-300 via-orange-300 to-rose-300 px-5 py-4 font-semibold text-slate-900 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Predicting..." : "Run Prediction"}
            </button>

            <label className="flex items-center gap-2 rounded-xl border border-white/10 bg-black/20 p-3 text-sm text-slate-200">
              <input
                type="checkbox"
                checked={includeReasoning}
                onChange={(e) => setIncludeReasoning(e.target.checked)}
                className="h-4 w-4 accent-amber-300"
              />
              Include Gemini reasoning (OpenRouter)
            </label>
          </form>
        </section>

        <aside className="space-y-6">
          <section className="rounded-3xl border border-white/20 bg-[#091a2f]/85 p-6 shadow-2xl backdrop-blur-md md:p-8">
          <h2 className="text-xl font-semibold text-amber-200">Output</h2>
          <p className="mt-2 text-sm text-slate-300">
            Response from your deployment endpoint appears here.
          </p>

          <div className="mt-6 rounded-2xl border border-white/10 bg-black/25 p-5">
            {!result && !error && (
              <p className="text-sm text-slate-400">
                Submit input data to see prediction output.
              </p>
            )}
            {error && (
              <p className="text-sm font-medium text-rose-300">{error}</p>
            )}
            {result && (
              <div className="space-y-3 text-sm">
                <p className="text-slate-300">Engine ID</p>
                <p className="text-2xl font-bold text-white">
                  {result.engine_id}
                </p>
                <p className="text-slate-300">Predicted RUL</p>
                <p className="text-4xl font-black text-amber-300">
                  {result.predicted_rul.toFixed(2)}
                </p>
                <p className="rounded-xl border border-amber-200/20 bg-amber-200/10 p-3 text-slate-100">
                  {result.response}
                </p>

                {result.cost_analysis && (
                  <div className="space-y-2 rounded-xl border border-cyan-200/20 bg-cyan-200/10 p-3 text-slate-100">
                    <p className="text-sm font-semibold text-cyan-100">Cost Analysis</p>
                    <p>
                      Probability of Failure: {(result.cost_analysis.probability_of_failure * 100).toFixed(1)}%
                    </p>
                    <p>Maintain Now Cost: ${result.cost_analysis.cost_if_maintain_now.toLocaleString()}</p>
                    <p>Expected Wait Cost: ${result.cost_analysis.expected_cost_if_wait.toLocaleString()}</p>
                    <p>
                      Estimated Savings (Maintain Now): ${result.cost_analysis.estimated_savings_if_maintain_now.toLocaleString()}
                    </p>
                    <p className="font-semibold">
                      Recommendation: {result.cost_analysis.recommendation === "maintain_now" ? "Maintain now" : "Wait and monitor"}
                    </p>
                  </div>
                )}

                {result.reasoning && (
                  <div className="space-y-2 rounded-xl border border-violet-200/20 bg-violet-200/10 p-3 text-slate-100">
                    <p className="text-sm font-semibold text-violet-100">Gemini Reasoning</p>
                    <p className="whitespace-pre-wrap">{result.reasoning}</p>
                  </div>
                )}

                {result.reasoning_error && (
                  <p className="rounded-xl border border-rose-200/20 bg-rose-200/10 p-3 text-rose-200">
                    {result.reasoning_error}
                  </p>
                )}
              </div>
            )}
          </div>
          </section>

          <section className="rounded-3xl border border-white/20 bg-[#102641]/90 p-6 shadow-2xl backdrop-blur-md md:p-8">
            <h2 className="text-xl font-semibold text-cyan-100">Ask Assistant</h2>
            <p className="mt-2 text-sm text-slate-300">
              Ask follow-up questions about this prediction, risk, and maintenance decision.
            </p>

            <div className="mt-4 max-h-72 space-y-3 overflow-y-auto rounded-2xl border border-white/10 bg-black/20 p-4">
              {chatMessages.length === 0 && (
                <p className="text-sm text-slate-400">
                  Start with a question like: Why is the recommendation to wait and monitor?
                </p>
              )}
              {chatMessages.map((message, idx) => (
                <div
                  key={`${message.role}-${idx}`}
                  className={`rounded-xl p-3 text-sm ${
                    message.role === "user"
                      ? "ml-8 border border-amber-200/20 bg-amber-200/10 text-amber-50"
                      : "mr-8 border border-cyan-200/20 bg-cyan-200/10 text-cyan-50"
                  }`}
                >
                  {message.content}
                </div>
              ))}
            </div>

            <form onSubmit={onAskQuestion} className="mt-4 flex gap-2">
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about risk, cost, or next action..."
                className="flex-1 rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm text-white outline-none ring-cyan-300 placeholder:text-slate-500 focus:ring-2"
              />
              <button
                type="submit"
                disabled={chatLoading}
                className="rounded-xl bg-cyan-300 px-4 py-2 text-sm font-semibold text-slate-900 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {chatLoading ? "Asking..." : "Ask"}
              </button>
            </form>
          </section>
        </aside>
      </main>
    </div>
  );
}

function FieldGroup({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <fieldset>
      <legend className="mb-3 text-xs uppercase tracking-[0.25em] text-cyan-200">
        {title}
      </legend>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{children}</div>
    </fieldset>
  );
}

function NumberField({
  name,
  value,
  onChange,
}: {
  name: string;
  value: string;
  onChange: (name: string, value: string) => void;
}) {
  return (
    <label className="flex flex-col gap-1 rounded-xl border border-white/10 bg-black/20 p-3">
      <span className="text-xs uppercase tracking-[0.2em] text-slate-300">
        {name}
      </span>
      <input
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(name, e.target.value)}
        className="rounded-lg border border-white/20 bg-white/10 px-3 py-2 text-sm text-white outline-none ring-amber-300 placeholder:text-slate-500 focus:ring-2"
      />
    </label>
  );
}
