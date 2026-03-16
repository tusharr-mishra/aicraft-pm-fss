import argparse
import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

import pandas as pd

from predict_rul import (
    build_columns,
    fit_preprocessor,
    format_response,
    load_base_data,
    load_model,
    make_engine_sequence,
)


def maintenance_cost_analysis(
    predicted_rul: float,
    preventive_cost: float = 20000.0,
    corrective_repair_cost: float = 80000.0,
    downtime_cost_per_cycle: float = 2000.0,
    downtime_cycles_if_failure: float = 5.0,
    secondary_damage_cost: float = 15000.0,
    inspection_cost: float = 1000.0,
    monitoring_cost_per_cycle: float = 50.0,
    discount_rate_per_cycle: float = 0.0,
    weibull_shape: float = 2.0,
    planning_horizon: int = 20,
) -> dict:
    """Estimate expected maintenance economics using reliability assumptions.

    Model notes:
    - Failure within planning horizon is estimated using a Weibull residual-life model.
    - Predicted RUL is treated as mean residual life; Weibull scale is inferred from mean.
    - Wait strategy includes monitoring + discounted scheduled maintenance if no failure.
    - Failure strategy includes corrective repair, downtime impact, and secondary damage.
    """
    predicted_rul = max(float(predicted_rul), 1e-6)
    horizon = max(int(planning_horizon), 1)
    shape = max(float(weibull_shape), 1e-3)

    # Infer Weibull scale from mean residual life: E[T] = lambda * Gamma(1 + 1/k)
    weibull_scale = predicted_rul / math.gamma(1.0 + 1.0 / shape)
    weibull_scale = max(weibull_scale, 1e-6)

    probability_of_failure = 1.0 - math.exp(-((horizon / weibull_scale) ** shape))
    probability_of_failure = min(max(probability_of_failure, 0.0), 1.0)

    discount_factor = (1.0 + max(float(discount_rate_per_cycle), 0.0)) ** (-horizon)

    cost_if_maintain_now = float(preventive_cost) + float(inspection_cost)
    future_planned_cost_if_survive = float(preventive_cost) * discount_factor

    unplanned_failure_event_cost = (
        float(corrective_repair_cost)
        + float(secondary_damage_cost)
        + float(downtime_cost_per_cycle) * float(downtime_cycles_if_failure)
    )

    expected_cost_if_wait = (
        probability_of_failure * unplanned_failure_event_cost
        + (1.0 - probability_of_failure) * future_planned_cost_if_survive
        + float(inspection_cost)
        + float(monitoring_cost_per_cycle) * horizon
    )

    estimated_savings_if_maintain_now = expected_cost_if_wait - cost_if_maintain_now

    recommendation = "maintain_now" if estimated_savings_if_maintain_now > 0 else "wait_and_monitor"

    return {
        "predicted_rul": round(float(predicted_rul), 4),
        "planning_horizon": horizon,
        "probability_of_failure": round(float(probability_of_failure), 4),
        "cost_if_maintain_now": round(float(cost_if_maintain_now), 2),
        "expected_cost_if_wait": round(float(expected_cost_if_wait), 2),
        "estimated_savings_if_maintain_now": round(float(estimated_savings_if_maintain_now), 2),
        "weibull_shape": round(shape, 4),
        "weibull_scale": round(float(weibull_scale), 4),
        "discount_factor": round(float(discount_factor), 6),
        "assumptions": {
            "residual_life_distribution": "weibull",
            "predicted_rul_treated_as": "mean_residual_life",
            "expected_wait_cost_formula": "P_fail*C_failure + (1-P_fail)*C_planned_discounted + inspection + monitoring",
        },
        "model_inputs": {
            "preventive_cost": float(preventive_cost),
            "corrective_repair_cost": float(corrective_repair_cost),
            "downtime_cost_per_cycle": float(downtime_cost_per_cycle),
            "downtime_cycles_if_failure": float(downtime_cycles_if_failure),
            "secondary_damage_cost": float(secondary_damage_cost),
            "inspection_cost": float(inspection_cost),
            "monitoring_cost_per_cycle": float(monitoring_cost_per_cycle),
            "discount_rate_per_cycle": float(discount_rate_per_cycle),
            "weibull_shape": float(shape),
        },
        "recommendation": recommendation,
    }


class PredictorService:
    def __init__(
        self,
        model_path: Path,
        train_file: Path,
        test_file: Path,
        openrouter_api_key: str | None = None,
        openrouter_model: str = "google/gemini-2.0-flash-001",
    ):
        self.model = load_model(model_path)
        self.sequence_length = int(self.model.input_shape[1])
        self.model_feature_count = int(self.model.input_shape[2])
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

        train_df, test_df = load_base_data(train_file, test_file)
        self.scaler, self.feature_cols = fit_preprocessor(train_df)
        self.test_df = test_df

        if len(self.feature_cols) != self.model_feature_count:
            raise ValueError(
                f"Model expects {self.model_feature_count} features, but preprocessing produced {len(self.feature_cols)}."
            )

    def _require_openrouter_key(self):
        if not self.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is not configured on server.")

    def _openrouter_chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        self._require_openrouter_key()

        body = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        req = url_request.Request(
            self.openrouter_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with url_request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except url_error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {error_text}") from exc
        except url_error.URLError as exc:
            raise RuntimeError(f"OpenRouter connection error: {exc.reason}") from exc

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no choices.")
        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("OpenRouter returned empty content.")
        return content

    def predict_for_engine(
        self,
        engine_id: int,
        include_reasoning: bool = False,
        cost_parameters: dict | None = None,
    ) -> dict:
        sequence = make_engine_sequence(
            self.test_df,
            engine_id=engine_id,
            feature_cols=self.feature_cols,
            scaler=self.scaler,
            sequence_length=self.sequence_length,
        )
        prediction = float(self.model.predict(sequence, verbose=0).reshape(-1)[0])
        response = format_response(engine_id, prediction)
        return self._augment_response(response, include_reasoning, cost_parameters)

    def predict_for_sequence(
        self,
        rows: list[dict],
        include_reasoning: bool = False,
        cost_parameters: dict | None = None,
    ) -> dict:
        if not rows:
            raise ValueError("'sequence' must contain at least one row.")

        df = pd.DataFrame(rows)
        required_raw = ["setting1", "setting2", "setting3", *[f"sensor{i}" for i in range(1, 22)]]
        missing = [c for c in required_raw if c not in df.columns]
        if missing:
            raise ValueError(
                "Sequence rows must include setting1, setting2, setting3, sensor1..sensor21. "
                f"Missing: {missing}"
            )

        if "engine_id" not in df.columns:
            df.insert(0, "engine_id", 1)
        if "cycle" not in df.columns:
            df.insert(1, "cycle", range(1, len(df) + 1))

        # Keep only expected raw columns and preserve order.
        ordered_cols = [c for c in build_columns() if c in df.columns]
        df = df[ordered_cols]

        engine_id = int(df["engine_id"].iloc[0])
        sequence = make_engine_sequence(
            df,
            engine_id=engine_id,
            feature_cols=self.feature_cols,
            scaler=self.scaler,
            sequence_length=self.sequence_length,
        )
        prediction = float(self.model.predict(sequence, verbose=0).reshape(-1)[0])
        response = format_response(engine_id, prediction)
        return self._augment_response(response, include_reasoning, cost_parameters)

    def _augment_response(
        self,
        base_response: dict,
        include_reasoning: bool,
        cost_parameters: dict | None,
    ) -> dict:
        predicted_rul = float(base_response["predicted_rul"])
        allowed_cost_keys = {
            "preventive_cost",
            "corrective_repair_cost",
            "downtime_cost_per_cycle",
            "downtime_cycles_if_failure",
            "secondary_damage_cost",
            "inspection_cost",
            "monitoring_cost_per_cycle",
            "discount_rate_per_cycle",
            "weibull_shape",
            "planning_horizon",
        }
        safe_overrides = {}
        if isinstance(cost_parameters, dict):
            safe_overrides = {k: cost_parameters[k] for k in cost_parameters if k in allowed_cost_keys}

        cost_analysis = maintenance_cost_analysis(predicted_rul=predicted_rul, **safe_overrides)

        enriched = {
            **base_response,
            "cost_analysis": cost_analysis,
        }

        if include_reasoning:
            try:
                enriched["reasoning"] = self._generate_reasoning(enriched)
            except Exception as exc:
                enriched["reasoning_error"] = f"Reasoning unavailable: {exc}"

        return enriched

    def _generate_reasoning(self, prediction_payload: dict) -> str:
        prompt = (
            "You are an aviation maintenance analyst.\n"
            "Given the prediction payload below, provide concise operational guidance in 5-8 lines.\n"
            "Include: risk level, maintenance action, and short rationale.\n"
            "Use plain language and avoid speculation.\n\n"
            f"Payload:\n{json.dumps(prediction_payload, indent=2)}"
        )
        return self._openrouter_chat(
            system_prompt="You provide maintenance recommendations from ML outputs.",
            user_prompt=prompt,
            temperature=0.2,
        )

    def chat_with_reference(self, question: str, reference_context: dict | None = None) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        expected_answer = self._answer_expected_question(question, reference_context)
        if expected_answer is not None:
            return {"question": question.strip(), "answer": expected_answer}

        reference_text = "No prediction context provided."
        if reference_context is not None:
            reference_text = json.dumps(reference_context, indent=2)

        user_prompt = (
            "You are helping with CMAPSS engine health decisions.\n"
            "Answer the question using the provided reference context when relevant.\n"
            "If context is missing for the question, say what additional data is needed.\n"
            "Keep response concise and practical.\n\n"
            f"Reference context:\n{reference_text}\n\n"
            f"Question:\n{question.strip()}"
        )

        answer = self._openrouter_chat(
            system_prompt="You are an expert maintenance and reliability assistant.",
            user_prompt=user_prompt,
            temperature=0.3,
        )
        return {"question": question.strip(), "answer": answer}

    def _answer_expected_question(self, question: str, reference_context: dict | None) -> str | None:
        """Return deterministic answers for common maintenance questions.

        This improves reliability for operational queries even without an LLM call.
        """
        q = question.strip().lower()
        if not reference_context:
            return None

        predicted_rul = reference_context.get("predicted_rul")
        cost_ctx = reference_context.get("cost_analysis", {})

        try:
            predicted_rul = float(predicted_rul)
        except (TypeError, ValueError):
            predicted_rul = None

        if predicted_rul is None:
            return None

        # 1) "cost for 10 more cycles" style questions.
        if "cost" in q and ("cycle" in q or "horizon" in q or "wait" in q):
            horizon_match = re.search(r"(\d+)\s*(more\s*)?(cycles?|cycle)", q)
            if not horizon_match:
                horizon_match = re.search(r"(\d+)\s*(day|days|week|weeks)?", q)

            horizon = int(horizon_match.group(1)) if horizon_match else int(cost_ctx.get("planning_horizon", 20))
            if horizon <= 0:
                horizon = 1

            model_inputs = cost_ctx.get("model_inputs", {}) if isinstance(cost_ctx, dict) else {}
            preventive_cost = float(model_inputs.get("preventive_cost", 20000.0))
            corrective_repair_cost = float(model_inputs.get("corrective_repair_cost", 80000.0))
            downtime_cost_per_cycle = float(model_inputs.get("downtime_cost_per_cycle", 2000.0))
            downtime_cycles_if_failure = float(model_inputs.get("downtime_cycles_if_failure", 5.0))
            secondary_damage_cost = float(model_inputs.get("secondary_damage_cost", 15000.0))
            inspection_cost = float(model_inputs.get("inspection_cost", 1000.0))
            monitoring_cost_per_cycle = float(model_inputs.get("monitoring_cost_per_cycle", 50.0))
            discount_rate_per_cycle = float(model_inputs.get("discount_rate_per_cycle", 0.0))
            weibull_shape = float(model_inputs.get("weibull_shape", cost_ctx.get("weibull_shape", 2.0)))

            recalculated = maintenance_cost_analysis(
                predicted_rul=predicted_rul,
                preventive_cost=preventive_cost,
                corrective_repair_cost=corrective_repair_cost,
                downtime_cost_per_cycle=downtime_cost_per_cycle,
                downtime_cycles_if_failure=downtime_cycles_if_failure,
                secondary_damage_cost=secondary_damage_cost,
                inspection_cost=inspection_cost,
                monitoring_cost_per_cycle=monitoring_cost_per_cycle,
                discount_rate_per_cycle=discount_rate_per_cycle,
                weibull_shape=weibull_shape,
                planning_horizon=horizon,
            )

            return (
                f"For a {horizon}-cycle horizon: expected wait cost is ${recalculated['expected_cost_if_wait']:,.2f}, "
                f"maintain-now cost is ${recalculated['cost_if_maintain_now']:,.2f}, "
                f"failure probability is {recalculated['probability_of_failure'] * 100:.2f}%, "
                f"and recommendation is {recalculated['recommendation']}."
            )

        # 2) Recommendation meaning.
        if "wait_and_monitor" in q or ("what does" in q and "recommendation" in q) or ("recommendation" in q and "mean" in q):
            recommendation = str(cost_ctx.get("recommendation", "wait_and_monitor"))
            if recommendation == "wait_and_monitor":
                return (
                    "wait_and_monitor means immediate maintenance is not economically justified right now. "
                    "Continue monitoring sensor trends and re-evaluate if RUL drops quickly or failure probability rises."
                )
            return (
                "maintain_now means expected risk-adjusted future cost is higher than preventive maintenance cost, "
                "so immediate maintenance is recommended."
            )

        # 3) Risk/probability interpretation.
        if "probability" in q or "risk" in q or "safe" in q:
            prob = float(cost_ctx.get("probability_of_failure", 0.0))
            level = "low" if prob < 0.05 else "moderate" if prob < 0.2 else "high"
            return (
                f"Current estimated failure probability is {prob * 100:.2f}% ({level} risk) over the configured horizon. "
                "Risk should be reviewed again after new cycle data arrives."
            )

        return None


def create_handler(service: PredictorService):
    class PredictHandler(BaseHTTPRequestHandler):
        def _write_json(self, status_code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self._write_json(200, {"status": "ok"})
            else:
                self._write_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path not in ["/predict", "/chat"]:
                self._write_json(404, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    raise ValueError("Request body is empty.")

                raw_body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw_body)

                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object.")

                if self.path == "/chat":
                    question = str(payload.get("question", ""))
                    reference_context = payload.get("reference_context")
                    response = service.chat_with_reference(
                        question=question,
                        reference_context=reference_context if isinstance(reference_context, dict) else None,
                    )
                    self._write_json(200, response)
                    return

                include_reasoning = bool(payload.get("include_reasoning", False))
                cost_parameters = payload.get("cost_parameters")
                if cost_parameters is not None and not isinstance(cost_parameters, dict):
                    raise ValueError("'cost_parameters' must be an object when provided.")

                if "sequence" in payload:
                    if not isinstance(payload["sequence"], list):
                        raise ValueError("'sequence' must be a list of row objects.")
                    response = service.predict_for_sequence(
                        payload["sequence"],
                        include_reasoning=include_reasoning,
                        cost_parameters=cost_parameters,
                    )
                elif "engine_id" in payload:
                    response = service.predict_for_engine(
                        int(payload["engine_id"]),
                        include_reasoning=include_reasoning,
                        cost_parameters=cost_parameters,
                    )
                else:
                    raise ValueError("Provide either 'engine_id' or 'sequence' in request body.")

                self._write_json(200, response)
            except ValueError as exc:
                self._write_json(400, {"error": str(exc)})
            except json.JSONDecodeError:
                self._write_json(400, {"error": "Invalid JSON body."})
            except Exception as exc:
                self._write_json(500, {"error": f"Internal server error: {exc}"})

        def log_message(self, fmt: str, *args):
            return

    return PredictHandler


def parse_args():
    parser = argparse.ArgumentParser(description="CMAPSS model deployment HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=Path, default=Path("cmapss_cnn_lstm_best.keras"))
    parser.add_argument("--train-file", type=Path, default=Path("data/train_FD001.txt"))
    parser.add_argument("--test-file", type=Path, default=Path("data/test_FD001.txt"))
    parser.add_argument(
        "--openrouter-model",
        default=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001"),
        help="OpenRouter model id used when include_reasoning=true.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    openrouter_api_key = "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"
    service = PredictorService(
        args.model_path,
        args.train_file,
        args.test_file,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=args.openrouter_model,
    )
    handler = create_handler(service)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Server running on http://{args.host}:{args.port}")
    print("POST /predict with JSON body {'engine_id': 1} or {'sequence': [...]} ")
    print("POST /chat with JSON body {'question': '...', 'reference_context': {...}}")
    print("Optional JSON field: include_reasoning=true (requires OPENROUTER_API_KEY)")
    print("GET /health")
    server.serve_forever()


if __name__ == "__main__":
    main()