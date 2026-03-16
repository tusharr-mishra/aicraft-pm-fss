import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

keras = tf.keras
layers = tf.keras.layers


class AttentionLayer(layers.Layer):
    """Custom attention layer used by the saved attention model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


def build_columns():
    return [
        "engine_id",
        "cycle",
        "setting1",
        "setting2",
        "setting3",
        *[f"sensor{i}" for i in range(1, 22)],
    ]


def add_advanced_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    sensor_cols = [f"sensor{i}" for i in range(1, 22)]
    out = df.copy()

    for engine_id in out["engine_id"].unique():
        mask = out["engine_id"] == engine_id
        for col in sensor_cols:
            out.loc[mask, f"{col}_rolling_mean"] = out.loc[mask, col].rolling(
                window=window_size, min_periods=1
            ).mean()
            out.loc[mask, f"{col}_rolling_std"] = out.loc[mask, col].rolling(
                window=window_size, min_periods=1
            ).std().fillna(0)
            first_val = out.loc[mask, col].iloc[0]
            out.loc[mask, f"{col}_trend"] = out.loc[mask, col] - first_val

    out["cycle_norm"] = out.groupby("engine_id")["cycle"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    return out


def load_base_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = build_columns()
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=cols)
    train_df["RUL"] = train_df.groupby("engine_id")["cycle"].transform("max") - train_df["cycle"]
    train_df["RUL"] = train_df["RUL"].clip(upper=125)
    return train_df, test_df


def fit_preprocessor(train_df: pd.DataFrame):
    train_feat = add_advanced_features(train_df)
    all_features = [
        col for col in train_feat.columns if col not in ["engine_id", "cycle", "RUL"]
    ]
    feature_variance = train_feat[all_features].var()
    feature_cols = feature_variance[feature_variance > 0.001].index.tolist()

    scaler = StandardScaler()
    scaler.fit(train_feat[feature_cols].astype(np.float64))
    return scaler, feature_cols


def make_engine_sequence(
    df: pd.DataFrame,
    engine_id: int,
    feature_cols: list[str],
    scaler: StandardScaler,
    sequence_length: int,
) -> np.ndarray:
    engine_df = df[df["engine_id"] == engine_id].copy()
    if engine_df.empty:
        raise ValueError(f"Engine id {engine_id} not found in provided data.")

    engine_df = add_advanced_features(engine_df)
    missing_cols = [c for c in feature_cols if c not in engine_df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required columns after feature engineering: {missing_cols}")

    features = scaler.transform(engine_df[feature_cols].astype(np.float64)).astype(np.float32)

    if len(features) >= sequence_length:
        seq = features[-sequence_length:]
    else:
        pad = np.zeros((sequence_length - len(features), len(feature_cols)), dtype=np.float32)
        seq = np.vstack([pad, features])

    return np.expand_dims(seq, axis=0)


def load_model(model_path: Path):
    return keras.models.load_model(
        model_path,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    )


def read_custom_sequence(sequence_file: Path) -> pd.DataFrame:
    df = pd.read_csv(sequence_file)
    required_raw = ["setting1", "setting2", "setting3", *[f"sensor{i}" for i in range(1, 22)]]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(
            "Custom sequence CSV must include columns: setting1, setting2, setting3, sensor1..sensor21. "
            f"Missing: {missing}"
        )

    if "engine_id" not in df.columns:
        df.insert(0, "engine_id", 1)
    if "cycle" not in df.columns:
        df.insert(1, "cycle", np.arange(1, len(df) + 1))
    return df


def format_response(engine_id: int, prediction: float) -> dict:
    return {
        "engine_id": int(engine_id),
        "predicted_rul": float(round(prediction, 4)),
        "response": f"Predicted Remaining Useful Life for engine {engine_id}: {prediction:.2f}",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict CMAPSS RUL from parameterized inputs.")
    parser.add_argument("--model-path", type=Path, default=Path("cmapss_cnn_lstm_best.keras"))
    parser.add_argument("--train-file", type=Path, default=Path("data/train_FD001.txt"))
    parser.add_argument("--test-file", type=Path, default=Path("data/test_FD001.txt"))
    parser.add_argument(
        "--engine-id",
        type=int,
        default=1,
        help="Engine id from test file for prediction (ignored when --sequence-file is used).",
    )
    parser.add_argument(
        "--sequence-file",
        type=Path,
        default=None,
        help="Optional CSV containing one engine sequence with columns setting1..3 and sensor1..21.",
    )
    parser.add_argument(
        "--response-format",
        choices=["json", "text"],
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional path to save the response payload.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(args.model_path)
    sequence_length = int(model.input_shape[1])
    model_feature_count = int(model.input_shape[2])

    train_df, test_df = load_base_data(args.train_file, args.test_file)
    scaler, feature_cols = fit_preprocessor(train_df)

    if len(feature_cols) != model_feature_count:
        raise ValueError(
            f"Model expects {model_feature_count} features, but preprocessing produced {len(feature_cols)}."
        )

    if args.sequence_file is not None:
        custom_df = read_custom_sequence(args.sequence_file)
        sequence = make_engine_sequence(
            custom_df,
            engine_id=int(custom_df["engine_id"].iloc[0]),
            feature_cols=feature_cols,
            scaler=scaler,
            sequence_length=sequence_length,
        )
        pred = float(model.predict(sequence, verbose=0).reshape(-1)[0])
        response = format_response(int(custom_df["engine_id"].iloc[0]), pred)
    else:
        sequence = make_engine_sequence(
            test_df,
            engine_id=args.engine_id,
            feature_cols=feature_cols,
            scaler=scaler,
            sequence_length=sequence_length,
        )
        pred = float(model.predict(sequence, verbose=0).reshape(-1)[0])
        response = format_response(args.engine_id, pred)

    if args.response_format == "json":
        output = json.dumps(response, indent=2)
    else:
        output = response["response"]

    if args.output_file is not None:
        args.output_file.write_text(output, encoding="utf-8")

    print(output)


if __name__ == "__main__":
    main()