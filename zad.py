import argparse
import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
import sys


def train_model(epochs: int):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_test, y_test)
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[INFO] Test accuracy: {acc:.4f}")
    return model, history


def save_learning_curve_csv(history, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.history.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows(zip(*[history.history[k] for k in keys]))
    print(f"[OK] Zapisano krzywą uczenia → {csv_path}")


def plot_learning_curve_from_csv(csv_path: Path):
    import matplotlib.pyplot as plt
    import csv as _csv

    if not csv_path.exists():
        print(f"[WARN] Brak pliku z krzywą: {csv_path}")
        return

    rows = list(_csv.DictReader(open(csv_path, "r", encoding="utf-8")))
    if not rows:
        print("[WARN] Plik CSV pusty.")
        return

    epochs = range(1, len(rows) + 1)
    loss = [float(r["loss"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows] if "val_loss" in rows[0] else None
    val_acc = (
        [float(r["val_accuracy"]) for r in rows] if "val_accuracy" in rows[0] else None
    )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train")
    if val_loss:
        plt.plot(epochs, val_loss, label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="train")
    if val_acc:
        plt.plot(epochs, val_acc, label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


def preprocess_image(path: Path):
    img = tf.keras.utils.load_img(path, color_mode="grayscale", target_size=(28, 28))
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    arr = tf.expand_dims(arr, axis=0)
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="MNIST: zapisz/wczytaj model (.keras). Krzywa uczenia do CSV. Opcjonalnie rozpoznaj obraz i/lub pokaż wykres."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model.keras"),
        help="Ścieżka do pliku modelu .keras (domyślnie model.keras).",
    )
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=Path("learning_curve.csv"),
        help="Plik z krzywą uczenia (CSV).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Liczba epok treningu przy uczeniu od zera.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Wymuś trening nawet, jeśli model już istnieje.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Po prostu pokaż wykres z --curve-csv (nawet bez treningu).",
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="(opcjonalnie) Ścieżka do obrazka z cyfrą do rozpoznania.",
    )
    args = parser.parse_args()

    history = None

    if len(sys.argv) == 1:
        if args.curve_csv.exists():
            plot_learning_curve_from_csv(args.curve_csv)
            return
        else:
            print("[INFO] Brak learning_curve.csv -> trening, zapis i wykres...")
            model, history = train_model(args.epochs)
            args.model.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.model)
            save_learning_curve_csv(history, args.curve_csv)
            plot_learning_curve_from_csv(args.curve_csv)
            return

    if args.plot and not args.curve_csv.exists():
        print("[INFO] --plot bez CSV -> trening, zapis i wykres...")
        model, history = train_model(args.epochs)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model)
        save_learning_curve_csv(history, args.curve_csv)
        plot_learning_curve_from_csv(args.curve_csv)
        return

    if args.model.exists() and not args.train:
        print(f"[INFO] Wczytuję model: {args.model}")
        model = tf.keras.models.load_model(args.model)
    else:
        print("[INFO] Trening modelu…")
        model, history = train_model(args.epochs)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model)
        print(f"[OK] Zapisano model → {args.model}")
        save_learning_curve_csv(history, args.curve_csv)

    if args.plot:
        plot_learning_curve_from_csv(args.curve_csv)

    if args.image:
        if not args.image.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {args.image}")
        x = preprocess_image(args.image)
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        print(f"[PRED] {args.image.name}: {pred} (p={conf:.4f})")


if __name__ == "__main__":
    main()
