import csv
from .utils import _fetch


def save_csv(data: dict, filepath) -> None:
    """
    Save a dictionary {key: value} into a CSV file with 2 columns: key, value.
    """
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "value"])
        for key, value in data.items():
            writer.writerow([key, value])


def load_csv(filepath) -> dict:
    """
    Load a CSV file with 2 columns (key, value) into a dictionary.
    """
    data = {}
    with open(_fetch(filepath), "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            key, value = row
            try:
                value = float(value)
            except ValueError:
                pass
            data[key] = value
    return data