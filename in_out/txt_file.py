from .utils import _fetch


def load_txt(filepath) -> str:
    """Loads a .txt file (located at filepath) and returns its full content as a string."""
    with open(_fetch(filepath), "r", encoding="utf-8") as txtfile:
        content = txtfile.read()
    return content


def save_txt(content: str, filepath):
    """Saves a string into a .txt file (named and located after filepath)."""
    with open(filepath, "w", encoding="utf-8") as txtfile:
        txtfile.write(content)
