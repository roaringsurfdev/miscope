"""Utility functions for the dashboard."""


def parse_checkpoint_epochs(text: str) -> list[int]:
    """Parse comma-separated checkpoint epochs from text input.

    Returns:
        Sorted list of unique epoch numbers, or empty list if text is blank.
    """
    if not text.strip():
        return []

    epochs = []
    for part in text.split(","):
        part = part.strip()
        if part:
            epochs.append(int(part))

    return sorted(set(epochs))
