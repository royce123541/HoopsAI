"""Redis pub/sub channel names, shared by publishers (live, replay) and the API."""


def channel(game_id: str) -> str:
    return f"game:{game_id}"
