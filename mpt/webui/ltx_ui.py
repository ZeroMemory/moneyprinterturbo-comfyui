"""Pure state helpers for the Streamlit LTX review flow."""


def can_plan_ltx(status: dict) -> bool:
    return bool(status.get("enabled")) and not status.get("errors")


def can_render_ltx(scenes: list[dict]) -> bool:
    return bool(scenes) and all(
        scene.get("status") in {"approved", "complete"} for scene in scenes
    )


def ltx_status(settings: dict, errors: tuple[str, ...]) -> dict:
    return {"enabled": bool(settings.get("enabled")), "errors": list(errors)}
