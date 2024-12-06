import pathlib


def get_fixture_path(filename: str, integration: str | None = None) -> pathlib.Path:
    """Get path of fixture."""
    if integration is None and "/" in filename and not filename.startswith("helpers/"):
        integration, filename = filename.split("/", 1)

    if integration is None:
        return pathlib.Path(__file__).parent.joinpath("fixtures", filename)

    return pathlib.Path(__file__).parent.joinpath(
        "components", integration, "fixtures", filename
    )
