# =============================================================================
# Put the CAS-V board online as a Hugging Face Space, in one command:
#   1. log in if needed: asks once for a token (hidden), checks that it is
#      complete, valid and allowed to write, and saves it on this PC
#   2. check the Space exists, you can write to it, and it runs on ZeroGPU
#   3. build the bundle (scripts/build_space.py) and upload it, replacing the
#      Space's files (.gitattributes is always kept)
#   4. follow the Space until it runs, or print the end of its error log
#
# The Space must already exist: https://huggingface.co/new-space with SDK
# Gradio and hardware ZeroGPU.
#
# Usage:
#   py -m scripts.deploy_space patwan/cas-v-board
# =============================================================================

import argparse
import getpass
import os
import time

from huggingface_hub import HfApi, login
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError, RepositoryNotFoundError

from casv.log import setup_logging
from scripts.build_space import DEFAULT_OUT, MARKER, build

HUB = "https://huggingface.co"
TOKENS_URL = f"{HUB}/settings/tokens"
NEW_SPACE_URL = f"{HUB}/new-space"
PROGRESS = {
    "BUILDING": "Building: installing PyTorch and the other packages (5-10 min the first time)",
    "RUNNING_BUILDING": "Building the new version (the old one keeps running meanwhile)",
    "APP_STARTING": "Starting the app: loading CLIP, the vocabularies and AudioLDM",
    "RUNNING_APP_STARTING": "Starting the new version",
}
FAILED = {"BUILD_ERROR": True, "CONFIG_ERROR": True, "RUNTIME_ERROR": False}  # stage -> show the build log?
SETTLE_SECONDS = 120  # right after an upload the Space can still report its previous status


def space_id(text: str) -> str:
    """'patwan/cas-v-board' from the id or the Space's URL."""
    repo_id = text.strip().removeprefix(f"{HUB}/spaces/").strip("/")
    if repo_id.count("/") != 1:
        raise SystemExit(f"Give the Space as <user>/<name>, e.g. patwan/cas-v-board (got '{text}').")
    return repo_id


def token_role(info: dict) -> str:
    return info.get("auth", {}).get("accessToken", {}).get("role", "unknown")


def ask_for_token(attempts: int = 3) -> dict:
    """Ask for a token until one is complete, valid and can write; save it. Returns whoami()."""
    print(
        "\nLog in to Hugging Face (only once: the token is saved on this PC).\n"
        f"  1. Open {TOKENS_URL} and click 'Create new token'.\n"
        "  2. Choose 'Write', type any name, click 'Create token', then click 'Copy'.\n"
        "  3. Paste it here (right-click or Ctrl+V). It stays invisible; press Enter.\n"
        "Never paste a token into a chat or a file.\n"
    )
    for _ in range(attempts):
        token = getpass.getpass("Token: ").strip()
        if not token.startswith("hf_"):
            print(
                "  That is not a whole token: tokens start with hf_ and are about 37 characters. Use the Copy button."
            )
            continue
        try:
            info = HfApi(token=token).whoami()
        except HfHubHTTPError:
            print("  Hugging Face does not accept that token (deleted, or not copied in full). Try again.")
            continue
        if token_role(info) == "read":
            print("  That token can only read. Create one with 'Write' access.")
            continue
        login(token=token, add_to_git_credential=False)
        return info
    raise SystemExit("Not logged in; nothing was uploaded. Run the command again when you have the token.")


def current_user(api: HfApi) -> dict:
    """whoami() of the saved login, asking for a token when there is none."""
    try:
        return api.whoami()
    except LocalTokenNotFoundError:
        return ask_for_token()
    except HfHubHTTPError:
        if os.environ.get("HF_TOKEN"):
            raise SystemExit(
                "The HF_TOKEN environment variable holds an invalid token; remove it and try again."
            ) from None
        print("The saved Hugging Face login no longer works (was the token deleted?).")
        return ask_for_token()


def check_space(api: HfApi, repo_id: str, user: dict) -> None:
    """The Space exists, this login may write to it, and it runs on ZeroGPU (warn otherwise)."""
    owner = repo_id.split("/")[0]
    allowed = {user["name"], *(org["name"] for org in user.get("orgs", []))}
    if owner not in allowed:
        raise SystemExit(f"You are logged in as {user['name']}, which cannot upload to {owner}'s Spaces.")
    try:
        api.space_info(repo_id)
    except RepositoryNotFoundError:
        raise SystemExit(
            f"There is no Space {repo_id}. Create it at {NEW_SPACE_URL} (SDK: Gradio, hardware: ZeroGPU), "
            "then run this again."
        ) from None
    runtime = api.get_space_runtime(repo_id)
    hardware = str(runtime.requested_hardware or runtime.hardware or "")
    if hardware and "zero" not in hardware.lower():
        print(
            f"Warning: the Space's hardware is '{hardware}', not ZeroGPU, so audio would be generated on CPU "
            f"(minutes per clip). Change it at {HUB}/spaces/{repo_id}/settings (Space hardware)."
        )


def show_log(api: HfApi, repo_id: str, build_log: bool, lines: int = 40) -> None:
    try:
        log_lines = [line.rstrip() for line in api.fetch_space_logs(repo_id, build=build_log)]
    except HfHubHTTPError as error:
        print(f"(Could not fetch the log: {error})")
        return
    print(f"\nLast {min(lines, len(log_lines))} lines of the {'build' if build_log else 'app'} log:")
    for line in log_lines[-lines:]:
        print(f"  {line}")


def follow(api: HfApi, repo_id: str, minutes: int) -> int:
    """Print the Space's status until it runs (0) or fails (1)."""
    url = f"{HUB}/spaces/{repo_id}"
    start, last, moved = time.time(), None, False
    print(f"\nFollowing the build (Ctrl+C stops watching; the Space keeps building). Your Space: {url}")
    try:
        while time.time() - start < minutes * 60:
            try:
                runtime = api.get_space_runtime(repo_id)
            except HfHubHTTPError:
                time.sleep(10)
                continue
            stage = str(getattr(runtime.stage, "value", runtime.stage))
            moved = moved or stage in PROGRESS
            settled = moved or time.time() - start > SETTLE_SECONDS
            if stage != last:
                print(f"  [{(time.time() - start) / 60:4.1f} min] {PROGRESS.get(stage, stage)}")
                last = stage
            if settled and stage == "RUNNING":
                print(f"\nDone: the board is live at {url}\nAnyone with that link can use it.")
                return 0
            if settled and stage in FAILED:
                message = (runtime.raw or {}).get("errorMessage")
                print(f"\nThe Space stopped with {stage}" + (f": {message}" if message else "."))
                show_log(api, repo_id, build_log=FAILED[stage])
                print("\nSend these lines to Claude (not your token) to get it fixed.")
                return 1
            time.sleep(15)
    except KeyboardInterrupt:
        print(f"\nStopped watching. Check the Space at {url}")
        return 0
    print(f"\nStill not running after {minutes} min; it may need longer. Check {url}")
    return 0


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Build, upload and follow the CAS-V board on a Hugging Face Space.")
    parser.add_argument("space", help="the Space, e.g. patwan/cas-v-board (or its URL)")
    parser.add_argument("--wait", type=int, default=20, help="minutes to follow the build afterwards (0: do not)")
    args = parser.parse_args()
    repo_id = space_id(args.space)

    user = current_user(HfApi())
    api = HfApi()  # with the saved token
    print(f"Logged in to Hugging Face as {user['name']}.")
    check_space(api, repo_id, user)

    files = build(DEFAULT_OUT)
    size_mb = sum(path.stat().st_size for path in files) / 2**20
    print(f"\nUploading {len(files)} files ({size_mb:.0f} MB) to {repo_id}; this replaces the Space's files...")
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=DEFAULT_OUT,
            commit_message="Deploy the CAS-V comparison board",
            commit_description=(DEFAULT_OUT / MARKER).read_text(encoding="utf-8"),
            delete_patterns="*",
        )
    except HfHubHTTPError as error:
        raise SystemExit(f"Upload failed: {error}") from None
    print("Uploaded.")
    if args.wait > 0:
        raise SystemExit(follow(api, repo_id, args.wait))


if __name__ == "__main__":
    main()
