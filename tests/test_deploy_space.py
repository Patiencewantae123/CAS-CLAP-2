"""scripts/deploy_space.py logic with a fake Hugging Face API (no network, no token)."""

from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from scripts import deploy_space


def test_space_id_accepts_id_or_url():
    assert deploy_space.space_id("patwan/cas-v-board") == "patwan/cas-v-board"
    assert deploy_space.space_id(" https://huggingface.co/spaces/patwan/cas-v-board/ ") == "patwan/cas-v-board"
    with pytest.raises(SystemExit):
        deploy_space.space_id("cas-v-board")


def test_token_prompt_refuses_partial_invalid_and_read_only_tokens(monkeypatch, capsys):
    typed = iter(["HFAKg6iSZUWRjhbsImy0K8kBBMuayn0", "hf_deleted", "hf_readonly", "hf_good"])
    users = {
        "hf_readonly": {"name": "patwan", "auth": {"accessToken": {"role": "read"}}},
        "hf_good": {"name": "patwan", "auth": {"accessToken": {"role": "write"}}},
    }
    checked, saved = [], []

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            checked.append(self.token)
            if self.token not in users:
                request = httpx.Request("GET", "https://huggingface.co/api/whoami-v2")
                raise HfHubHTTPError("401 Unauthorized", response=httpx.Response(401, request=request))
            return users[self.token]

    monkeypatch.setattr(deploy_space.getpass, "getpass", lambda prompt: next(typed))
    monkeypatch.setattr(deploy_space, "HfApi", FakeApi)
    monkeypatch.setattr(deploy_space, "login", lambda token, add_to_git_credential: saved.append(token))

    assert deploy_space.ask_for_token(attempts=4)["name"] == "patwan"
    assert saved == ["hf_good"]
    assert checked == ["hf_deleted", "hf_readonly", "hf_good"]  # the partial token never reached the network
    out = capsys.readouterr().out
    assert "not a whole token" in out and "does not accept" in out and "only read" in out


def test_token_prompt_gives_up_after_the_attempts(monkeypatch):
    monkeypatch.setattr(deploy_space.getpass, "getpass", lambda prompt: "not-a-token")
    with pytest.raises(SystemExit, match="nothing was uploaded"):
        deploy_space.ask_for_token(attempts=2)


def test_cannot_deploy_to_someone_elses_space():
    user = {"name": "patwan", "orgs": [{"name": "my-lab"}]}
    with pytest.raises(SystemExit, match="cannot upload"):
        deploy_space.check_space(api=None, repo_id="someone/space", user=user)


class FakeRuntimeApi:
    """Reports the given stages in turn, then repeats the last one."""

    def __init__(self, stages, log=("line 1", "ModuleNotFoundError: No module named 'x'")):
        self.stages, self.log = list(stages), log

    def get_space_runtime(self, repo_id):
        stage = self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]
        return SimpleNamespace(stage=stage, raw={"errorMessage": "Exit code 1"} if "ERROR" in stage else {})

    def fetch_space_logs(self, repo_id, build=False):
        return iter(self.log)


def test_follow_waits_past_the_old_status_until_running(monkeypatch, capsys):
    monkeypatch.setattr(deploy_space.time, "sleep", lambda seconds: None)
    api = FakeRuntimeApi(["RUNNING", "RUNNING_BUILDING", "RUNNING_APP_STARTING", "RUNNING"])  # a redeploy
    assert deploy_space.follow(api, "patwan/cas-v-board", minutes=1) == 0
    out = capsys.readouterr().out
    assert "Building the new version" in out and "live at https://huggingface.co/spaces/patwan/cas-v-board" in out


def test_follow_prints_the_log_when_the_build_fails(monkeypatch, capsys):
    monkeypatch.setattr(deploy_space.time, "sleep", lambda seconds: None)
    api = FakeRuntimeApi(["NO_APP_FILE", "BUILDING", "BUILD_ERROR"])
    assert deploy_space.follow(api, "patwan/cas-v-board", minutes=1) == 1
    out = capsys.readouterr().out
    assert "BUILD_ERROR: Exit code 1" in out and "No module named 'x'" in out and "not your token" in out
