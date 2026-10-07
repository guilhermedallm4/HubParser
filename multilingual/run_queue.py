"""Runs one machine's job queue (jobs.json): for each job, the Optuna search and then
the final training. Finished steps are skipped, so the script can simply be started
again after any interruption. Results (folds.jsonl, best.json, final.json) are
committed and pushed to git every few hours and at the end of each step.

Usage:
  python run_queue.py --machine maquina_beto          # run (use nohup, see INSTRUCOES_CLAUDE.md)
  python run_queue.py --machine maquina_beto --status # progress report, no training
  python run_queue.py --status                        # progress of every job in jobs.json
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
JOBS = json.loads((ROOT / "jobs.json").read_text())


def reload_jobs(push):
    """Pick up queue changes pushed from the other machine before starting each job."""
    global JOBS
    if push:
        subprocess.run(["git", "pull", "-q", "--rebase", "--autostash"], cwd=REPO, check=False)
    JOBS = json.loads((ROOT / "jobs.json").read_text())
SYNC_EVERY_S = 6 * 3600
STALL_S = 45 * 60      # a step whose log does not grow for this long is considered hung
MAX_RESTARTS = 3       # restarts after a hang or an error exit, per step
RETRY_WAIT_S = 5 * 60
POOL = "compartilhado"  # jobs taken by whichever machine finishes its own queue first
CONFIGS = json.loads((ROOT / "data" / "search_configs.json").read_text())["configs"]


def config_index(hp):
    for i, c in enumerate(CONFIGS):
        if all(c[k] == hp.get(k) for k in ("learning_rate", "weight_decay", "warmup_ratio")):
            return i
    return None


def job_name(j):
    return f"{j['encoder']}__{j['head']}__{j['corpus']}"


def search_complete(j):
    """All 10 configurations x 5 folds done (best.json written by the fixed-list search)."""
    f = ROOT / "results" / job_name(j) / "best.json"
    return f.exists() and json.loads(f.read_text()).get("n_configs") == 10


def final_up_to_date(j):
    """final.json exists and was trained with the current best configuration."""
    d = ROOT / "results" / job_name(j)
    if not (d / "final.json").exists() or not (d / "best.json").exists():
        return False
    best = json.loads((d / "best.json").read_text())["best"]["hyperparameters"]
    used = json.loads((d / "final.json").read_text())["hyperparameters"]
    keys = ("learning_rate", "weight_decay", "warmup_ratio", "num_train_epochs")
    return all(best[k] == used[k] for k in keys)


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def git_sync(message, push=True):
    paths = ["multilingual/results"]
    subprocess.run(["git", "add", *paths], cwd=REPO, check=False)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
        return
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=REPO, check=False)
    if push:
        subprocess.run(["git", "pull", "-q", "--rebase", "--autostash"], cwd=REPO, check=False)
        r = subprocess.run(["git", "push", "-q"], cwd=REPO, check=False)
        log("git push ok" if r.returncode == 0 else "git push FAILED (results stay committed locally)")


def claim_owner(j):
    f = ROOT / "results" / job_name(j) / "CLAIM"
    return f.read_text().strip() if f.exists() else None


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).returncode


def try_claim(j, machine, push):
    """Reserve a shared-pool job by committing results/<job>/CLAIM and pushing it.
    If the other machine pushed a claim for the same job first, the rebase conflicts on
    CLAIM: our claim is dropped and the job stays with the other machine."""
    owner = claim_owner(j)
    if owner:
        return owner == machine
    f = ROOT / "results" / job_name(j) / "CLAIM"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(machine + "\n")
    if not push:
        return True
    rel = str(f.relative_to(REPO))
    git("add", rel)
    git("commit", "-q", "-m", f"multilingual: {machine} reserva {job_name(j)}")
    for _ in range(5):
        if git("push", "-q") == 0:
            return True
        if git("pull", "-q", "--rebase") != 0:          # conflict: the other machine claimed it
            git("rebase", "--abort")
            git("reset", "-q", "--keep", "HEAD~1")       # drop our claim commit
            git("pull", "-q", "--rebase", "--autostash")
            return claim_owner(j) == machine
        if claim_owner(j) != machine:                    # defensive: claim replaced upstream
            return False
        time.sleep(5)
    log(f"could not push the claim for {job_name(j)}; will retry later")
    git("reset", "-q", "--keep", "HEAD~1")
    f.unlink(missing_ok=True)
    return False


def next_pool_job(machine, push):
    for j in JOBS.get(POOL, []):
        if search_complete(j) and final_up_to_date(j):
            continue
        owner = claim_owner(j)
        if owner == machine:
            return j
        if owner is None and try_claim(j, machine, push):
            log(f"claimed {job_name(j)} from the shared pool")
            return j
    return None


def gpu_ok():
    r = subprocess.run([sys.executable, "-c", "import torch; x = torch.ones(1024, 1024, device='cuda'); "
                        "torch.cuda.synchronize(); print(float((x @ x).sum()))"], capture_output=True, text=True)
    return r.returncode == 0


def run_step(module, j, push, machine):
    """Run one step as a subprocess. If its log stops growing for STALL_S (a hang, e.g. a
    DataLoader deadlock), kill the whole process group and run the step again; if it exits
    with an error (e.g. a transient CUDA error), wait RETRY_WAIT_S and run it again. The
    search resumes from the last finished fold. Gives up after MAX_RESTARTS restarts."""
    name = job_name(j)
    (ROOT / "logs").mkdir(exist_ok=True)
    logpath = ROOT / "logs" / f"{name}__{module}.log"
    cmd = [sys.executable, "-m", f"hubparser_ml.{module}", "--encoder", j["encoder"], "--head", j["head"], "--corpus", j["corpus"]]
    for attempt in range(1, MAX_RESTARTS + 2):
        log(f"start {module} {name}" + (f" (attempt {attempt})" if attempt > 1 else ""))
        logfile = open(logpath, "a")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        last_sync = last_growth = time.time()
        last_size = logpath.stat().st_size
        stalled = False
        while p.poll() is None:
            time.sleep(60)
            size = logpath.stat().st_size
            if size != last_size:
                last_size, last_growth = size, time.time()
            elif time.time() - last_growth > STALL_S:
                stalled = True
                log(f"STALLED {module} {name}: log unchanged for {STALL_S // 60} min; killing and restarting")
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
                break
            if time.time() - last_sync > SYNC_EVERY_S:
                git_sync(f"multilingual: progresso {name} ({machine})", push)
                last_sync = time.time()
        logfile.close()
        shutil.rmtree(ROOT / "results" / name / "tmp_run", ignore_errors=True)
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        if stalled:
            continue
        if p.returncode != 0:
            # transient GPU/driver errors (e.g. "CUDA error: the launch timed out") end the
            # process with an error; wait, check the GPU and run the step again
            log(f"FAILED {module} {name} (exit {p.returncode}); see logs/{name}__{module}.log")
            git_sync(f"multilingual: progresso {name} ({machine})", push)
            if attempt <= MAX_RESTARTS:
                time.sleep(RETRY_WAIT_S)
                log(f"GPU check before retrying: {'ok' if gpu_ok() else 'FAILED'}")
                continue
            sys.exit(p.returncode)
        git_sync(f"multilingual: {name} {'busca concluída' if module == 'search' else 'treino final concluído'} ({machine})", push)
        log(f"done {module} {name}")
        return
    log(f"FAILED {module} {name}: failed or stalled {MAX_RESTARTS + 1} times in a row")
    git_sync(f"multilingual: progresso {name} ({machine})", push)
    sys.exit(1)


def status(machines):
    for m in machines:
        print(f"\n== {m}")
        for j in JOBS[m]:
            d = ROOT / "results" / job_name(j)
            raw = [json.loads(l) for l in (d / "folds.jsonl").read_text().splitlines()] if (d / "folds.jsonl").exists() else []
            uniq = {}
            for f in raw:
                ci = config_index(f["hyperparameters"])
                if ci is not None:
                    uniq.setdefault((ci, f["fold"]), dict(f, trial=ci))
            folds = list(uniq.values())
            if search_complete(j) and final_up_to_date(j):
                state = "final pronto"
            elif search_complete(j):
                state = "busca pronta"
            else:
                state = f"busca {len(folds)}/50 folds" if folds else "pendente"
            mins = [f["minutes"] for f in folds]
            extra = ""
            if folds:
                by_trial = {}
                for f in folds:
                    by_trial.setdefault(f["trial"], []).append(f["las"])
                full = {t: sum(v) / len(v) for t, v in by_trial.items() if len(v) == 5}
                best = f" melhor LAS médio {max(full.values()):.4f}" if full else ""
                eta = (50 - len(folds)) * (sum(mins) / len(mins)) / 60
                extra = f" | {sum(mins) / len(mins):.0f} min/fold, ~{eta:.0f} h restantes na busca{best}"
            if state == "final pronto":
                fin = json.loads((d / "final.json").read_text())["test"]
                extra = " | " + " ".join(f"{l}: LAS {v['las']:.2f}" for l, v in fin["greedy"].items()) + " (gulosa)"
            owner = f" [{claim_owner(j)}]" if m == POOL and claim_owner(j) else (" [livre]" if m == POOL else "")
            print(f"  {job_name(j):42s} {state}{owner}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=[k for k in JOBS if not k.startswith("_") and k != POOL])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="commit results locally without pushing")
    args = ap.parse_args()
    machines = [args.machine] if args.machine else [k for k in JOBS if not k.startswith("_")]
    if args.machine and POOL in JOBS:
        machines.append(POOL)
    if args.status:
        status(machines)
        return
    assert args.machine, "--machine is required to run"
    if not (ROOT / "data" / "built" / "hubparser_multilingual").exists():
        log("building datasets")
        subprocess.run([sys.executable, "build_data.py"], cwd=ROOT, check=True)
    while True:
        reload_jobs(not args.no_push)
        pending = [j for j in JOBS[args.machine] if not search_complete(j) or not final_up_to_date(j)]
        j = pending[0] if pending else next_pool_job(args.machine, not args.no_push)
        if j is None:
            break
        if not search_complete(j):
            run_step("search", j, not args.no_push, args.machine)
        if not final_up_to_date(j):
            run_step("final_train", j, not args.no_push, args.machine)
    log("queue finished")
    status([args.machine, POOL] if POOL in JOBS else [args.machine])


if __name__ == "__main__":
    main()
