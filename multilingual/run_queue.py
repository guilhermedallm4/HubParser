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
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
JOBS = json.loads((ROOT / "jobs.json").read_text())
SYNC_EVERY_S = 6 * 3600


def job_name(j):
    return f"{j['encoder']}__{j['head']}__{j['corpus']}"


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


def run_step(module, j, push, machine):
    name = job_name(j)
    (ROOT / "logs").mkdir(exist_ok=True)
    logfile = open(ROOT / "logs" / f"{name}__{module}.log", "a")
    cmd = [sys.executable, "-m", f"hubparser_ml.{module}", "--encoder", j["encoder"], "--head", j["head"], "--corpus", j["corpus"]]
    log(f"start {module} {name}")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT)
    last_sync = time.time()
    while p.poll() is None:
        time.sleep(60)
        if time.time() - last_sync > SYNC_EVERY_S:
            git_sync(f"multilingual: progresso {name} ({machine})", push)
            last_sync = time.time()
    if p.returncode != 0:
        log(f"FAILED {module} {name} (exit {p.returncode}); see logs/{name}__{module}.log")
        git_sync(f"multilingual: progresso {name} ({machine})", push)
        sys.exit(p.returncode)
    git_sync(f"multilingual: {name} {'busca concluída' if module == 'search' else 'treino final concluído'} ({machine})", push)
    log(f"done {module} {name}")


def status(machines):
    for m in machines:
        print(f"\n== {m}")
        for j in JOBS[m]:
            d = ROOT / "results" / job_name(j)
            folds = [json.loads(l) for l in (d / "folds.jsonl").read_text().splitlines()] if (d / "folds.jsonl").exists() else []
            state = "final pronto" if (d / "final.json").exists() else ("busca pronta" if (d / "best.json").exists() else
                    (f"busca {len(folds)}/50 folds" if folds else "pendente"))
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
            if (d / "final.json").exists():
                fin = json.loads((d / "final.json").read_text())["test"]
                extra = " | " + " ".join(f"{l}: LAS {v['las']:.2f}" for l, v in fin["greedy"].items()) + " (gulosa)"
            print(f"  {job_name(j):42s} {state}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=[k for k in JOBS if not k.startswith("_")])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="commit results locally without pushing")
    args = ap.parse_args()
    machines = [args.machine] if args.machine else [k for k in JOBS if not k.startswith("_")]
    if args.status:
        status(machines)
        return
    assert args.machine, "--machine is required to run"
    if not (ROOT / "data" / "built" / "hubparser_multilingual").exists():
        log("building datasets")
        subprocess.run([sys.executable, "build_data.py"], cwd=ROOT, check=True)
    for j in JOBS[args.machine]:
        d = ROOT / "results" / job_name(j)
        if not (d / "best.json").exists():
            run_step("search", j, not args.no_push, args.machine)
        if not (d / "final.json").exists():
            run_step("final_train", j, not args.no_push, args.machine)
    log("queue finished")
    status([args.machine])


if __name__ == "__main__":
    main()
