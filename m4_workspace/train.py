import argparse, json, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from batch import collate, to_device, target_label
from predict import write_csv

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "stocknet-code/src/config.yml"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)["logits"]
        prob = logits.softmax(-1)[:, 1].tolist()
        pred = logits.argmax(-1).tolist()
        true = target_label(batch).tolist()
        for s, d, t, p, pr in zip(batch["stock_symbol"], batch["target_date"], true, pred, prob):
            rows.append({"stock": s, "target_date": d, "y_true": t, "y_pred": p, "p_rise": pr})
    model.train()
    tp = sum(r["y_true"] == 1 and r["y_pred"] == 1 for r in rows)
    tn = sum(r["y_true"] == 0 and r["y_pred"] == 0 for r in rows)
    fp = sum(r["y_true"] == 0 and r["y_pred"] == 1 for r in rows)
    fn = sum(r["y_true"] == 1 and r["y_pred"] == 0 for r in rows)
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {"n": len(rows), "acc": (tp + tn) / len(rows),
            "mcc": (tp * tn - fp * fn) / den if den else None}, rows


def score(m):  # dev selection key: MCC first, then accuracy
    return (m["mcc"] if m["mcc"] is not None else -1.0, m["acc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--run-name")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))["model"]
    epochs = args.epochs or cfg["n_epochs"]
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.fake:
        from fake_data import FakeDataset
        from dummy_model import DummyModel
        train_ds, dev_ds, test_ds = (FakeDataset(2000, 1, "train"),
                                     FakeDataset(400, 2, "dev"), FakeDataset(400, 3, "test"))
        model = DummyModel()
    else:
        raise SystemExit("Real data/model not plugged in yet (waiting on M2 DataPipe + M1 model).")

    model.to(device)
    run_dir = HERE.parent / "experiments" / (args.run_name or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True)
    writer = SummaryWriter(run_dir / "tb")
    bs = cfg["batch_size"]
    train_dl = DataLoader(train_ds, bs, shuffle=bool(cfg["shuffle"]), collate_fn=collate)
    dev_dl = DataLoader(dev_ds, bs, collate_fn=collate)
    test_dl = DataLoader(test_ds, bs, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best, step = None, 0
    def check_dev():
        nonlocal best
        m, _ = evaluate(model, dev_dl, device)
        writer.add_scalar("dev/acc", m["acc"], step)
        if m["mcc"] is not None:
            writer.add_scalar("dev/mcc", m["mcc"], step)
        print(f"step {step} dev {m}")
        if best is None or score(m) > score(best["dev"]):
            best = {"step": step, "dev": m}
            torch.save(model.state_dict(), run_dir / "best.pt")

    for epoch in range(epochs):
        for batch in train_dl:
            batch = to_device(batch, device)
            out = model(batch, step)
            loss = out["loss"] if "loss" in out else F.cross_entropy(out["logits"], target_label(batch))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            step += 1
            writer.add_scalar("train/loss", loss.item(), step)
            if step % args.eval_every == 0:
                check_dev()
        print(f"epoch {epoch} done, last loss {loss.item():.4f}")
    check_dev()

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    ckpt = f"{run_dir.name}/best.pt@{best['step']}"
    dev_m, dev_rows = evaluate(model, dev_dl, device)
    test_m, test_rows = evaluate(model, test_dl, device)
    write_csv(run_dir / "dev.csv", "dev", dev_rows, ckpt)
    write_csv(run_dir / "test.csv", "test", test_rows, ckpt)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"seed": args.seed, "fake": args.fake, "config": cfg, "git": git,
               "best_step": best["step"], "dev": dev_m, "test": test_m}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("best_step", "dev", "test")}, indent=2))


if __name__ == "__main__":
    main()