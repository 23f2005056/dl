"""M4 trainer (E0/E1): M2 batches -> M1 StockNet(MIE, [M2 E1 price GRU], M4 VMD, ATA) -> dev-selected checkpoint -> M3 CSVs."""
import argparse, json, subprocess, sys, time
from pathlib import Path
import torch, yaml
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
for p in ("m1_workspace", "m2_workspace", "m3_workspace"):
    sys.path.insert(0, str(ROOT / p))
from stocknet_mappings import StockNetMappings
from price_reader import PriceReader
from tweet import TweetReader
from sample_builder import StockNetSampleBuilder
from batch_builder import StockNetBatchBuilder
from batch_adapter import prepare_torch_batch
from stocknet import StockNet
from utils import seed_everything, clip_gradients
from evaluate import evaluate
from vmd import VMD
from losses import kl_lambda, stocknet_loss
from predict import write_csv
from e1_integration import E1Integration
from mie import build_glove_matrix

CODE, DATA = ROOT / "stocknet-code", ROOT / "stocknet-dataset"


def make_builder(stocks, batch_size):
    m = StockNetMappings(CODE / "src/config.yml", CODE / "res/vocab.txt")
    sb = StockNetSampleBuilder(m, PriceReader(DATA / "price/preprocessed"),
                               TweetReader(tweet_root=DATA / "tweet/preprocessed",
                                           vocab_path=CODE / "res/vocab.txt"))
    return StockNetBatchBuilder(sb, stocks or list(m.stock_id_dict), batch_size=batch_size), m

def forward(model, b):
    """E0: StockNet as-is (MIE concatenates the 3 raw prices -> [B,5,103]).
    E1: MIE tweet vector c_t [B,5,100] + M2 GRU price encoder [B,5,50] -> [B,5,150] -> VMD -> ATA."""
    if getattr(model, "e1", None) is None:
        return model(b)
    _, c_t = model.mie(b["tweet_ids"], b["price_features"], b["n_words"], b["n_msgs"], b.get("ss_index"))
    x_t = model.e1(c_t, b["price_features"], b["T_lengths"])
    g, kl = model.vmd(x_t, b["T_lengths"], b["y_historical"])
    logits, attn, aux_logits = model.ata(g, b["T_lengths"])
    return logits, kl, aux_logits, attn

def train_step(model, opt, b, step, cfg):
    logits, kl, aux_logits, attn = forward(model, b)
    lam = kl_lambda(step, cfg["kl_lambda_anneal_rate"], cfg["kl_lambda_start_step"])
    loss = stocknet_loss(logits, aux_logits, attn, kl, b["y_historical"], b["targets"],
                         b["T_lengths"], lam, cfg["alpha"])
    opt.zero_grad()
    loss.backward()
    clip_gradients(model, cfg["clip"])
    opt.step()
    return loss.item(), torch.isfinite(kl).all().item()


@torch.no_grad()
def predict(model, builder, split, device):
    model.eval()
    rows = []
    for raw in builder.iter_batches(split):          # dev/test use batch_size=1 (M2, original)
        b = prepare_torch_batch(raw, device)
        logits = forward(model, b)[0]
        for s, d, t, p, pr in zip(b["stock_names"], b["target_dates"], b["targets"].tolist(),
                                  logits.argmax(-1).tolist(), logits.softmax(-1)[:, 1].tolist()):
            rows.append({"stock": s, "target_date": d, "y_true": t, "y_pred": p, "p_rise": pr})
    model.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--stocks", help="comma-separated subset for quick runs")
    ap.add_argument("--overfit", type=int, help="overfit N real samples (smoke test)")
    ap.add_argument("--run-name")
    ap.add_argument("--price-encoder", choices=["raw", "gru"], default="raw",help="raw = E0 (3 raw price features), gru = E1 (M2 GRU price encoder)")
    ap.add_argument("--glove", default=str(CODE / "res" / "glove.twitter.27B.50d.txt"),help="GloVe Twitter 50d file (frozen embeddings, as in the original)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CODE / "src/config.yml", encoding="utf-8"))["model"]
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder, maps = make_builder(args.stocks.split(",") if args.stocks else None,
                                 args.overfit or cfg["batch_size"])
    if not Path(args.glove).is_file():
        raise SystemExit(f"GloVe file not found: {args.glove} (frozen random embeddings would be invalid)")
    glove = build_glove_matrix(maps.vocab_id_dict, args.glove, embed_dim=50)
    x_size = 103 if args.price_encoder == "raw" else 150
    model = StockNet(vmd_module=VMD(x_size=x_size, dropout=cfg["dropout_vmd_in"]),
                     vocab_size=len(maps.vocab_id_dict), pretrained_embeddings=glove,
                     freeze_embeddings=True, daily_att="y")
    if args.price_encoder == "gru":
        model.e1 = E1Integration()      # submodule: trained by Adam, saved in best_dev_model.pt
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    if args.overfit:                                   # master plan 9.3: >= 90% on 5 real samples
        b = prepare_torch_batch(next(builder.iter_batches("train")), device)
        for step in range(1, 301):
            loss, ok = train_step(model, opt, b, step, cfg)
            assert ok and loss == loss, "non-finite loss/KL"
        model.eval()
        with torch.no_grad():
            acc = (forward(model, b)[0].argmax(-1) == b["targets"]).float().mean().item()
        print(f"overfit: loss {loss:.4f}, train acc {acc:.2f}")
        return

    run = ROOT / "experiments" / (args.run_name or time.strftime(f"{args.price_encoder}-%Y%m%d-%H%M%S"))
    run.mkdir(parents=True)
    tb = SummaryWriter(run / "tb")
    best, step = None, 0
    for epoch in range(1, (args.epochs or cfg["n_epochs"]) + 1):
        t0 = time.time()
        for raw in builder.iter_batches("train"):
            step += 1
            loss, _ = train_step(model, opt, prepare_torch_batch(raw, device), step, cfg)
            tb.add_scalar("train/loss", loss, step)
        dev = evaluate([{**r, "split": "dev"} for r in predict(model, builder, "dev", device)], "dev")
        tb.add_scalar("dev/acc", dev["accuracy"], epoch)
        tb.add_scalar("dev/mcc", dev["mcc"] or 0.0, epoch)
        print(f"epoch {epoch} ({time.time() - t0:.0f}s) loss {loss:.4f} "
              f"dev acc {dev['accuracy']:.4f} mcc {dev['mcc']}")
        key = (dev["accuracy"], dev["mcc"] if dev["mcc"] is not None else -1.0)
        if best is None or key > best["key"]:          # select on dev accuracy, then MCC
            best = {"key": key, "epoch": epoch, "dev": dev}
            torch.save(model.state_dict(), run / "best_dev_model.pt")

    model.load_state_dict(torch.load(run / "best_dev_model.pt", map_location=device))
    ckpt = f"{run.name}/best_dev_model.pt@epoch{best['epoch']}"
    for split in ("dev", "test"):
        write_csv(run / f"{split}.csv", split, predict(model, builder, split, device), ckpt)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (run / "summary.json").write_text(json.dumps(
        {"seed": args.seed, "price_encoder": args.price_encoder,"git": git, "config": cfg, "best_epoch": best["epoch"],
         "best_dev": best["dev"]}, indent=2))
    print(f"best epoch {best['epoch']}; CSVs in {run}")


if __name__ == "__main__":
    main()