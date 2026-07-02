import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset, Subset

from PNEA_MIL import PNEA_MIL


class SlideDataset(Dataset):
    """
    Minimal dataset for PNEA-MIL.

    The CSV file must have two columns:
        file,label

    Example:
        slide_001.pt,0
        slide_002.pt,1

    Each .pt file stores one slide's patch features:
        x.shape = [num_patches, feature_dim]

    Training:
        randomly sample a fixed number of patches from each slide.

    Testing:
        use all patches from each slide.
    """

    def __init__(self, csv_file,num_patches=None):
        self.df = pd.read_csv(csv_file)
        self.num_patches = num_patches

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = torch.load( row["file"], map_location="cpu", weights_only=True)
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.tensor(int(row["label"]), dtype=torch.long)

        if self.num_patches is not None:
            n = x.shape[0]
            idx = torch.randint(low=0, high=n, size=(self.num_patches,))
            x = x[idx]

        return x, y




def train_one_epoch(model, loader, device):
    model.train()

    total_loss = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        model.optim.zero_grad()

        loss = model.return_loss(x, y)

        loss.backward()
        model.optim.step()

        total_loss += loss.item() * y.shape[0]
        total_n += y.shape[0]

    return total_loss / total_n



@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        loss = model.return_loss(x, y)

        total_loss += loss.item() * y.shape[0]
        total_n += y.shape[0]

    return total_loss / total_n


    

@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    for x, y in loader:
        # Test loader uses batch_size = 1, so x can contain all patches
        # from one slide without needing padding.
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

        if prob.shape[1] == 2:
            y_score.extend(prob[:, 1].cpu().numpy())
        else:
            y_score.extend(prob.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    auc = roc_auc_score(y_true, y_score) 
    ba = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return auc, ba, mcc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", type = str, default = "")
    parser.add_argument("--output_model_file", default="./model/model")
    parser.add_argument("--percent_validation", type=float, default=0.2)

    parser.add_argument("--D_in", type=int, default=1024)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--C", type=int, default=2)

    parser.add_argument("--num_train_patches", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_threshold", type=float, default=0.0005)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_l1", type=float, default=5.0)
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args(
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


    # Training data is split into training set and validation set for early stop purposes.

    train_full_dataset = SlideDataset(
        csv_file=args.train_csv,
        num_patches=args.num_train_patches
    )
    
    val_full_dataset = SlideDataset(
        csv_file=args.train_csv,
        num_patches=None
    )


    train_idx, val_idx = train_test_split(
        np.arange(len(train_full_dataset)),
        test_size=args.percent_validation,
        random_state=args.random_seed,
    )

    train_dataset = Subset(train_full_dataset, train_idx)
    val_dataset = Subset(val_full_dataset, val_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )


    if args.test_csv != "":
        test_dataset = SlideDataset(
            csv_file=args.test_csv,
            num_patches=None,
        )
    
    
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
        )

    model = PNEA_MIL(
        D_in=args.D_in,
        D=args.D,
        C=args.C,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_l1=args.lambda_l1,
        seed = args.random_seed
    ).to(device)


    best_loss = 1e20
    best_state = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
        )

        validation_loss = evaluate_loss(
            model=model,
            loader=train_loader,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Training loss={train_loss:.6f} | "
            f"Validation loss={validation_loss:.6f}"
        )

        if validation_loss < best_loss - np.abs(best_loss) * args.early_stop_threshold:
            best_loss = validation_loss
            patience_count = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_loss": validation_loss,
                "args": vars(args),
            }
        else:
            patience_count += 1

        if patience_count >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    output_path = Path(args.output_model_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)

    print(f"Saved best model to {output_path}")

 
    if args.test_csv != "":
        
        model.load_state_dict( torch.load(args.output_model_file, weights_only=True) ["model_state_dict"] )
        auc, ba, mcc = evaluate_metrics(model, test_loader, device)
    
        print(f"Performance on Test Data: ROC AUC={auc:.4f}| BA={auc:.4f} | MCC={auc:.4f}")
        

if __name__ == "__main__":
    main()
