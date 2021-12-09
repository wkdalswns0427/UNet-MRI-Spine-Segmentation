import torch
from tqdm import tqdm
from loss_function import one_hot_nd
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        optimizer.zero_grad()
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward -- float16
        with torch.cuda.amp.autocast():
            predictions = model(data)
            with torch.no_grad():
                targets = one_hot_nd(targets, predictions.size(-3), 2).to(predictions)
            loss = loss_fn(predictions, targets)

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

# not using scaler for memory issues
def train_fn_no_scale(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        optimizer.zero_grad()
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward -- float32
        predictions = model(data)
        with torch.no_grad():
            targets = one_hot_nd(targets, predictions.size(-3), 2).to(predictions)

        loss = loss_fn(predictions, targets)

        # backward
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
