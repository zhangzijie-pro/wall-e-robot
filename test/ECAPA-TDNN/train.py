def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for anchor, positive, negative in dataloader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)

        loss = criterion(emb_anchor, emb_positive, emb_negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

