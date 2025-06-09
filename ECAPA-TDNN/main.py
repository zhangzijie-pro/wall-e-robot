from tqdm import tqdm
from ecapa import *
from loss import AAMSoftmaxLoss
from custom_dataset import *
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

voice_path = Path("TRAIN")       
output_dir = Path("output")

processor = DatasetProcessor(voice_path, output_dir)
processor.process_dataset()

dataset = SpeakerDataset(output_dir, output_dir / "speaker_to_id.json")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = ECAPA_TDNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

embedding_dim = 192
num_speakers = len(dataset.speaker_to_id)
loss_fn = AAMSoftmaxLoss(embedding_dim=embedding_dim, num_classes=num_speakers).to(device)

# Step 4: 开始训练
num_epochs = 30
model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in loop:
        features, labels = batch  # features: [B, T, F], labels: [B]
        
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(features)  # [B, 192]
        loss = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()

        # 统计训练准确率
        with torch.no_grad():
            norm_embeddings = nn.functional.normalize(embeddings)
            logits = torch.matmul(norm_embeddings, loss_fn.weight.T)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total * 100
    print(f"Epoch {epoch+1} Finished | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # 保存模型
torch.save(model.state_dict(), f"ecapa_tdnn.pt")