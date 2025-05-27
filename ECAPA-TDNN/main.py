from ecapa import *
from loss import TripletLoss
from custom_dataset import *
from train import train
import torch.optim as optim
from turn import export_Model

voice_path = r"Dataset/voice"  # 修改为你的数据目录
dataset = TripDataSet(voice_path)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)
model = ECAPA_TDNN().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = TripletLoss(margin=0.3)

# for epoch in range(1):
#     loss = train(model, dataloader, optimizer, criterion, device)
#     print(f"Epoch {epoch}, Loss: {loss:.4f}")
#     torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

# turn to onnx
dummy_input = torch.zeros(2, 200, 80)
export_Model(model,"model_epoch_0.pt").onnx(dummy_input, "mfcc", "feat_dim")
