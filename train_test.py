import torch
import torch.optim as optim


def train_encoder_model(model, train_dataloader, dev_dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    model_accuracy: float = 0.0
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            if i % 100 == 0:
                print(f"Batch {i}/{len(train_dataloader)}")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)
            labels = batch["labels"].to(device)
            segment_embedding = batch["segment_embedding"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, segment_embedding, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        accuracy: float = evaluate_model(model, dev_dataloader, device)
        if accuracy > model_accuracy:
            model_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
    print(f"Best Accuracy: {model_accuracy:.4f}")


def evaluate_model(model, data_loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for i, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        # 预测标签
        _, predicted = torch.max(outputs, dim=2)
        predicted = predicted[attention_mask.bool().squeeze(1).squeeze(1)].cpu().numpy()
        labels = labels[attention_mask.bool().squeeze(1).squeeze(1)].cpu().numpy()
        # 计算准确率
        correct += (predicted == labels).sum().item()
        total += labels.shape[0]
        print(f"Batch {i + 1}/{len(data_loader)}")
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    model.train()


class Trainer:
    def __init__(self, model, train_loader, dev_loader, test_loader, 
                device="cuda", num_epochs=50, lr=1e-5):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr

    def run(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        train_encoder_model(self.model, self.train_loader, self.dev_loader, criterion, optimizer, self.num_epochs, self.device)
