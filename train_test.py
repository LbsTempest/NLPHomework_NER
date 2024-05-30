import torch
import torch.optim as optim
from torch.nn import DataParallel


def train_encoder_model(model, train_dataloader, dev_dataloader, criterion, optimizer, num_epochs, device):
    """
    Train an encoder-only model.

    Args:
        model (torch.nn.Module): The encoder model to be trained.
        train_dataloader (DataLoader): Dataloader for the training dataset.
        dev_dataloader (DataLoader): Dataloader for the development/validation dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the model on (e.g., "cuda" or "cpu").

    Returns:
        None
    """
    model.train()
    model_accuracy: float = 0.0
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)
            labels = batch["labels"].to(device)
            segment_embedding = batch["segment_embedding"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, segment_embedding, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Batch {i}/{len(train_dataloader)} Loss: {loss.item()}")
            with open("encoder_loss.txt", "a") as f:
                f.write(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_dataloader)}, Loss: {loss.item()}\n")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        accuracy: float = evaluate_encoder_model(model, dev_dataloader, device)
        if accuracy > model_accuracy:
            model_accuracy = accuracy
            torch.save(model.state_dict(), f"Epoch_{epoch}_best_encoder_model.pth")
        print(f"Best Accuracy: {model_accuracy:.4f}")
        with open("encoder_loss.txt", "a") as f:
            f.write(f"Epoch:{epoch+1}/{num_epochs}, Loss: {loss.item()}\n")
            f.write(f"Best Accuracy: {model_accuracy:.4f}\n")


def train_decoder_model(model, train_dataloader, dev_dataloader, criterion, optimizer, num_epochs, device):
    """
    Train a decoder-only model.

    Args:
        model (torch.nn.Module): The decoder model to be trained.
        train_dataloader (DataLoader): Dataloader for the training dataset.
        dev_dataloader (DataLoader): Dataloader for the development/validation dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the model on (e.g., "cuda" or "cpu").

    Returns:
        None
    """
    model.train()
    model_accuracy: float = 0.0
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Batch {i}/{len(train_dataloader)} Loss: {loss.item()}")
            with open("decoder_loss.txt", "a") as f:
                f.write(f"Epoch:{epoch+1}/{num_epochs}, Batch {i}/{len(train_dataloader)}, Loss: {loss.item()}\n")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        accuracy: float = evaluate_decoder_model(model, dev_dataloader, device)
        if accuracy > model_accuracy:
            model_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
    print(f"Best Accuracy: {model_accuracy:.4f}")
    with open("decoder_loss.txt", "a") as f:
        f.write(f"Best Accuracy: {model_accuracy:.4f}\n")

def evaluate_encoder_model(model, data_loader, device) -> float:
    """
    Evaluate the performance of an encoder model.

    Args:
        model (torch.nn.Module): The encoder model to be evaluated.
        data_loader (DataLoader): Dataloader for the evaluation dataset.
        device (torch.device): Device to run the model on (e.g., "cuda" or "cpu").

    Returns:
        float: The accuracy of the model on the evaluation dataset.
    """
    model.eval()
    correct = 0
    total = 0
    for i, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)
        labels = batch["labels"].to(device)
        segment_embedding = batch["segment_embedding"].to(device)

        outputs = model(input_ids, segment_embedding, attention_mask)

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

    return accuracy


def evaluate_decoder_model(model, data_loader, device) -> float:
    """
    Evaluate the performance of a decoder model.

    Args:
        model (torch.nn.Module): The decoder model to be evaluated.
        data_loader (DataLoader): Dataloader for the evaluation dataset.
        device (torch.device): Device to run the model on (e.g., "cuda" or "cpu").

    Returns:
        float: The accuracy of the model on the evaluation dataset.
    """
    model.eval()
    correct = 0
    total = 0
    for i, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)

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

    return accuracy


def test_encoder_model(model, dataloader, device, id2label):
    """
    Test the encoder model and save the predicted labels.

    Args:
        model (torch.nn.Module): The encoder model to be tested.
        dataloader (DataLoader): Dataloader for the test dataset.
        device (torch.device): Device to run the model on (e.g., "cuda" or "cpu").
        id2label (dict): Dictionary mapping label IDs to label names.

    Returns:
        None
    """
    model.eval()
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)
        # labels = batch["labels"].to(device)
        segment_embedding = batch["segment_embedding"].to(device)

        outputs = model(input_ids, segment_embedding, attention_mask)

        _, batch_predicted = torch.max(outputs, dim=2)
        for j in range(batch_predicted.shape[0]):
            predicted = batch_predicted[j][attention_mask.bool().squeeze(1).squeeze(1)[j]].cpu().numpy()
            predicted_labels = [id2label[i] for i in predicted]
            with open("./ner-data/dev_TAG_con.txt", "a") as f:
                f.write(f"{' '.join(predicted_labels)}\n")
        print(f"Batch {i + 1}/{len(dataloader)}")

class Trainer:
    """
    Trainer class to manage training, validation, and testing of models.

    Args:
        model (torch.nn.Module): The model to be trained.
        model_type (str): The type of model ("encoder_only" or "decoder_only").
        train_loader (DataLoader): Dataloader for the training dataset.
        dev_loader (DataLoader): Dataloader for the development/validation dataset.
        test_loader (DataLoader): Dataloader for the test dataset.
        device (str): Device to run the model on (e.g., "cuda" or "cpu").
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
    """
    def __init__(self, model, model_type, train_loader, dev_loader, test_loader, 
                device="cuda", num_epochs=50, lr=1e-5):
        self.model = model
        self.model_type = model_type
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr

    def train(self, multi_gpu: bool):
        """
        Train the model.

        Args:
            multi_gpu (bool): Whether to use multiple GPUs for training.

        Returns:
            None
        """
        if multi_gpu:
            self.model = DataParallel(self.model, device_ids=[2, 4])
            self.model.to(self.model.device_ids[0])
            self.device = self.model.device_ids[0]
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if self.model_type == "encoder_only":
            train_encoder_model(self.model, self.train_loader, self.dev_loader, criterion, optimizer, self.num_epochs, self.device)
        elif self.model_type == "decoder_only":
            train_decoder_model(self.model, self.train_loader, self.dev_loader, criterion, optimizer, self.num_epochs, self.device)
