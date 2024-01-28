import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from einsten_chat777.models import Einsten777

class EinstenModelTrainer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = Einsten777.from_pretrained("bert-base-uncased", num_labels=8)
        self.optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_args = TrainingArguments(
            output_dir=model_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            save_total_limit=2,
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_strategy="steps",
            save_on_each_node=True,
        )

    def train(self, train_loader, eval_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask = batch
            labels = batch[0].unsqueeze(1)
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, eval_loader):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids, attention_mask = batch
                labels = batch[0].unsqueeze(1)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                _, predicted = torch.max(outputs.logits, dim=1)
                correct += (predicted == labels).sum().item()
        return correct / len(eval_loader)

    def save(self, epoch, loss):
        self.model.save_pretrained(os.path.join(self.model_dir, f"einsten777_{epoch}_{loss:.4f}.pth"))

class EinstenModelSaver:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def save(self, model):
        torch.save(model.state_dict(), os.path.join(self.model_dir, "einsten777.pth"))

    def load(self):
        return torch.load(os.path.join(self.model_dir, "einsten777.pth"), map_location=torch.device("cuda"))