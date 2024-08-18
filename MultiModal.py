import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import datetime as dt
from tqdm import tqdm
import os
import preprocessing 

# EarlyStopping 클래스 정의 (여기에서는 사용한다고 가정하고 추가했습니다)
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Embedding 클래스 정의 (기본적인 형태로 추가했습니다. 실제 기능에 맞게 수정하세요.)
class Embedding:
    def clean_text(self, text):
        # 텍스트 정리 작업 (여기서는 간단히 텍스트를 소문자로 변환)
        return text.lower()

    def hashtag_extraction(self, text):
        # 해시태그 추출 작업 (여기서는 간단히 "#"으로 시작하는 단어를 추출하는 예시)
        return " ".join([word for word in text.split() if word.startswith("#")])

# 조회수 카테고리화 함수 정의
def categorize_views(views):
    # 조회수를 카테고리로 변환하는 예시 함수 (구체적인 구현은 데이터에 맞게 수정)
    if views < 1000:
        return "Low"
    elif views < 10000:
        return "Medium"
    else:
        return "High"

# MultimodalDataset 클래스 정의
class MultimodalDataset(Dataset):
    def __init__(self, texts, subscribers, labels, tokenizer, max_len):
        self.texts = texts
        self.subscribers = subscribers
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        subscribers = self.subscribers.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'subscribers': torch.tensor(subscribers, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# MultimodalBertModel 클래스 정의
class MultimodalBertModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(MultimodalBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + 1, 128)  # BERT의 출력과 구독자 수를 결합
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask, subscribers, labels=None):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = bert_output.pooler_output
        combined_input = torch.cat((pooled_output, subscribers.unsqueeze(1)), dim=1)
        x = torch.relu(self.fc1(combined_input))
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits

# 학습 함수 정의
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        subscribers = d["subscribers"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            subscribers=subscribers,
            labels=labels
        )

        loss = outputs[0]
        logits = outputs[1]
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# 평가 함수 정의
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            subscribers = d["subscribers"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                subscribers=subscribers,
                labels=labels
            )

            loss = outputs[0]
            logits = outputs[1]
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# 메인 실행부
if __name__ == "__main__":
    x = dt.datetime.now()
    month = x.month
    year = x.year
    day = x.day
    today = f"{year}-{month}-{day}"
    df_list = glob.glob("/home/bailey/workspace/test/*.csv")
    df = pd.DataFrame()
    for i in df_list:
        temp = pd.read_csv(i)
        # temp = temp.dropna(how="any")
        df = pd.concat((temp, df), axis=0)
    
    # df = df.groupby("video_id").apply(lambda x: x.sort_values(by="video_published").head(1))
    df = df.sample(frac=1, random_state=1000)
    df['vd.hashtag'].apply(lambda x: preprocessing.clean_text(x))
    df['vd.hashtag'] = df['vd.hashtag'].fillna(df['vd.video_tags'])
    # nan값 삭제 
    df = df.dropna(how="any")
    df['vd.video_tags'] = df['vd.video_tags'].apply(lambda x: preprocessing.clean_text(x))
    df['video_hashtag'] = df['vd.hashtag'].apply(lambda x: preprocessing.clean_text(x))
    df['view_category'] = df['vh.video_views'].apply(categorize_views)
    
    le = LabelEncoder()
    df['view_category'] = le.fit_transform(df['view_category'])

    model_name = 'monologg/kobert'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = MultimodalBertModel(bert_model_name=model_name, num_labels=5)  # 5개의 카테고리

    df['text'] = df['video_tags'] + " " + df['video_title'] + " " + df['video_hashtags']

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'], df['view_category'], test_size=0.2, random_state=42)
    
    train_subs, test_subs = train_test_split(df['channel_subscribers'], test_size=0.2, random_state=42)

    # 데이터셋 및 데이터로더 생성
    train_dataset = MultimodalDataset(train_texts, train_subs, train_labels, tokenizer, max_len=32)
    test_dataset = MultimodalDataset(test_texts, test_subs, test_labels, tokenizer, max_len=32)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    # GPU 사용 설정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-8)
    EPOCHS = 1000
    patience = 10
    min_delta = 1e-16
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    PATH = "/home/bailey/workspace/related_words/model/best_model/"
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            device
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            test_dataloader,
            device
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        torch.save(model.state_dict(), f"{PATH}best_model_{today}_{epoch}.pt")
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
