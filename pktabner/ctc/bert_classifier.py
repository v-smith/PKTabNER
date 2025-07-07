from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput


def format_input_for_cell(cell_content, row_context, col_context, pos_row, pos_col, rev_pos_row, rev_pos_col):
    text_parts = [
        "<CELL>", cell_content,
        "<ROW_HEADERS>", row_context,
        "<COL_HEADERS>", col_context,
        "<POS>", f"{pos_row},{pos_col}",
        "<RPOS>", f"{rev_pos_row},{rev_pos_col}",
    ]

    a = 1
    return " ".join(text_parts)


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, label_list, input_col_map, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.input_col_map = input_col_map
        self.max_length = max_length
        a = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = self.label_list[idx]

        input_text = format_input_for_cell(
            row['cell_text'],
            row['row_headers_flat'],
            row['col_headers_flat'],
            row['pos_row_idx'],
            row['pos_col_idx'],
            row['rev_pos_row_idx'],
            row['rev_pos_col_idx']
        )

        # Tokenize full input
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Generate `input_type_ids` by mapping token positions based on special tokens
        input_ids = encoding["input_ids"][0]
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        input_type_ids = self._assign_input_type_ids(input_tokens)

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"][0],
            "input_type_ids": input_type_ids,
            "labels": torch.tensor(label, dtype=torch.long)
        }

    def _assign_input_type_ids(self, tokens):
        type_map = {
            "<CELL>": 0,
            "<ROW_HEADERS>": 1,
            "<COL_HEADERS>": 2,
            "<POS>": 3,
            "<RPOS>": 4,
        }

        current_type = 0
        input_type_ids = []

        for tok in tokens:
            if tok in type_map:
                current_type = type_map[tok]
            input_type_ids.append(current_type)

        return torch.tensor(input_type_ids, dtype=torch.long)







class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()

        if num_layers == 1:
            self.seq = nn.Sequential(nn.Linear(input_dim, output_dim, bias=True))
        elif num_layers == 2:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True), nn.GELU(), nn.Linear(hidden_dim, output_dim, bias=True))
        elif num_layers == 3:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True), nn.GELU(),
                nn.Linear(hidden_dim, output_dim, bias=True))

            raise NotImplementedError(f"MLP layer number = {num_layers} is not implemented!")

        self.seq.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        return self.seq(x)


class SciBertWithAdditionalFeatures(nn.Module):
    """
    Adapted SciBert Model with addition of table input type embeddings, adapted from https://github.com/allenai/S2abEL.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.text_encoder = AutoModel.from_pretrained(config.get("pretrained"))
        self.input_type_embedding = nn.Embedding(len(config.get("input_cols")), 768)
        self._init_weights(self.input_type_embedding)

        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.num_labels = config.get("num_labels")
        self.classifier = MLPHead(output_dim=config.get("num_labels"), input_dim=768, hidden_dim=32,
                                  num_layers=config.get("num_MLP_layers"))

        if "class_weights" in config:
            weights = config["class_weights"]
            self.class_weights = weights
        else:
            self.class_weights = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                input_type_ids: Optional[torch.LongTensor] = None,
                labels=None
                ):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        position_ids = self.text_encoder.embeddings.position_ids[:, : seq_length]
        extended_attention_mask: torch.Tensor = attention_mask[:, None, None, :]
        input_type_embeddings = self.input_type_embedding(input_type_ids) # used to differentiate token origin (e.g., cell, column header etc.)

        # adding extra trainable embeddings to denote token types.
        word_embeddings = self.text_encoder.embeddings.word_embeddings(input_ids)
        position_embeddings = self.text_encoder.embeddings.position_embeddings(position_ids)

        embeddings = word_embeddings + input_type_embeddings + position_embeddings
        embeddings = self.text_encoder.embeddings.LayerNorm(embeddings)
        embedding_output = self.text_encoder.embeddings.dropout(embeddings)

        encoder_outputs = self.text_encoder.encoder(hidden_states=embedding_output,
                                                    attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]  # last_hidden_state

        logits = None
        loss = None
        if labels is not None:
            # average of output token embeddings
            pooled_output = torch.mean(sequence_output, dim=1) # OR use CLS: pooled_output = self.text_encoder.pooler(sequence_output)  # (BS, 768)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            loss_fct = CrossEntropyLoss(
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if labels is not None:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        else:
            return BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
            )


def train_one_epoch(
    model, dataloader, optimizer, scheduler, device,
    epoch, writer, val_loader,
    scaler=None, eval_every=100,
    clip_grad_norm=1.0,
    global_step_start=0,
    accumulation_steps=1
):
    model.train()
    total_loss = 0
    global_step = global_step_start

    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"].to(dtype=torch.float),
                input_type_ids=batch["input_type_ids"],
                labels=batch["labels"]
            )
            loss = outputs.loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        writer.add_scalar("Loss/train_batch", loss.item() * accumulation_steps, global_step)

        if (step + 1) % eval_every == 0:
            val_loss, val_f1 = evaluate(model, val_loader, device)
            writer.add_scalar("Loss/val_step", val_loss, global_step)
            writer.add_scalar("F1/val_step", val_f1, global_step)

        global_step += 1

    avg_loss = total_loss / len(dataloader)
    return avg_loss, global_step


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"].to(dtype=torch.float),
                input_type_ids=batch["input_type_ids"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, f1
