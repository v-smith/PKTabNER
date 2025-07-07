"""Please note this script is intended to be used in the PKNER repo: """

import os

from transformers import BertTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pktabner.models.bert import load_pretrained_model
from pktabner.models.utils import predict_pl_bert_ner
from pktabner.utils import read_jsonl, clean_instance_span, get_ner_scores

test_data = list(read_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/ner_annotated_ready/pkner_test.jsonl"))

predict_sentences = []

for item in test_data:
    text = item["text"]
    for span in item.get("spans", []):
        if span["label"] == "PK":
            predict_sentences.append(item)

# ============== 1. Load model and tokenizer ========================= #
model_checkpoint = "/home/vsmith/PycharmProjects/PKNER/checkpoints/biobert-pk_ner-trained.ckpt"
pl_model = load_pretrained_model(model_checkpoint_path=model_checkpoint, gpu=True)
tokenizer = BertTokenizerFast.from_pretrained(pl_model.bert.name_or_path)

# ============= 2. Load corpus  ============================ #
true_entities = [
    clean_instance_span([span for span in x["spans"] if span.get("label") == "PK"])
    for x in predict_sentences
]

texts_to_predict = [sentence["text"] for sentence in predict_sentences]

a = 1
# ============= 4. Predict  ============================ #
predicted_entities = predict_pl_bert_ner(inp_texts=texts_to_predict, inp_model=pl_model, inp_tokenizer=tokenizer,
                                         batch_size=8, n_workers=2)

predicted_entities_offsets = [clean_instance_span(x) for x in predicted_entities]

get_ner_scores(pred_ents_ch=predicted_entities_offsets, true_ents_ch=true_entities, inp_tags=["PK"],
                   original_annotations=predict_sentences, display_errors=True)


a = 1