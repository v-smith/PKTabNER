import random

from pktabner.utils import read_jsonl, write_jsonl

negative_cell_data = list(read_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/Non_PK_updated_parsing_cells_non_numeric_ner-annotated_5755.jsonl"))
print(len(negative_cell_data))

train_data = list(read_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/ner_annotated_ready/pkner_train.jsonl"))

a = 1
# Remove non-PK spans
for item in negative_cell_data:
    item["spans"] = []
print(f"Negs: {len(negative_cell_data)}")


def deduplicate_by_text(dict_list):
    seen = set()
    deduped = []
    for d in dict_list:
        text_val = d.get("text")
        if text_val not in seen:
            seen.add(text_val)
            deduped.append(d)
    return deduped

deduped_negative_cell_data = deduplicate_by_text(negative_cell_data)
print(len(deduped_negative_cell_data))  # should be 2

def sample_dicts(dict_list, sample_size=500, seed=42):
    random.seed(seed)  # optional, for reproducibility
    return random.sample(dict_list, k=sample_size)

sample_size = int(1 * len(train_data))

sampled_negs = sample_dicts(deduped_negative_cell_data, sample_size=sample_size, seed=42)

final_train = train_data + sampled_negs
random.shuffle(final_train)
print(len(final_train))

write_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/ner_annotated_ready/pkner_train_neg100.jsonl", final_train)


a = 1