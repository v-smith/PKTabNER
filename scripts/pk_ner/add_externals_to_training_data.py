import random

from pktabner.utils import read_jsonl, write_jsonl

external_data = list(read_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_sentences/test.jsonl"))
print(len(external_data))

train_data = list(read_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/ner_annotated_ready/pkner_train.jsonl"))


def sample_dicts(dict_list, sample_size=500, seed=42):
    random.seed(seed)  # optional, for reproducibility
    return random.sample(dict_list, k=sample_size)

sample_size = int(1 * len(train_data))

sampled_negs = sample_dicts(external_data, sample_size=sample_size, seed=42)

final_train = train_data + sampled_negs
random.shuffle(final_train)

print(len(final_train))

write_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/ner_annotated_ready/pkner_train_external100.jsonl", final_train)


a = 1