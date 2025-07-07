from transformers import AutoTokenizer

check_tokens = ["tMAX(h)", "CMAX(ng/mL/pg/mL)", "θVC (L)", "θCL (L/h)", "θVT (L)",
                "Lambdaz/h", "t1/2z/h", "CLz/L/h/kg",
                "Vd/F - θ2 (L)", "Flast(%)", "Finf(%)", "k0.5^8", "AUC0-6^12", "CL/FAQ",
                "F%", "F(%)", "tmax-sp (h)", "AUClung^2", "Cavg,trachea^1", "AUClung"]

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
#'allenai/scibert_scivocab_uncased'
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# dmis-lab/biobert-v1.1
for x in check_tokens:
    tokens = tokenizer.tokenize(x)
    print(f"{x}: {tokens}\n")

a = 1
