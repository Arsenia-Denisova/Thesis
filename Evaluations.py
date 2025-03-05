# load libraries
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
import pandas as pd
from comet import download_model, load_from_checkpoint


# process files with translations for BLEU metric

with open("correct_target_gpt.txt", 'r', encoding='utf-8') as ref_gpt:
    ref_lines_gpt = [line.strip().split() for line in ref_gpt.readlines()]

with open("original_target_gpt.txt", 'r', encoding='utf-8') as trans_gpt:
    trans_lines_gpt = [line.strip().split() for line in trans_gpt.readlines()]

with open("correct_target_deepl.txt", 'r', encoding='utf-8') as ref_deepl:
    ref_lines_deepl = [line.strip().split() for line in ref_deepl.readlines()]

with open("original_target_deepl.txt", 'r', encoding='utf-8') as trans_deepl:
    trans_lines_deepl = [line.strip().split() for line in trans_deepl.readlines()]


# run BLEU 

smooth_fn = SmoothingFunction().method1
bleu_gpt = []
for ref, trans in zip(ref_lines_gpt, trans_lines_gpt):
    bleu_score = sentence_bleu([ref], trans, smoothing_function=smooth_fn)
    bleu_gpt.append(bleu_score)

average_bleu_gpt = sum(bleu_gpt) / len(bleu_gpt)

bleu_deepl = []
for ref, trans in zip(ref_lines_deepl, trans_lines_deepl):
    bleu_score = sentence_bleu([ref], trans, smoothing_function=smooth_fn)
    bleu_deepl.append(bleu_score)

average_bleu_deepl = sum(bleu_deepl) / len(bleu_deepl)


# process files with sentences for other metrics

df = pd.read_csv('rlc_test.csv')
original_src = df["text_orig"]
correct_src = df["text_cor"]

with open("original_src.txt", "w") as f:
    for s in original_src:
        f.write(s + "\n")

with open("correct_src.txt", "w") as f:
    for s in correct_src:
        f.write(s + "\n")

with open("correct_target_gpt.txt", 'r', encoding='utf-8') as f:
    reference_gpt = [line.strip() for line in f]

with open("original_target_gpt.txt", 'r', encoding='utf-8') as f:
    translation_gpt = [line.strip() for line in f]

with open("original_src.txt", 'r', encoding='utf-8') as f:
    source= [line.strip() for line in f]

with open("correct_src.txt", 'r', encoding='utf-8') as f:
    source_cor= [line.strip() for line in f]

with open("correct_target_deepl.txt", 'r', encoding='utf-8') as f:
    reference_deepl = [line.strip() for line in f]

with open("original_target_deepl.txt", 'r', encoding='utf-8') as f:
    translation_deepl = [line.strip() for line in f]


# run chrF on corpus

chrf_gpt = sacrebleu.corpus_chrf(translation_gpt, [reference_gpt])
chrf_deepl = sacrebleu.corpus_chrf(translation_deepl, [reference_deepl])

# run chrF on sentences

chrf_gpt_sent = [sacrebleu.sentence_chrf(hyp, [ref]).score for hyp, 
                 ref in zip(translation_gpt, reference_gpt)]
chrf_deepl_sent = [sacrebleu.sentence_chrf(hyp, [ref]).score for hyp, 
                   ref in zip(translation_deepl, reference_deepl)]


# run COMET22

COMET22_path = download_model("Unbabel/wmt22-comet-da")
COMET22 = load_from_checkpoint(COMET22_path)

COMET22_inputs_gpt = [
    {"src": src, "mt": trans, "ref": ref}
    for src, trans, ref in zip(source, translation_gpt, reference_gpt)
]
COMET22_inputs_deepl = [
    {"src": src, "mt": trans, "ref": ref}
    for src, trans, ref in zip(source, translation_deepl, reference_deepl)
]

COMET22_gpt = COMET22.predict(COMET22_inputs_gpt, batch_size=8)
COMET22_gpt.system_score
COMET22.scores

COMET22_deepl = COMET22.predict(COMET22_inputs_deepl, batch_size=8)
COMET22_deepl.system_score
COMET22_deepl.scores


# run COMETKIWI

COMETKIWI_path = download_model("Unbabel/wmt22-cometkiwi-da")
COMETKIWI = load_from_checkpoint(COMETKIWI_path)

COMETKIWI_inputs_orig_gpt = [{"src": src, "mt": trans} for src, 
                             trans in zip(source, translation_gpt)]

COMETKIWI_inputs_orig_deepl = [{"src": src, "mt": trans} for src, 
                               trans in zip(source, translation_deepl)]

COMETKIWI_inputs_cor_gpt = [{"src": src, "mt": trans} for src, 
                            trans in zip(source_cor, reference_gpt)]

COMETKIWI_inputs_cor_deepl = [{"src": src, "mt": trans} for src, 
                              trans in zip(source_cor, reference_deepl)]

COMETKIWI_gpt_orig = COMETKIWI.predict(COMETKIWI_inputs_orig_gpt, batch_size=8)
COMETKIWI_gpt_orig.system_score
COMETKIWI_gpt_orig.scores

COMETKIWI_deepl_orig = COMETKIWI.predict(COMETKIWI_inputs_orig_deepl, batch_size=8)
COMETKIWI_deepl_orig.system_score
COMETKIWI_deepl_orig.scores

COMETKIWI_gpt_cor = COMETKIWI.predict(COMETKIWI_inputs_cor_gpt, batch_size=8)
COMETKIWI_gpt_cor.system_score
COMETKIWI_gpt_cor.scores

COMETKIWI_deepl_cor = COMETKIWI.predict(COMETKIWI_inputs_cor_deepl, batch_size=8)
COMETKIWI_deepl_cor.system_score
COMETKIWI_deepl_cor.scores



