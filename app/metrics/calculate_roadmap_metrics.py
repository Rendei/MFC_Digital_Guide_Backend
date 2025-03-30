from bert_score import score as bert_score
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def evaluate_text(generated_text, reference_text):
    # BLEU
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(
        [reference_text.split()], 
        generated_text.split(),
        smoothing_function=smoothie
    )

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_text, reference_text)[0]

    # BERTScore
    P, R, F1 = bert_score([generated_text], [reference_text], lang="ru")
    bert_f1 = F1.mean().item()

    metrics = {
        "BLEU": round(bleu_score, 3),
        "ROUGE-1 F1": round(rouge_scores["rouge-1"]["f"], 3),
        "ROUGE-2 F1": round(rouge_scores["rouge-2"]["f"], 3),
        "ROUGE-L F1": round(rouge_scores["rouge-l"]["f"], 3),
        "BERTScore F1": round(bert_f1, 3)
    }

    return metrics
