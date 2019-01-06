import decoder
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import progressbar as pb
import time
import rouge

def prepare_results(metric,p, r, f):
    return '{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric.upper(),'P', p, 'R', r, 'F1', f)

def evaluate_model(model, decoder):
    actual_bleu, predicted_bleu = list(), list()
    actual_rouge, predicted_rouge = list(), list()
    decoder.load_test_data()
    b = pb.ProgressBar(maxval=len(decoder.test_captions),widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    b.start()
    i=1
    print("Evaluating test images")
    for ts_key, ts_captions_list in decoder.test_captions.items():
        out = decoder.sequence_evaluation(model, decoder.image_encodings[ts_key])
        references = [d.split() for d in ts_captions_list]
        predicted_bleu.append(out.split())
        actual_bleu.append(references)
        predicted_rouge.append(out)
        actual_rouge.append(ts_captions_list)
        b.update(i)
        i += 1  
    b.finish()  
    # calculate BLEU score
    print("\nEvaluation BLEU\n")
    print('BLEU-1: %f' % corpus_bleu(actual_bleu, predicted_bleu, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual_bleu, predicted_bleu, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual_bleu, predicted_bleu, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual_bleu, predicted_bleu, weights=(0.25, 0.25, 0.25, 0.25)))
    # calculate ROUGE score
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg='Avg',
                           apply_best='Avg',
                           alpha=0.5,
                           weight_factor=1.2,
                           stemming=True)
    scores = evaluator.get_scores(predicted_rouge, actual_rouge)
    print("\nEvaluation ROUGE with Avg\n")
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print(prepare_results(metric, results['p'], results['r'], results['f']))

print("Loading data...")
start = time.time()
filename = '../result/model-epoch01-train_loss4.63-val_loss4.09.h5'
model = load_model(filename)
dc = decoder.decoder()
end = time.time()
print("Data loaded!.\nTime: %0.2f seconds." % (end - start))

print("Evaluating BLEU and ROUGE...")
start = time.time()
evaluate_model(model,dc)
end = time.time()
print("\nEvaluation done!.\nTime: %0.2f seconds." % (end - start))

