from nltk.translate.bleu_score import corpus_bleu
import progressbar as pb
import rouge
import glob
import os
import numpy as np

def prepare_results(metric,p, r, f):
    return '{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric.upper(),'P', p, 'R', r, 'F1', f)

def evaluate_model(model, decoder):
    np.random.seed(5050)
    actual_bleu, predicted_bleu = list(), list()
    actual_rouge, predicted_rouge = list(), list()
    decoder.load_test_data()
    #b = pb.ProgressBar(maxval=len(decoder.test_captions),widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    #b.start()
    #i=1
    print("Evaluating test images")
    for ts_key, ts_captions_list in decoder.test_captions.items():
        out = decoder.sequence_evaluation(model, decoder.image_encodings[ts_key])
        references = [d.split() for d in ts_captions_list]
        predicted_bleu.append(out.split())
        actual_bleu.append(references)
        predicted_rouge.append(out)
        actual_rouge.append(ts_captions_list)
        #b.update(i)
        #i += 1  
    #b.finish()  
    # calculate BLEU score
    print("\nEvaluation BLEU\n")
    bleu1=corpus_bleu(actual_bleu, predicted_bleu, weights=(1.0, 0, 0, 0))
    bleu2=corpus_bleu(actual_bleu, predicted_bleu, weights=(0.5, 0.5, 0, 0))
    bleu3=corpus_bleu(actual_bleu, predicted_bleu, weights=(0.3, 0.3, 0.3, 0))
    bleu4=corpus_bleu(actual_bleu, predicted_bleu, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1: %f' % bleu1)
    print('BLEU-2: %f' % bleu2)
    print('BLEU-3: %f' % bleu3)
    print('BLEU-4: %f' % bleu4)
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
        
    files_list = glob.glob('../result/*.h5') 
    filename = max(files_list, key=os.path.getctime).split('\\')[1]
    f=open("tests.txt", "a+")
    line=filename + ";" + str(bleu1) + ";" + str(bleu2) + ";" + str(bleu3) + ";" + str(bleu4) + ";" + "\n"
    f.write(line)
    f.close
    return bleu1
    


