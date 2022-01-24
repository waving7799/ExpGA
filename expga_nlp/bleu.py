from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from preprocessing_data.read_files import split_imdb_files,split_agnews_files,split_yahoo_files
from nlp_fariness.process_data import word_wiki_process, read_wiki_files,read_sst_files

smooth = SmoothingFunction()  # 定义平滑函数对象
# dataset = 'imdb'
# files_dict = {"imdb": split_imdb_files, 'wiki': read_wiki_files, 'sst': read_sst_files}
# train_texts, train_labels, test_texts, test_labels = files_dict[dataset]()

# references = [['我', '是', '谁'], ['ta ', '是', '谁']]
references = ['我', '是', '谁','我', '是', '谁','我', '是', '谁','我', '是', '谁','我', '是', '谁']
candidate = ['我', '是', 'ta','我', '是', '谁','我', '是', '谁','我', '是', '谁','我', '是', '谁']
# references =  ['this', 'is' 'test']
# candidate = ['this', 'is', 'a', 'test']


weights = [1]

# corpus_score_2 = corpus_bleu(references, candidate)

corpus_score_2 = corpus_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
corpus_score_4 = corpus_bleu(references, candidate,weights, smoothing_function=smooth.method1)
sentence_bleu_6 = sentence_bleu(references, candidate, weights)
print(corpus_score_2)
print(corpus_score_4)
print(sentence_bleu_6)


