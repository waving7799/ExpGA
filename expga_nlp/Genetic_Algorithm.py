import sys
sys.path.insert(0, '../')
import numpy as np
from keras.preprocessing.text import Tokenizer
from preprocessing_data.word_level_process import  get_tokenizer
from preprocessing_data.read_files import split_imdb_files, split_yahoo_files, split_agnews_files
from keras.preprocessing import sequence
from utils.config_nlp import config
from OpenAttack.substitutes import CounterFittedSubstitute
from OpenAttack.text_processors import DefaultTextProcessor
from keras.models import load_model
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DEFAULT_SKIP_WORDS = {"the", "and", "a", "of", "to", "is", "it", "in", "i", "this", "that", "was", "as", "for", "with",
                      "movie", "but", "film", "on", "not", "you", "he", "are", "his", "have", "be"}

DEFAULT_CONFIG = {
    "skip_words": DEFAULT_SKIP_WORDS,
    "pop_size": 10,
    "perturb_num": 10,
    "neighbour_threshold": 0.4,
    "processor": DefaultTextProcessor(),
    "top_n1": 20,
    "mutation": 0.4,
    "substitute": None,
}

def word_process(train_texts,tokenizer,dataset):
    maxlen = config.word_max_len[dataset]
    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=maxlen, padding='post', truncating='post')
    return x_train

class GeneticAlgorithm():
    def __init__(self, **kwargs):
        """
        :param list skip_words: A list of words which won't be replaced during the attack. **Default:** A list of words that is most frequently used.
        :param int pop_size: Genetic algorithm popluation size. **Default:** 20
        :param int max_iter: Maximum generations of genetic algorithm. **Default:** 20
        :param float neighbour_threshold: Threshold used in substitute module. **Default:** 0.5
        :param int top_n1: Maximum candidates of word substitution. **Default:** 20
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`CounterFittedSubstitute`
        :Classifier Capacity: Probability

        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = CounterFittedSubstitute()


    def __call__(self, clsf, x_orig, tokenizer, max_iters, func, sens_key, dataset):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        x_orig_t = word_process(x_orig, tokenizer, dataset)
        x_orig = x_orig[0].lower()

        x_orig = self.config["processor"].get_tokens(x_orig)

        target_pre = clsf.predict(x_orig_t)[0]
        target = np.argmax(target_pre)

        x_pos = list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        neighbours_nums = [
            self.get_neighbour_num(word, pos) if word not in self.config["skip_words"] else 0
            for word, pos in zip(x_orig, x_pos)
        ]
        neighbours = [
            self.get_neighbours(word, pos, self.config["top_n1"])
            if word not in self.config["skip_words"]
            else []
            for word, pos in zip(x_orig, x_pos)
        ]
        #
        if np.sum(neighbours_nums) == 0:
            return None
        w_select_probs = neighbours_nums / np.sum(neighbours_nums)

        # value = value[:len(x_orig)]
        # w_select_probs = value / np.sum(value)
        # w_select_probs = np.ones(len(x_orig))
        pop = [  # generate population
            self.perturb(
                clsf, x_orig, x_orig, neighbours, w_select_probs, target, tokenizer,sens_key, dataset
            )
            for _ in range(self.config["pop_size"])
        ]


        for i in range(max_iters):
            pop_t = word_process(pop, tokenizer, dataset)
            pop_preds = clsf.predict(pop_t)
            top_attack = np.argmax(-pop_preds[:, target])

            pop_scores, isFinish = func(pop,sens_key,clsf, tokenizer,target,dataset,x_orig)
            # if isFinish==True:
            #     return None

            # pop_scores = 1.0 - pop_scores
            if np.sum(pop_scores) == 0:
                return None
            pop_scores = pop_scores / np.sum(pop_scores)

            # selection
            select_index = np.random.choice(np.arange(self.config["pop_size"]), self.config["pop_size"], replace=True,
                                             p=pop_scores)
            new_pop = [pop[i] for i in select_index]
            pop = new_pop

            # crossover
            elite = [pop[top_attack]]
            parent_indx_1 = np.random.choice(
                self.config["pop_size"], size=self.config["pop_size"] - 1
            )
            parent_indx_2 = np.random.choice(
                self.config["pop_size"], size=self.config["pop_size"] - 1
            )
            children = [
                self.crossover(pop[p1], pop[p2])
                for p1, p2 in zip(parent_indx_1, parent_indx_2)
            ]

            # mutation
            mutations = []
            for x_cur in children:
                if np.random.rand() < self.config["mutation"]:
                    mutations.append(self.perturb(
                    clsf, x_orig, x_orig, neighbours,  w_select_probs, target, tokenizer,sens_key, dataset
                ))
                else:
                    mutations.append(x_cur)

            # children = [
            #     self.perturb(
            #         clsf, x_cur, x_orig, w_select_probs, x_pos, target, tokenizer, sens_key, dataset
            #     ) for x_cur in children
            # ]

            pop = elite + mutations

        return None  # Failed


    def get_neighbour_num(self, word, pos):
        threshold = self.config["neighbour_threshold"]
        try:
            return len(self.config["substitute"](word, pos, threshold=threshold))
        except :
            return 0

    def get_neighbours(self, word, pos, num):
        threshold = self.config["neighbour_threshold"]
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.config["substitute"](word, pos, threshold=threshold)[:num],
                )
            )
        except :
            return []

    def select_best_replacements(
            self, clsf, mod_idx, neighbours, x_cur, x_orig, target, tokenizer, dataset
    ):

        def do_replace(word):
            ret = x_cur.copy()
            for i in range(len(word)):
                ret[mod_idx[i]] = neighbours[i][word[i]]
            return ret

        new_list = []
        rep_words = []

        for i in range(len(mod_idx)):
            word = []
            for k in range(len(mod_idx)):

                word.append(np.random.choice(len(neighbours[k]), 1)[0])
            new_list.append(do_replace(word))
            rep_words.append(word)

        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)


        test_input = word_process(new_list, tokenizer, dataset)
        pred_scores = clsf.predict(test_input)
        pred_scores = pred_scores[:,target]
        new_scores = pred_scores[-1] - pred_scores[:-1]

        if np.max(new_scores) > 0 :
            return new_list[np.argmax(new_scores)]
        else:
            return x_cur


    def perturb(
            self, clsf, x_cur, x_orig, neighbours,  w_select_probs, target, tokenizer,sens_key, dataset
    ):
        x_len = len(x_cur)
        num_mods = 0
        for i in range(x_len):
            if x_cur[i] != x_orig[i]:
                num_mods += 1
        # mod_idx = np.random.choice(len(w_select_probs), 1, p=w_select_probs)[0]
        mod_idx = np.random.choice(x_len, self.config["perturb_num"], p=w_select_probs)

        # if num_mods < np.sum(np.sign(w_select_probs)):  # exists at least one indx not modified
        # while x_cur[mod_idx] != x_orig[mod_idx] or x_cur[mod_idx] == sens_key:  # already modified
            # mod_idx = np.random.choice(len(w_select_probs), 1, p=w_select_probs)[0]  # random another indx
        mod_word = [x_cur[idx] for idx in mod_idx]
        count = 0
        while sens_key in set(mod_word):  # already modified
            # sens_index = mod_idx.index(sens_key)
            # mod_idx.reomve(sens_index)
            mod_idx = np.random.choice(x_len, self.config["perturb_num"], p=w_select_probs)
            mod_word = [x_orig[idx] for idx in mod_idx]
            count +=1
            if count>10:
                return x_cur

        # neighbours = [self.get_neighbours(x_orig[idx], x_pos[idx], self.config["top_n1"])
        #               for idx in mod_idx]
        neighbours = [neighbours[idx] for idx in mod_idx]

        return self.select_best_replacements(
            clsf, mod_idx, neighbours, x_cur, x_orig, target, tokenizer, dataset
        )

    def crossover(self, x1, x2):
        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret

if __name__ == '__main__':
    ga = GeneticAlgorithm()
    test = ["If you like adult comedy cartoons, like South Park, then this is nearly a similar format about the small adventures of three teenage girls at Bromwell High. Keisha, Natella and Latrina have given exploding sweets and behaved like bitches, I think Keisha is a good leader. There are also small stories going on with the teachers of the school. There's the idiotic principal, Mr. Bip, the nervous Maths teacher and many others. The cast is also fantastic, Lenny Henry's Gina Yashere, EastEnders Chrissie Watts, Tracy-Ann Oberman, Smack The Pony's Doon Mackichan, Dead Ringers' Mark Perry and Blunder's Nina Conti. I didn't know this came from Canada, but it is very good. Very good!"]

    model_path = "../models/keras/imdb_word_lstm.h5"

    dataset = "imdb"
    train_texts, train_labels, test_texts, test_labels = split_imdb_files()
    tokenizer = Tokenizer(num_words=config.num_words[dataset])
    tokenizer.fit_on_texts(test)
    # test_input = word_process(test, tokenizer)

    clsf = load_model(model_path)
    # result = clsf.predict(test_input)
    result = ga(clsf,test,tokenizer,max_iters=20)
    print(result)