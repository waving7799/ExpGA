import  sys
sys.path.insert(0, '../')  # the code for fair classification is in this directory
from preprocessing_data.read_files import split_imdb_files,split_agnews_files,split_yahoo_files
from keras.models import load_model
import numpy as np
import random
import time
from keras.preprocessing.text import Tokenizer
import shap
from expga_nlp.process_data import word_wiki_process, read_wiki_files,read_sst_files
from expga_nlp.Genetic_Algorithm import GeneticAlgorithm as GA
import keras.backend as K
from tensorflow.python.platform import flags
from scipy.spatial.distance import  cdist
from keras.preprocessing import sequence
import requests
from expga_nlp.config_word import config as conf
from utils.config_nlp import config
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from lime.lime_tabular import LimeTabularExplainer
import lime
import warnings

co = ConfigProto()
co.gpu_options.allow_growth = True
session = InteractiveSession(config=co)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)


from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smooth = SmoothingFunction()  # 定义平滑函数对象


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

Skip_Words = conf.Skip_Words
attributes_key = conf.attributes_key
country_set = conf.country_set


global_disc_inputs = set()
global_disc_inputs_list = []
local_disc_inputs = set()
local_disc_inputs_list = []
tot_inputs = set()
disc_key = []
location = []

def get_tokenizer(dataset,train_texts):
    tokenizer = Tokenizer(num_words=config.num_words[dataset])
    tokenizer.fit_on_texts(train_texts)
    return tokenizer

def word_process(train_texts,tokenizer,dataset):
    maxlen = config.word_max_len[dataset]
    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=maxlen, padding='post', truncating='post')
    return x_train


def ConstructExplainer(train_vectors, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(train_vectors, feature_names=feature_names,
                                                       class_names=class_names, discretize_continuous=False)
    return explainer

def get_lime_values(x_train, x_test, model, toplabel,explainer):
    num = 20
    warnings.filterwarnings("ignore")

    exp = explainer.explain_instance(x_test[0], model.predict, num_features=num)
    explain_labels = exp.available_labels()[0]
    exp_result = exp.as_list(label=explain_labels)
    rank = []
    for j in range(len(exp_result)):
        rank.append(exp_result[j][0])
        if len(rank) == 20: break
    return rank

def get_shap_values(x_train,x_test,model,toplabel):
    background = x_train[np.random.choice(x_train.shape[0], 50, replace=False)]
    # e = shap.DeepExplainer(model, background)
    # shap_values = e.shap_values(x_test)
    explainer = shap.KernelExplainer(model.predict, background, link="logit")
    shap_values = explainer.shap_values(x_test)
    rank = []
    top_values = shap_values[toplabel][0]
    value = abs(top_values)
    sorted_shapValue = []
    for j in range(len(top_values)):
        temp = []
        temp.append(int(j))
        temp.append(top_values[j])
        sorted_shapValue.append(temp)
    sorted_shapValue.sort(key=lambda x: x[1], reverse=True)
    exp_result = sorted_shapValue
    for k in range(len(exp_result)):
        rank.append(exp_result[k][0])
        if len(rank)==20: break
    return rank, value


def global_discovery(iteration,params,input_bounds,sensitive_param):
    samples = []
    while len(samples) < iteration:
        x = np.zeros(params)
        for i in range(params):
            # random.seed(time.time())
            x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
        x[sensitive_param - 1] = 0

        samples.append(x)

    return samples


def evaluate_global(train_tests,max_global,sensitive_param,tokenizer,model,dataset,explainer):
    reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])
    x_train = word_process(train_tests, tokenizer,dataset)
    seeds = []
    sens_key = []
    for i in range(max_global):
        inp = [train_tests[i].lower()]
        tot_inputs.add(tuple(map(tuple, inp)))
        inp_t = word_process(inp, tokenizer,dataset)
        toplabel = np.argmax(model.predict(inp_t))
        rank, value = get_shap_values(x_train, inp_t, model, toplabel)
        #rank = get_lime_values(x_train, inp_t, model, toplabel, explainer)
        xai_word = [reverse_word_index.get(inp_t[0][rank[k]]) for k in range(len(rank))]
        print(xai_word)
        for word in xai_word:
            try:
                if word not in Skip_Words:
                    re_text = ' http://api.conceptnet.io/query?start=/c/en/'+word+'&rel=/r/IsA&limit=100'
                    obj = requests.get(re_text).json()
                    if len(obj["edges"])>0:
                        # inp_key = set(list(map(lambda x: x["end"]['@id'].split("/")[3] , obj['edges'])))
                        inp_key = []
                        for edge in obj["edges"]:
                            if edge["weight"]>=1:
                                inp_key.append(edge["end"]['@id'].split("/")[3])
                        inp_key = set(inp_key)
                        if len(inp_key & attributes_key)>0 or len(inp_key & country_set)>0:
                            seeds.append(inp)
                            sens_key.append(word)
                            location.append(xai_word.index(word))
                            print(seeds)
            except:
                pass
    return seeds, sens_key


def evaluate_local(pop,sens_key,model, tokenizer,target,dataset,x_ori):
    fitness = []
    isFinish = False
    weights = [1, 0, 0, 0]
    for i in range(len(pop)):
        inp = pop[i]
        bleu_score = corpus_bleu(x_ori, inp, weights, smoothing_function=smooth.method1)
        print(bleu_score)
        if bleu_score<0.8:
            fitness.append(0)
            continue
        ret0 = inp.copy()
        ret1 = inp.copy()
        index = inp.index(sens_key)
        if sens_key in country_set:
            rep_word = random.choice(list(country_set))
            ret1 = ret1[:index]+[rep_word] + ret1[index+1:]

        else:
            rep_words = random.sample(country_set,2)
            ret1.insert(index,rep_words[1])
        tot_inputs.add(tuple(inp))
        ret0_t = word_process([ret0], tokenizer,dataset)
        ret1_t = word_process([ret1], tokenizer,dataset)
        pred0 = model.predict(ret0_t)
        pred1 = model.predict(ret1_t)
        out0 = np.argmax(pred0)
        out1 = np.argmax(pred1)

        if (abs(out0 - out1) > 0 and (tuple(map(tuple, inp)) not in global_disc_inputs)
                and (tuple(map(tuple, inp)) not in local_disc_inputs)):
            disc_key.append(sens_key)
            local_disc_inputs.add(tuple(map(tuple, inp)))
            local_disc_inputs_list.append(" ".join([str(x) for x in inp]))

            print("Percentage discriminatory inputs - " + str(
                float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
            print("Total Inputs are " + str(len(tot_inputs)))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
            isFinish = True

        fitness.append(abs(pred0[0][target]-pred1[0][target]))
    return fitness, isFinish


def xai_fair_testing(dataset,sens_param,model_path,max_global, max_local):

    model = load_model(model_path)
    start = time.time()

    files_dict = {"imdb":split_imdb_files,'wiki':read_wiki_files, 'sst':read_sst_files}
    train_texts, train_labels, test_texts, test_labels = files_dict[dataset]()
    tokenizer = get_tokenizer(dataset,train_texts)

    model_name = model_path.split("_")[-1].split(".")[0]
    file_name = "expga_{}_{}_{}.txt".format(model_name,dataset,sens_param)
    f =open(file_name,"a")
    f.write("iter:"+str(iter)+"------------------------------------------"+"\n"+"\n")
    f.close()

    x_train = word_process(train_texts, tokenizer, dataset)
    class_names = ["neg","pos"]
    feature_names = np.arange(0, len(x_train[0]), 1)
    explainer = ConstructExplainer(x_train, feature_names, class_names)
    seed_list, seeds_key = evaluate_global(train_texts,max_global,sens_param, tokenizer,model,dataset, explainer)

    
    print('Finish Searchseed')
    print(seeds_key)

    print("Finished Global Search")
    print('length of total input is:' + str(len(tot_inputs)))
    print('length of global discovery is:' + str(len(seed_list)))
    print("Percentage discriminatory inputs - " + str(float(len(seed_list)
                                                            + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
    end = time.time()
    print('Total time:' + str(end - start))

    f = open(file_name,"a")
    f.write('length of total input is:' + str(len(tot_inputs))+"\n")
    f.write('length of global discovery is:' + str(len(seed_list))+"\n")
    f.write("Percentage discriminatory inputs - " + str(float(len(seed_list)
                                                            + len(local_disc_inputs_list)) / float(
        len(tot_inputs)) * 100)+"\n")
    f.write('Total time:' + str(end - start)+"\n"+"\n")
    f.close()

    global_path = './results/{}-{}-{}-globallist{}'.format(dataset,sens_param,model_name)
    global_path2 = './results/{}-{}-{}-globalkey{}'.format(dataset,sens_param,model_name)
    np.save(global_path, seed_list)
    np.save(global_path2, seeds_key)


    print(seeds_key)
    print("")
    print("Starting Local Search")

    ga = GA()
    start = time.time()
    count = 3600
    for i in range(len(seed_list)):
        inp = seed_list[i]
        sens_key = seeds_key[i]
        ga(model, inp, tokenizer, max_local, evaluate_local, sens_key,dataset)
        try:
            ga(model, inp, tokenizer, max_local, evaluate_local, sens_key, dataset)

            print('length of local discovery is:' + str(len(local_disc_inputs_list)))

            print("Percentage discriminatory inputs - " + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))
        except:
            pass
                                                          
        end = time.time()
        use_time = end-start
        if use_time >= count:

            f = open(file_name,"a")
            f.write("Percentage discriminatory inputs - " + str(
                float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                / float(len(tot_inputs)) * 100)+"\n")
            f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list))+"\n")
            f.write('use time:' + str(end - start)+"\n"+"\n")
            f.close()

            print("Percentage discriminatory inputs - " + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))

            print('use time:' + str(end - start))
            count += 600

    f = open(file_name,"a")
    f.write("Percentage discriminatory inputs - " + str(
        float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
        / float(len(tot_inputs)) * 100)+"\n")
    f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list))+"\n")
    f.write('use time:' + str(end - start)+"\n"+"\n")
    f.close()

    print("Local Search Finished")
    print('length of total input is:' + str(len(tot_inputs)))

    print('length of local discovery is:' + str(len(local_disc_inputs_list)))
    print("Percentage discriminatory inputs - " + str(float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))



def main(argv=None):
    FLAGS = flags.FLAGS

    start = time.time()
    xai_fair_testing(dataset=FLAGS.dataset,
                    sens_param=FLAGS.sens_param,
                    model_path=FLAGS.model_path,
                    max_global=FLAGS.max_global,
                    max_local=FLAGS.max_local)
    end = time.time()
    print('total time:' + str(end - start))


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "sst", "the name of dataset")
    flags.DEFINE_string('sens_param', "country", 'sensitive index')
    flags.DEFINE_string('model_path', "../models/nlp/sst_word_cnn.h5", 'the path for testing model')
    flags.DEFINE_integer('max_global', 1000, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 20, 'maximum number of samples for local search')
    main()