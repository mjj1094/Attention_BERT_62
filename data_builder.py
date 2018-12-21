#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import codecs
import random
from subprocess import *

from conf import *
from buildTree import get_info_from_file
import utils
import json
import logging
from data_generater import *
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
print("PID", os.getpid(), file=sys.stderr)
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.set_device(args.gpu)
import pickle
sys.setrecursionlimit(1000000)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
MAX = 2

def get_sentence(zp_sentence_index,zp_index,nodes_info):
    nl,wl = nodes_info[zp_sentence_index]
    return_words = []
    for i in range(len(wl)):
        this_word = wl[i].word
        if i == zp_index:
            return_words.append("**pro**")
        else:
            if not (this_word == "*pro*"): 
                return_words.append(this_word)
    return " ".join(return_words)

def get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result):
    nl,wl = nodes_info[candi_sentence_index]
    candi_word = []
    for i in range(candi_begin,candi_end+1):
        candi_word.append(wl[i].word)
    candi_word = "_".join(candi_word)

    candi_info = [str(res_result),candi_word]
    return candi_info

def setup():
    utils.mkdir(args.data)
    utils.mkdir(args.data+"train/")
    utils.mkdir(args.data+"train_reduced/")
    utils.mkdir(args.data+"test/")
    utils.mkdir(args.data+"test_reduced/")

def list_vectorize(wl,words):
    il = []
    for w in wl:
        word = w.word
        if word in words:
            # index = words.index(word)
            index = words[word]
        else:
            index = 0
        il.append(index) 
    return il

def generate_vector(path,files):
    read_f = open(args.data+"emb", "rb")
    _, _, wd = pickle.load(read_f, encoding='latin1')
    read_f.close()
    f = open(args.data+'vocab_attention.json', 'r')
    words=json.load(f)
    f.close()

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir+'vocab.txt')
    orig_to_tok_maps_bert = []
    vectorized_sentences_bert = []
    mask_sentences_bert = []

    paths = [w.strip() for w in open(files).readlines()]
    #paths = utils.get_file_name(path,[])
    total_sentence_num = 0
    vectorized_sentences = []

    zp_info = []

    startt = timeit.default_timer()
    done_num = 0
    for p in paths:
        if p.strip().endswith("DS_Store"):continue
        done_num += 1
        file_name = args.data + p.strip()
        if file_name.endswith("onf"):

            if args.reduced == 1 and done_num >= 3:break

            zps,azps,candi,nodes_info = get_info_from_file(file_name,2)
            anaphorics = []
            ana_zps = []
            for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
                for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                    ana_zps.append((zp_sentence_index,zp_index))

            si2reali = {}
            for k in nodes_info:
                nl,wl = nodes_info[k]
                vectorize_words = list_vectorize(wl,words)
                vectorized_sentences.append(vectorize_words)                
                bert_tokens = []
                orig_to_tok_map = []
                orig_tokens=[w.word for w in wl]
                bert_tokens.append("[CLS]")
                for i, orig_token in enumerate(orig_tokens):
                    orig_to_tok_map.append(len(bert_tokens))
                    if "*pro*"in orig_token:
                        bert_tokens.extend(["[MASK]"])
                    else:
                        bert_tokens.extend(tokenizer.tokenize(orig_token))
                bert_tokens.append("[SEP]")

                orig_to_tok_maps_bert.append(orig_to_tok_map)
                indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
                max_index_bert = len(indexed_tokens)
                indexed_tokens=indexed_tokens[:min(args.max_sent_len,max_index_bert)]
                sent_bert_mask = (len(indexed_tokens) * [1] + (args.max_sent_len - len(indexed_tokens)) * [0])
                indexed_tokens = (indexed_tokens  + (args.max_sent_len - len(indexed_tokens)) * [0])
                vectorized_sentences_bert.append(indexed_tokens)
                mask_sentences_bert.append(sent_bert_mask)

                si2reali[k] = total_sentence_num
                total_sentence_num += 1

            for (sentence_index,zp_index) in zps:
                ana = 0
                if (sentence_index,zp_index) in ana_zps:
                    ana = 1
                index_in_file = si2reali[sentence_index]
                zp = (index_in_file,sentence_index,zp_index,ana)
                zp_nl,zp_wl = nodes_info[sentence_index]

                candi_info = []
                if ana == 1:
                    for ci in range(max(0,sentence_index-2),sentence_index+1):
                        candi_sentence_index = ci  
                        candi_nl,candi_wl = nodes_info[candi_sentence_index]

                        for (candi_begin,candi_end) in candi[candi_sentence_index]:
                            if ci == sentence_index and candi_end > zp_index:
                                continue
                            res = 0
                            if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                                res = 1
                            candi_index_in_file = si2reali[candi_sentence_index]

                            ifl = get_fl((sentence_index,zp_index),(candi_sentence_index,candi_begin,candi_end),zp_wl,candi_wl,wd)

                            candidate = (candi_index_in_file,candi_sentence_index,candi_begin,candi_end,res,-res,ifl)
                            candi_info.append(candidate)
                zp_info.append((zp,candi_info))

    endt = timeit.default_timer()
    print(file=sys.stderr)
    print("Total use %.3f seconds for Data Generating"%(endt-startt), file=sys.stderr)
    vectorized_sentences = numpy.array(vectorized_sentences)
    vectorized_sentences_bert = numpy.array(vectorized_sentences_bert)
    mask_sentences_bert = numpy.array(mask_sentences_bert)
    orig_to_tok_maps_bert = numpy.array(orig_to_tok_maps_bert)
    return zp_info,vectorized_sentences,vectorized_sentences_bert,orig_to_tok_maps_bert,mask_sentences_bert

def generate_vector_data(file_name="",test_only=False):
    DATA = args.data+'zp_data/'
    train_data_path = args.data + "train/"+file_name
    test_data_path = args.data + "test/"+file_name
    if args.reduced == 1:
        train_data_path = args.data + "train_reduced/"+file_name
        test_data_path = args.data + "test_reduced/"+file_name

    if not test_only:
        train_zp_info, train_vectorized_sentences,train_vectorized_sentences_bert,train_orig_to_tok_maps_bert,train_mask_sentences_bert = generate_vector(DATA+"train/"+file_name,args.data+"train_list")
        train_vec_path = train_data_path + "sen.npy"
        numpy.save(train_vec_path,train_vectorized_sentences)
        train_vec_path = train_data_path + "sen_bert.npy"
        numpy.save(train_vec_path, train_vectorized_sentences_bert)
        train_vec_path = train_data_path + "sen_mask_bert.npy"
        numpy.save(train_vec_path, train_mask_sentences_bert)
        train_vec_path = train_data_path + "orig_to_tok_bert.npy"
        numpy.save(train_vec_path, train_orig_to_tok_maps_bert)


        save_f = open(train_data_path + "zp_info", 'wb')
        pickle.dump(train_zp_info, save_f, protocol=pickle.HIGHEST_PROTOCOL)
        save_f.close()

    test_zp_info, test_vectorized_sentences,test_vectorized_sentences_bert,test_orig_to_tok_maps_bert,test_mask_sentences_bert = generate_vector(DATA+"test/"+file_name,args.data+"test_list")
    test_vec_path = test_data_path + "sen.npy"
    numpy.save(test_vec_path,test_vectorized_sentences)

    test_vec_path = test_data_path + "sen_bert.npy"
    numpy.save(test_vec_path, test_vectorized_sentences_bert)
    test_vec_path = test_data_path + "sen_mask_bert.npy"
    numpy.save(test_vec_path, test_mask_sentences_bert)
    test_vec_path = test_data_path + "orig_to_tok_bert.npy"
    numpy.save(test_vec_path, test_orig_to_tok_maps_bert)


    save_f = open(test_data_path + "zp_info", 'wb')
    pickle.dump(test_zp_info, save_f, protocol=pickle.HIGHEST_PROTOCOL)
    save_f.close()
def generate_input_data(file_name="",test_only=False):
     
    DATA = args.raw_data

    train_data_path = args.data + "train/"+file_name
    test_data_path = args.data + "test/"+file_name
    if args.reduced == 1:
        train_data_path = args.data + "train_reduced/"+file_name
        test_data_path = args.data + "test_reduced/"+file_name

    if not test_only:
        generate_vec(train_data_path)
    generate_vec(test_data_path)

def generate_vec(data_path):

    zp_candi_target = []
    zp_vec_index = 0
    candi_vec_index = 0

    zp_sent_bert=[]
    zp_sent_mask_bert=[]
    zp_orig_to_tok_bert=[]

    # zp_prefixs = []
    # zp_prefixs_mask = []
    # zp_postfixs = []
    # zp_postfixs_mask = []
    # candi_vecs = []
    # candi_vecs_mask = []
    ifl_vecs = []
    
    read_f = open(data_path + "zp_info","rb")
    zp_info_test = pickle.load(read_f)
    read_f.close()

    # vectorized_sentences = numpy.load(data_path + "sen.npy")
    vectorized_sentences_bert = numpy.load(data_path + "sen_bert.npy")
    vectorized_sentences_mask_bert = numpy.load(data_path + "sen_mask_bert.npy")
    orig_to_tok_bert=numpy.load(data_path + "orig_to_tok_bert.npy")
    for zp,candi_info in zp_info_test:
        index_in_file, sentence_index, zp_index, ana = zp
        if ana == 1:
            # word_embedding_indexs = vectorized_sentences[index_in_file]
            # max_index = len(word_embedding_indexs)

            # prefix = word_embedding_indexs[max(0, zp_index - 10):zp_index]
            # prefix_mask = (10 - len(prefix)) * [0] + len(prefix) * [1]
            # prefix = (10 - len(prefix)) * [0] + prefix
            #
            # zp_prefixs.append(prefix)
            # zp_prefixs_mask.append(prefix_mask)
            #
            # postfix = word_embedding_indexs[zp_index + 1:min(zp_index + 11, max_index)]
            # postfix_mask = (len(postfix) * [1] + (10 - len(postfix)) * [0])[::-1]
            # postfix = (postfix + (10 - len(postfix)) * [0])[::-1]#[0,0,0,..,1,1,1]
            #
            # zp_postfixs.append(postfix)
            # zp_postfixs_mask.append(postfix_mask)

            # word_embedding_indexs_bert = vectorized_sentences_bert[index_in_file]
            # max_index_bert = len(word_embedding_indexs_bert)
            # sent_bert=word_embedding_indexs_bert[:min(100,max_index_bert)]
            # sent_bert_mask = (len(sent_bert) * [1] + (100 - len(sent_bert)) * [0])
            # sent_bert = (sent_bert + (100 - len(sent_bert)) * [0])

            sent_bert = vectorized_sentences_bert[index_in_file]
            sent_bert_mask = vectorized_sentences_mask_bert[index_in_file]#no best,直接计算比较好

            zp_sent_bert.append(sent_bert)
            zp_sent_mask_bert.append(sent_bert_mask)
            zp_orig_to_tok_bert.append(orig_to_tok_bert[index_in_file])

            candi_vec_index_inside = []
            for candi_index_in_file,candi_sentence_index,candi_begin,candi_end,res,target,ifl in candi_info:
                # candi_word_embedding_indexs = vectorized_sentences[candi_index_in_file]
                # candi_vec = candi_word_embedding_indexs[candi_begin:candi_end+1]
                # if len(candi_vec) >= 8:#限制candi长度max为8
                #     candi_vec = candi_vec[-8:]
                # candi_mask = (8-len(candi_vec))*[0] + len(candi_vec)*[1]
                # candi_vec = (8-len(candi_vec))*[0] + candi_vec
                #
                # candi_vecs.append(candi_vec)
                # candi_vecs_mask.append(candi_mask)

                ifl_vecs.append(ifl)
                
                candi_vec_index_inside.append((candi_vec_index,res,target))

                candi_vec_index += 1

            zp_candi_target.append((zp_vec_index,candi_vec_index_inside)) #(zpi,candis(candij,res,target))

            zp_vec_index += 1
    save_f = open(data_path + "zp_candi_pair_info", 'wb')
    pickle.dump(zp_candi_target, save_f, protocol=pickle.HIGHEST_PROTOCOL)
    save_f.close()
    
    # zp_prefixs = numpy.array(zp_prefixs,dtype='int32')
    # numpy.save(data_path+"zp_pre.npy",zp_prefixs)
    # zp_prefixs_mask = numpy.array(zp_prefixs_mask,dtype='int32')
    # numpy.save(data_path+"zp_pre_mask.npy",zp_prefixs_mask)
    # zp_postfixs = numpy.array(zp_postfixs,dtype='int32')
    # numpy.save(data_path+"zp_post.npy",zp_postfixs)
    # zp_postfixs_mask = numpy.array(zp_postfixs_mask,dtype='int32')
    # numpy.save(data_path+"zp_post_mask.npy",zp_postfixs_mask)

    zp_sent_bert = numpy.array(zp_sent_bert, dtype='int32')
    numpy.save(data_path + "zp_sent_bert.npy", zp_sent_bert)
    zp_sent_mask_bert = numpy.array(zp_sent_mask_bert, dtype='int32')
    numpy.save(data_path + "zp_sent_mask_bert.npy", zp_sent_mask_bert)

    # zp_orig_to_tok_bert = numpy.array(zp_orig_to_tok_bert, dtype='int32')
    zp_orig_to_tok_bert = numpy.array(zp_orig_to_tok_bert)
    numpy.save(data_path + "zp_orig_to_tok_bert.npy", zp_orig_to_tok_bert)

    # candi_vecs = numpy.array(candi_vecs,dtype='int32')
    # numpy.save(data_path+"candi_vec.npy",candi_vecs)
    # candi_vecs_mask = numpy.array(candi_vecs_mask,dtype='int32')
    # numpy.save(data_path+"candi_vec_mask.npy",candi_vecs_mask)

    # assert len(ifl_vecs) == len(candi_vecs)

    ifl_vecs = numpy.array(ifl_vecs,dtype='float')
    numpy.save(data_path+"ifl_vec.npy",ifl_vecs)
  
def get_head_verb(index,wl):
    father = wl[index].parent
    while father:
        leafs = father.get_leaf()
        for ln in leafs:
            if ln.tag.startswith("V"):
                return ln
        father = father.parent

    return None

def build_zero_one(index,num):
    tmp_ones = [0]*num
    tmp_ones[index] = 1
    return tmp_ones
    
def get_fl(zp,candidate,wl_zp,wl_candi,wd):

    ifl = []

    (zp_sentence_index,zp_index) = zp
    (candi_sentence_index,candi_index_begin,candi_index_end) = candidate

    zp_node = wl_zp[zp_index]

    sentence_dis = zp_sentence_index - candi_sentence_index

    tmp_ones = [0]*3
    tmp_ones[sentence_dis] = 1 
    ifl += tmp_ones
   
    cloNP = 0
    if sentence_dis == 0:
        if candi_index_end <= zp_index:
            cloNP = 1
        for i in range(candi_index_end+1,zp_index):
            node = wl_zp[i]
            while True:
                if node.tag.startswith("NP"):
                    cloNP = 0
                    break
                node = node.parent
                if not node:
                    break
            if cloNP == 0:
                break

    tmp_ones = [0]*2
    tmp_ones[cloNP] = 1
    ifl += tmp_ones

    NP_node = None
    father = zp_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    z_NP = 0
    if NP_node:
        z_NP = 1
    tmp_ones = [0]*2
    tmp_ones[z_NP] = 1
    ifl += tmp_ones
    z_NinI = 0
    if NP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(NP_node):
                    z_NinI = 1
                break
            father = father.parent

    tmp_ones = [0]*2
    tmp_ones[z_NinI] = 1
    ifl += tmp_ones
    VP_node = None
    zVP = 0
    father = zp_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            zVP = 1
            break
        father = father.parent
    tmp_ones = [0]*2
    tmp_ones[zVP] = 1
    ifl += tmp_ones
    z_VinI = 0
    if VP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(VP_node):
                    z_VinI = 1
                break
            father = father.parent
    tmp_ones = [0]*2
    tmp_ones[z_VinI] = 1
    ifl += tmp_ones
    CP_node = None
    zCP = 0
    father = zp_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            zCP = 1
            break
        father = father.parent

    tmp_ones = [0]*2
    tmp_ones[zCP] = 1
    ifl += tmp_ones
    tags = zp_node.parent.tag.split("-") 
    zGram = 0
    zHl = 0
    if len(tags) == 2:
        if tags[1] == "SBJ":
            zGram = 1
        if tags[1] == "HLN":
            zHl = 1
    tmp_ones = [0]*2
    tmp_ones[zGram] = 1
    ifl += tmp_ones
    tmp_ones = [0]*2
    tmp_ones[zHl] = 1
    ifl += tmp_ones
    first_zp = 1
    for i in range(zp_index):
        if wl_zp[i].word == "*pro*":
            first_zp = 0
            break
    tmp_ones = [0]*2
    tmp_ones[first_zp] = 1
    ifl += tmp_ones
    last_zp = 1
    for i in range(zp_index+1,len(wl_zp)):
        if wl_zp[i].word == "*pro*":
            last_zp = 0
            break
    tmp_ones = [0]*2
    tmp_ones[last_zp] = 1
    ifl += tmp_ones
    zc = 0
    if zCP == 1:
        zc = 1
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                zc = 2
                break
            if father == CP_node:
                break
            father = father.parent 
    else:
        zc = 3
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent: #非根节点
                    zc = 4
                    break
            father = father.parent
    tmp_ones = [0]*5
    tmp_ones[zc] = 1
    ifl += tmp_ones
    candi_node = wl_candi[candi_index_begin]
    father_candi = candi_node.parent.tag
    NP_node = None
    father = candi_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    can_NinI = 0
    if NP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(NP_node):
                    can_NinI = 1
                break
            father = father.parent
    tmp_ones = [0]*2
    tmp_ones[can_NinI] = 1
    ifl += tmp_ones
    VP_node = None
    canVP = 0
    father = candi_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            canVP = 1
            break
        father = father.parent
    tmp_ones = [0]*2
    tmp_ones[canVP] = 1
    ifl += tmp_ones    
    can_VinI = 0
    if VP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(VP_node):
                    can_VinI = 1
                break
            father = father.parent
    tmp_ones = [0]*2
    tmp_ones[can_VinI] = 1
    ifl += tmp_ones
    CP_node = None
    canCP = 0
    father = candi_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            canCP = 1
            break
        father = father.parent
    tmp_ones = [0]*2
    tmp_ones[canCP] = 1
    ifl += tmp_ones
    tags = candi_node.parent.tag.split("-") 
    canGram = 0
    canADV = 0
    canTMP = 0
    canPN = 0
    canHl = 0
    if len(tags) == 2:
        if tags[1] == "SBJ":
            canGram = 1
        elif tags[1] == "OBJ":
            canGram = 2
        if tags[1] == "ADV":
            canADV = 1
        if tags[1] == "TMP":
            canTMP = 1
        if tags[1] == "PN":
            canPN = 1
        if tags[1] == "HLN":
            canHl = 1
    tmp_ones = [0]*3
    tmp_ones[canGram] = 1
    ifl += tmp_ones
    tmp_ones = [0]*2
    tmp_ones[canADV] = 1
    ifl += tmp_ones
    tmp_ones = [0]*2
    tmp_ones[canTMP] = 1
    ifl += tmp_ones
    tmp_ones = [0]*2
    tmp_ones[canPN] = 1
    ifl += tmp_ones
    tmp_ones = [0]*2
    tmp_ones[canHl] = 1
    ifl += tmp_ones
    canc = 0
    if canCP == 1:
        canc = 1
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                canc = 2
                break
            if father == CP_node:
                break
            father = father.parent 
    else:
        canc = 3
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent:
                    canc = 4
                    break
            father = father.parent
    tmp_ones = [0]*5
    tmp_ones[canc] = 1
    ifl += tmp_ones
    sibNV = 0
    if not sentence_dis == 0:
        sibNV = 0
    else:
        if abs(zp_index - candi_index_end) == 1:
            sibNV = 1
        else:
            if abs(zp_index - candi_index_begin) == 1:
                sibNV = 1
            else:
                if abs(zp_index - candi_index_begin) == 2:
                    if zp_index < candi_index_begin:
                        if wl_zp[zp_index+1].tag == "PU":
                            sibNV = 1
                elif abs(zp_index-candi_index_end) == 2:
                    if candi_index_end < zp_index:
                        if wl_zp[zp_index-1].tag == "PU":
                            sibNV = 1
    tmp_ones = [0]*2
    tmp_ones[sibNV] = 1
    ifl += tmp_ones
    gram_match = 0
    if not canGram == 0:
        if canGram == zGram:
            gram_match = 1
    tmp_ones = [0]*2
    tmp_ones[gram_match] = 1
    ifl += tmp_ones
    chv = get_head_verb(candi_index_begin,wl_candi)
    zhv = get_head_verb(zp_index,wl_zp)
    ch = wl_candi[candi_index_end]
    hc = "None"
    pc = "None"
    pz = "None"
    if ch:
        hc = ch.word
    if zhv:
        pz = zhv.word
    if chv:
        pc = chv.word
    tags = candi_node.parent.tag.split("-")
    canGram = "None"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            canGram = "SBJ"
        elif tags[1] == "OBJ":
            canGram = "OBJ"
    gc = canGram
    pcc = "None"
    for i in range(len(wl_zp)-1,zp_index,-1):
        if wl_zp[i].tag.find("PU") >= 0:
            pcc = wl_zp[i].word
            break 
    pc_pz = 0
    has = wd["%s_%s"%(hc,pcc)]
    if pc == pz:
        if canGram == "SBJ":
            pc_pz = 1
        elif canGram == "OBJ":
            pc_pz = 1
        else:
            pc_pz = 2
    tmp_ones = [0]*3
    tmp_ones[pc_pz] = 1
    ifl += tmp_ones
    tmp_ones = [0]*2
    tmp_ones[has] = 1
    ifl += tmp_ones
    return ifl

def generate_vec_bert(data_path,batch_size):
    zp_vec_index = 0
    zp_sent_cls_output_bert = []
    # zp_index_bert=[]
    candi_vecs_bert=[]
    candi_vecs_mask_bert=[]

    read_f = open(data_path + "zp_info", "rb")
    zp_info_test = pickle.load(read_f)
    read_f.close()

    orig_to_tok_bert = numpy.load(data_path + "orig_to_tok_bert.npy")
    f=open(data_path + "sent_output_bert.npy",'rb')
    vectorized_sentences_bert = numpy.load(f)
    old_num=0
    # max_zp_index=0
    for zp, candi_info in zp_info_test:
        index_in_file, sentence_index, zp_index, ana = zp
        if ana == 1:
            zp_index=orig_to_tok_bert[index_in_file][zp_index]

            # if zp_index > max_zp_index:
            #     max_zp_index = zp_index
            while((index_in_file-old_num) >= batch_size):
                vectorized_sentences_bert = numpy.load(f)
                old_num+=batch_size

            sent_cls_output_bert = [round(x.item(), 6) for x in vectorized_sentences_bert[index_in_file - old_num][zp_index]]
            zp_sent_cls_output_bert.append(sent_cls_output_bert)

            for candi_index_in_file,candi_sentence_index,candi_begin,candi_end,res,target,ifl in candi_info:
                # candi_word_embedding_indexs_bert = [round(x.item(), 6) for x in vectorized_sentences_bert[candi_index_in_file - old_num]]
                candi_word_embedding_indexs_bert = vectorized_sentences_bert[candi_index_in_file - old_num]
                candi_orig_to_tok = orig_to_tok_bert[candi_index_in_file]
                if candi_end+1>=len(candi_orig_to_tok):
                    candi_end_bert=candi_orig_to_tok[candi_end]+1
                else:
                    candi_end_bert = candi_orig_to_tok[candi_end + 1]
                candi_vec_bert = candi_word_embedding_indexs_bert[candi_orig_to_tok[candi_begin]:candi_end_bert][:,:256]
                if len(candi_vec_bert) >= 40:#限制candi长度max为8*5
                    candi_vec_bert = candi_vec_bert[-40:]
                candi_mask_bert = (40-len(candi_vec_bert))*[0] + len(candi_vec_bert)*[1]
                # candi_vec_bert = (40-len(candi_vec_bert))*[0] + candi_vec_bert
                # candi_vec_bert = numpy.concatenate((numpy.zeros((40 - len(candi_vec_bert), 768)), candi_vec_bert), axis=0)
                candi_vec_bert = numpy.concatenate((numpy.zeros((40 - len(candi_vec_bert), 256)), candi_vec_bert), axis=0)
                candi_vecs_bert.append(candi_vec_bert)
                candi_vecs_mask_bert.append(candi_mask_bert)

            zp_vec_index += 1

    zp_sent_cls_output_bert = numpy.array(zp_sent_cls_output_bert, dtype='float')#--------------
    numpy.save(data_path + "zp_sent_cls_output_bert.npy", zp_sent_cls_output_bert)
    candi_vecs_bert = numpy.array(candi_vecs_bert,dtype='float')
    numpy.save(data_path+"candi_vec_bert.npy",candi_vecs_bert)
    candi_vecs_mask_bert = numpy.array(candi_vecs_mask_bert,dtype='float')
    numpy.save(data_path+"candi_vec_mask_bert.npy",candi_vecs_mask_bert)


#获取sent的all tokens 的vector
def get_bert_out(output_path,local_rank,no_cuda,batch_size):
    startt = timeit.default_timer()

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(local_rank != -1)))

    model = BertModel.from_pretrained(args.bert_dir)
    model.to(device)
    # model.to(0)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    sent_bert = numpy.load(output_path + "sen_bert.npy")
    sent_mask_bert = numpy.load(output_path + "sen_mask_bert.npy")

    f = open(output_path+"sent_output_bert.npy", 'ab')
    num=0
    all_input_ids = torch.tensor(sent_bert, dtype=torch.int64).to(device)
    all_input_mask = torch.tensor(sent_mask_bert, dtype=torch.int64).to(device)
    all_example_index=torch.tensor(list(range(len(sent_bert))), dtype=torch.int64).to(device)
    eval_data = TensorDataset(all_input_ids, all_input_mask,all_example_index)
    if local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    for input_ids, input_mask,example_indices in eval_dataloader:
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers
        num+=len(sent_bert)
        outs = []
        for b, example_index in enumerate(example_indices):
            layer_output = all_encoder_layers[-1].detach().cpu().numpy()  # last layer
            layer_output = layer_output[b][:,:512] # sent b
            # out = [round(x.item(), 6) for x in layer_output[0]]  # [CLS]
            # outs.append(out)
            outs.append(layer_output)# all tokens-----------------
        outs = numpy.array(outs)
        numpy.save(f, outs)

    endt = timeit.default_timer()
    print(file=sys.stderr)
    print("Total use %.3f seconds for BERT Data Generating" % (endt - startt), file=sys.stderr)



if __name__ == "__main__":
    # build data from raw OntoNotes data
    setup()
    generate_vector_data()
    generate_input_data()
    # split training data into dev and train, saved in ./data/train_data
    get_bert_out(args.data+'train/',args.local_rank,args.no_cuda,args.data_batch_size)
    generate_vec_bert(args.data+'train/', args.data_batch_size)
    get_bert_out(args.data+'test/',args.local_rank,args.no_cuda,args.data_batch_size)
    generate_vec_bert(args.data+'test/', args.data_batch_size)
    train_generater = DataGnerater("train",nnargs["batch_size"])
    train_generater.devide()
    save_f = open(args.data+'train_data','wb')
    pickle.dump(train_generater, save_f, protocol=pickle.HIGHEST_PROTOCOL)
    save_f.close()




