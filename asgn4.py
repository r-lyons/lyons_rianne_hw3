
import dynet as dy
import numpy as np
import pickle as pkl
import random

def make_vocab_index(file_list, path_to_embeddings):
    """
    Creates a training word-to-embedding mapping. Reads
    in words from a list of training files (tokenized) 
    and matches them with their embeddings from PubMed.
    Saves the resulting vocabulary in a pickle file in the same directory.
    @params: file_list is a list of file paths to extract words,
             path_to_embeddings is the path to the embeddings file.
    """

    #init vocab
    vocab = {}

    #open file, read tokens
    for file_path in file_list:
      print('extracting vocab from file')
      with open(file_path, 'r') as f:
        for token in f.readline().split(' '):
          if token not in vocab.keys():
            vocab[token] = [-1]
          else:
            continue
 
    #add embeddings for vocab
    with open(path_to_embeddings, 'r') as e:
      for line in e:
        entry = line.split(' ')
        if entry[0] in vocab.keys():
          vocab[entry[0]] = np.array([float(num) for num in entry[1:]])     
        else:
          continue
      
    #for wd in vocab.keys():
      #if vocab[wd][0] == -1:
        #del vocab[wd]
    voc_dict = {wd: emb for wd,emb in vocab.items() if emb[0] != -1}
    
    voc_dict['UNK'] = np.random.uniform(-0.4, 0.4, 200)

    save_path = 'vocab.pickle'
    with open(save_path, 'wb') as v:
      pkl.dump(voc_dict, v)



def file_to_embedding(file_path, vocab):
    """
    Uses the vocab mapping to convert a file into a list of word 
    embeddings. 
    @params: file_path is the path for the file to convert; it may not
              have numbers (except for the document ID),
             vocab is the vocabulary mapping from a word to an embedding
    @returns: embed_file is a numpy array of word embeddings
    """

    doc_id = file_path.lstrip('abcdefghijklmnopqrstuvwxyz/-_ABCDEFGHIJKLMNOPQRSTUVWXYZ').partition('.')[0]
    #doc_id = file_path.lstrip('[aA-zZ]-_/').partition('.')[0]

    with open(file_path, 'r') as f:
      embed_file = np.array([vocab[token] if token in vocab.keys() else vocab['UNK'] for token in f.readline().split(' ')])

    return embed_file
    
#def run_one_doc(model, emb_doc, doc_labels, w_param, b_param):
def run_one_doc(model, first_level, emb_doc, doc_labels, w_param, b_param):
    """
    Runs the given model on one document and makes predictions.
    @params: first_level is I, O, or P,
             model is the LSTM model,
             emb_doc is a numpy array of embeddings for one document,
             doc_labels is a list of the labels associated with emb_doc,
             w_param is a Dynet parameter multiplied with the layer output,
             b_param is a Dynet parameter added to the product of output and w_param.
    @returns: pred_gold is a list of tuples in the form of (prediction, gold label)
    """
    dy.renew_cg()
    s = model.initial_state()
    i = dy.vecInput(200)
    o = dy.vecInput(200)
    p = dy.vecInput(200)
    si = s.add_input(i)
    so = s.add_input(o)
    sp = s.add_input(p)
    pred_gold = []
    
    for wdemb, label in zip(emb_doc, doc_labels):
      x = dy.inputVector(wdemb)
      if first_level == 'I':
        s2 = si.add_input(x)
      elif first_level == 'O':
        s2 = so.add_input(x)
      else:
        s2 = sp.add_input(x)
      out_class = dy.softmax((w_param*s2.output())+b_param)
      chosen_class = np.argmax(out_class.npvalue())
      pred_gold.append((int(chosen_class), int(label)))
    """
    for first_level, docs in data_dict.items():
      for wdemb, label in docs:
        x = dy.inputVector(wdemb)
        if first_level == 'I':
          s2 = si.add_input(x)
        elif first_level == 'O':
          s2 = so.add_input(x)
        else:
          s2 = sp.add_input(x)
        out_class = dy.softmax((w_param*s2.output())+b_param)
        chosen_class = np.argmax(out_class.npvalue())
        pred_gold.append((int(chosen_class), int(label)))
    
    for wdemb, label in zip(emb_doc, doc_labels):
      x = dy.inputVector(wdemb)
      s = s.add_input(x)
      out_class = dy.softmax((w_param*s.output())+b_param)
      chosen_class = np.argmax(out_class.npvalue())
      pred_gold.append((int(chosen_class), int(label)))
    """
    return pred_gold

#def build_model(emb_doc, doc_labels, model, w_param, b_param):
def build_model(first_level, model, emb_doc, doc_labels, w_param, b_param):
    """
    Runs the model for training, calculating the loss. 
    @params: first_level is I, O, or P,
             model is the LSTM model,
             emb_doc is a numpy array of embeddings for one document,
             doc_labels is a list of the labels associated with emb_doc,
             w_param is a Dynet parameter multiplied with the layer output,
             b_param is a Dynet parameter added to the product of output and w_param.
    @returns: the sum of the errors computed for the document
    """
    dy.renew_cg()
    s = model.initial_state()
    i = dy.vecInput(200)
    o = dy.vecInput(200)
    p = dy.vecInput(200)
    si = s.add_input(i)
    so = s.add_input(o)
    sp = s.add_input(p)
    loss = []
    
    for wdemb, label in zip(emb_doc, doc_labels):
      x = dy.inputVector(wdemb)
      if first_level == 'I':
        s2 = si.add_input(x)
      elif first_level == 'O':
        s2 = so.add_input(x)
      else:
        s2 = sp.add_input(x)
      loss.append(dy.pickneglogsoftmax((w_param*s2.output())+b_param,label))
    return dy.esum(loss)  
    """
    for first_level, docs in data_dict.items():
      for wdemb, label in docs:
        x = dy.inputVector(wdemb)
        if first_level == 'I':
          s2 = si.add_input(x)
        elif first_level == 'O':
          s2 = so.add_input(x)
        else:
          s2 = sp.add_input(x)
        loss.append(dy.pickneglogsoftmax((w_param*s2.output())+b_param,label))
    return dy.esum(loss)
    
    for wdemb, label in zip(emb_doc, doc_labels):
      x = dy.inputVector(wdemb)
      s = s.add_input(x)
      loss.append(dy.pickneglogsoftmax((w_param*s.output())+b_param,label))
    return dy.esum(loss)
    """
          

def main():

    train_paths = ['train/P/88754.tokens','train/I/3277760.tokens','train/O/352099.tokens','train/P/350565.tokens','train/O/1336890.tokens','train/I/8215273.tokens','train/I/43164.tokens','train/P/3137065.tokens']
    dev_paths = ['dev/I/16603337','dev/I/43164','dev/O/43164','dev/O/1300984','dev/P/7218018']
    test_paths = []
    #make vocab (only once)
    #make_vocab_index(train_paths, 'PubMed-w2v.txt')
    
    #load vocab
    with open('vocab.pickle', 'rb') as v:
      vocab_idx = pkl.load(v)
      
    #get embeddings to train
    train_doc_embeddings = [file_to_embedding(path, vocab_idx) for path in train_paths]
    
    #get dev embeddings
    dev_doc_embeddings = [file_to_embedding(path+'.tokens', vocab_idx) for path in dev_paths]
    
    hier_train_data = {'I':[],'O':[],'P':[]}
    hier_dev_data = {'I':[],'O':[],'P':[]}
    
    #get labels
    train_doc_labels = []
    for p in train_paths:
      doc_id = p.lstrip('train/').rstrip('.tokens')
      with open('train/'+doc_id+'_AGGREGATED.ann', 'r') as l:
        labels = [float(label) for label in l.readline().split(',')]
        train_doc_labels.append(labels)
        hier_train_data[doc_id[0]].append((train_doc_embeddings[train_paths.index(p)], labels))
        
    dev_doc_labels = []
    for pd in dev_paths:
      with open(pd+'_AGGREGATED.ann', 'r') as l:
        labels = [float(label) for label in l.readline().split(',')]
        dev_doc_labels.append(labels)
        hier_dev_data[pd[4]].append((dev_doc_embeddings[dev_paths.index(pd)], labels))
        
        
    train_data = zip(train_doc_embeddings,train_doc_labels)
    #train_data = {emb_doc: label_doc for e,l in zip(train_doc_embeddings,train_doc_labels)}
    pc = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(pc)
    lstm = dy.LSTMBuilder(1, 200, 80, pc)
    W = pc.add_parameters((8, 80))
    B = pc.add_parameters((8))
    
    for i in range(5):
      #train_data = random.shuffle(list(zip(train_doc_embeddings,train_doc_labels))) 
      #random.shuffle(train_data.keys())
      #print(train_data)
      for first_level, docs in hier_train_data.items():
        for instance, gold in docs:
          output_gold = run_one_doc(lstm, first_level, instance, gold, W, B)
      #for instance, gold in train_data:
        #output_gold = run_one_doc(lstm, hier_train_data, W, B)
        #output_gold = run_one_doc(lstm, instance, gold, W, B)
        #print(output_gold)
        #do calculations
      loss = build_model(first_level, lstm, instance, gold, W, B)
      loss.backward()
      trainer.update()
      
      
    for first_level, docs in hier_dev_data.items():
      for instance, gold in docs:
        dev_predgold = run_one_doc(lstm, first_level, instance, gold, W, B)
      
    #for doc,gold in zip(dev_doc_embeddings,dev_doc_labels):
      #dev_predgold = run_one_doc(lstm,doc,gold,W,B)
      print(dev_predgold)
      
      
    #save model
    dy.save("model",[lstm, W, B])
    #load model
    pc = dy.ParameterCollection()
    lstm_load, W_load, B_load = dy.load("model", pc)
    
    #test on test data
      


main()
