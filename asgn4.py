"""
Author: Rianne Lyons
Last modified: 11/25/2018
Filename: asgn4.py
CSC 585 assignment 4

Implementation of a hierarchical LSTM with
a mean teacher learning approach for the PICO dataset.
Requires DyNet, as well as the PICO dataset. Uses 
data divided into train/, dev/, and test/ directories in
the same directory as this file. Saves and loads pickle and 
DyNet files (for vocabulary and models).
"""
import dynet as dy
import numpy as np
import pickle as pkl
import random
import math

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
      
    return pred_gold

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
      dy.noise(x, 0.5) #noise for student model
      if first_level == 'I':
        s2 = si.add_input(x)
      elif first_level == 'O':
        s2 = so.add_input(x)
      else:
        s2 = sp.add_input(x)
      loss.append(dy.pickneglogsoftmax((w_param*s2.output())+b_param,label))
    return dy.esum(loss)  
    
def calculate_mse(t_preds, s_preds):
    """
    Calculates the mean squared error for the consistency cost
    between teacher and student predictions.
    @params: t_preds is the list of teacher predictions for a document,
             s_preds is the list of student predictions for a document
    @returns: the mean squared error between teacher and student predictions
    """
    return (1/len(t_preds))* np.sum(np.array([(float(t) - float(s))**2 for t, s in zip(t_preds, s_preds)]))
    

def train(student_lstm, teacher_lstm, W, B, trainer, hier_train_data):
    """
    Trains the model using the mean teacher approach (temporal ensemble). 
    @params: student_lstm is the hierarchical student model (with gaussian noise),
             teacher_lstm is the hierarchical teacher model (with averaged predictions),
             W is a weight parameter,
             B is a bias parameter,
             trainer is the DyNet SGD trainer,
             hier_train_data is a dictionary with keys as I, O, P and values as lists of documents
    """
    consistency_costs = []
    teacher_models = []
    
    for e in range(10): #epochs
      consistency = 0.0
      teacher_outputs = []
      student_outputs = []
      
      for first_level, docs in hier_train_data.items():
        for instance, gold in docs:
          student_output_gold = run_one_doc(student_lstm, first_level, instance, gold, W, B)
          teacher_output_gold = run_one_doc(teacher_lstm, first_level, instance, gold, W, B)
          teacher_outputs.append([t[0] for t in teacher_output_gold]) #list of only outputs for all docs (teacher preds)
          student_outputs.append([s[0] for s in student_output_gold]) #list of all docs (student preds)
          teacher_models.append(teacher_outputs) #list of all models (lists of doc preds)
          
      #update student model
      student_loss = build_model(first_level, student_lstm, instance, gold, W, B)
      student_loss.backward()
      trainer.update()
      
      #average all the teacher model predictions
      teacher_avgs = [np.sum(np.array([m[r] for m in teacher_models]), axis = 0)//len(teacher_models[0]) for r in range(len(teacher_models[0]))]
      
      #calculate consistency cost between teacher and student models
      for dt, ds in zip(teacher_avgs, student_outputs):
        consistency += calculate_mse(dt, ds)
      consistency_costs.append(consistency)
      #print(consistency_costs)  
      
      #early stopping condition: consistency cost
      if len(consistency_costs) > 5:
        diff1 = math.fabs(consistency_costs[-1] - consistency_costs[-2])
        diff2 = math.fabs(consistency_costs[-2] - consistency_costs[-3])
        if math.fabs(diff1 - diff2) < 0.0001:
          break  
        else:
          continue
      else:
        continue


def compute_metrics(results_dict):
    """
    Computes precision, recall, and F1 scores for the P, I, O categories.
    @params: results_dict is a dictionary with P, I, O as keys and lists of predictions
               for documents as values.
    @returns: metrics_dict is a dictionary with P, I, O as keys and [precision,
                recall, F1] as values.
    """
    num_pred_dict = {'I': 0, 'O': 0, 'P': 0}
    num_correct_dict = {'I': 0, 'O': 0, 'P': 0}
    total_pred = 0
    metrics_dict = {}
    
    for first_level in results_dict.keys(): #P, I, O
      num_correct = 0
      num_pred = 0
      for doc in results_dict[first_level]:
        for wd in doc: #(prediction, gold label)
          num_pred += 1
          if wd[0] == wd[1]:
            num_correct += 1
          else:
            continue 
      num_pred_dict[first_level] += num_pred
      num_correct_dict[first_level] += num_correct
      total_pred += num_pred
    
    if num_pred_dict['I'] != 0:
      i_recall = float(num_correct_dict['I'])/float(num_pred_dict['I'])
      i_precision = float(num_correct_dict['I'])/float(total_pred)
      i_f1 = float(2*i_recall*i_precision)/float(i_recall + i_precision)
      metrics_dict['I'] = [i_precision, i_recall, i_f1]
      
    if num_pred_dict['O'] != 0:
      o_recall = float(num_correct_dict['O'])/float(num_pred_dict['O'])
      o_precision = float(num_correct_dict['O'])/float(total_pred)
      o_f1 = float(2*o_recall*o_precision)/float(o_recall + o_precision)
      metrics_dict['O'] = [o_precision, o_recall, o_f1]
    
    if num_pred_dict['P'] != 0:
      p_recall = float(num_correct_dict['P'])/float(num_pred_dict['P'])
      p_precision = float(num_correct_dict['P'])/float(total_pred)
      p_f1 = float(2*p_recall*p_precision)/float(p_recall + p_precision)
      metrics_dict['P'] = [p_precision, p_recall, p_f1]
    
    
    return metrics_dict
        
        
          

def main():
    """
    Main function to control reading in data, creating vocabulary, mapping words to
    embeddings, initializing models, training, developing, and testing. Prints metrics for
    the first and second testing iterations, and requires DyNet teacher and student models
    to load.
    """

    train_paths = ['train/P/88754.tokens','train/I/3277760.tokens','train/O/352099.tokens','train/P/350565.tokens','train/O/1336890.tokens','train/I/8215273.tokens','train/I/43164.tokens','train/P/3137065.tokens']
    dev_paths = ['dev/I/16603337','dev/I/43164','dev/O/43164','dev/O/1300984','dev/P/7218018']
    test_paths = ['test/O/2474057','test/I/19931151']
    
    #make vocab (only once)
    #make_vocab_index(train_paths, 'PubMed-w2v.txt')
    
    #load vocab
    with open('vocab.pickle', 'rb') as v:
      vocab_idx = pkl.load(v)
      
    #get embeddings to train
    train_doc_embeddings = [file_to_embedding(path, vocab_idx) for path in train_paths]
    
    #get dev embeddings
    dev_doc_embeddings = [file_to_embedding(path+'.tokens', vocab_idx) for path in dev_paths]
    
    #get test embeddings
    test_doc_embeddings = [file_to_embedding(path+'.tokens', vocab_idx) for path in test_paths]
    
    hier_train_data = {'I':[],'O':[],'P':[]}
    hier_dev_data = {'I':[],'O':[],'P':[]}
    hier_test_data = {'I':[],'O':[],'P':[]}
    
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
        
    test_doc_labels = []
    for pd in test_paths:
      with open(pd+'_AGGREGATED.ann', 'r') as l:
        labels = [float(label) for label in l.readline().split(',')]
        test_doc_labels.append(labels)
        hier_test_data[pd[5]].append((test_doc_embeddings[test_paths.index(pd)], labels))
        
    #initialize models
    pc = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(pc)
    student_lstm = dy.LSTMBuilder(1, 200, 80, pc)
    teacher_lstm = dy.LSTMBuilder(1, 200, 80, pc)
    W = pc.add_parameters((8, 80))
    B = pc.add_parameters((8))
    
    #train
    train(student_lstm, teacher_lstm, W, B, trainer, hier_train_data)
    
    #run development
    #hier_dev_results = {'I':[],'O':[],'P':[]}
    #for first_level, docs in hier_dev_data.items():
      #for instance, gold in docs:
        #student_dev_predgold = run_one_doc(student_lstm, first_level, instance, gold, W, B)
        #teacher_dev_predgold = run_one_doc(teacher_lstm, first_level, instance, gold, W, B)
        #hier_dev_results[first_level].append(teacher_dev_predgold)
    
    #for error analysis   
    #print(hier_dev_results)
    
   
    #save models
    #dy.save("teacher_model",[teacher_lstm, W, B])
    #dy.save("student_model",[student_lstm, W, B])
    
    #load model
    pc = dy.ParameterCollection()
    teacher_lstm_load, W_load, B_load = dy.load("teacher_model", pc)
    student_lstm_load, W_load, B_load = dy.load("student_model", pc)
    
    #test on test data -- 1st iteration
    hier_test_results1 = {'I':[],'O':[],'P':[]}
    for first_level, docs in hier_test_data.items():
      for instance, gold in docs:
        teacher_test_predgold = run_one_doc(teacher_lstm_load, first_level, instance, gold, W_load, B_load)
        hier_test_results1[first_level].append(teacher_test_predgold)
        
    #test on test data - 2nd iteration
    hier_test_results2 = {'I':[],'O':[],'P':[]}
    for first_level, docs in hier_test_data.items():
      for instance, gold in docs:
        student_test_predgold = run_one_doc(student_lstm_load, first_level, instance, gold, W_load, B_load)
        hier_test_results2[first_level].append(student_test_predgold)
    
    #compute statistics
    metrics1 = compute_metrics(hier_test_results1)
    metrics2 = compute_metrics(hier_test_results2)
    print("First testing iteration: ", metrics1)
    print("Second testing iteration: ", metrics2)
    

main()
