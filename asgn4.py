
import dynet as dy
import numpy as np
import pickle as pkl

def make_vocab_index(file_list, path_to_embeddings):
    """
    Creates a training word-to-embedding mapping. Reads
    in words from a list of training files (tokenized) 
    and matches them with their embeddings from PubMed.
    Unknown words have values of -1 currently (need to change)
    """

    #init vocab
    vocab = {}

    #open file, read tokens
    for file_path in file_list:
      with open(file_path, 'r') as f:
        for token in f.readline().split(' '):
          if token not in vocab.keys():
            vocab[token] = np.random.uniform(-0.4, 0.4, 200)
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

    save_path = 'vocab.pickle'
    with open(save_path, 'wb') as v:
      pkl.dump(vocab, v)



def file_to_embedding(file_path, vocab):
    """
    file_path can't have numbers apart from the doc id
    """

    embed_file = {}
    doc_id = file_path.lstrip('abcdefghijklmnopqrstuvwxyz/-_ABCDEFGHIJKLMNOPQRSTUVWXYZ').partition('.')[0]
    #doc_id = file_path.lstrip('[aA-zZ]-_/').partition('.')[0]

    with open(file_path, 'r') as f:
      embed_file[doc_id] = np.array([vocab[token] if token in vocab.keys() else np.random.uniform(-0.4,0.4,200) for token in f.readline().split(' ')])

    return embed_file

def build_model(emb_files, file_labels):
    """
    emb_files is a dict of docs, file_labels is currently
    the list of labels for one doc
    """
    pc = dy.ParameterCollection()
    lstm = dy.LSTMBuilder(1, 200, 80, pc)
    W = pc.add_parameters(8, 80)
    B = pc.add_parameters(8)
    dy.renew_cg()
    output = []
    loss = []
    pred_gold = []
    for doc in emb_files.keys():
      s = lstm.initial_state()
      #loss = []
      for wdemb, label in zip(emb_files[doc], file_labels):
        x = dy.inputVector(wdemb)
        s = s.add_input(x)
        loss.append(dy.pickneglogsoftmax((W*s.output())+B,label))
        out_class = dy.softmax((W*s.output())+B)
        chosen_class = np.argmax(out_class.npvalue())
        pred_gold.append((chosen_class, label))
      #error = dy.esum(loss)

    trainer = dy.SimpleSGDTrainer(pc)
    for i in range(5):
      #error_val = error.value()
      error = dy.esum(loss)
      error_val = error.value()
      error.backward()
      trainer.update()
      print(pred_gold)
  
    
        
        

def main():
    #make_vocab_index(['/home/rianne/Documents/CSC/EBM-NLP-master/ebm_nlp__/documents/43164.tokens'], '/home/rianne/Documents/CSC/PubMed-w2v.txt')

    with open('vocab.pickle', 'rb') as v:
      vocab_idx = pkl.load(v)

    test_embed_file = file_to_embedding('/home/rianne/Documents/CSC/EBM-NLP-master/ebm_nlp__/documents/16603337.tokens', vocab_idx)
    with open('/home/rianne/Documents/CSC/EBM-NLP-master/ebm_nlp__/annotations/aggregated/hierarchical_labels/interventions/train/16603337_AGGREGATED.ann', 'r') as l:
      test_file_labels = [float(label) for label in l.readline().split(',')]

    build_model(test_embed_file, test_file_labels)


main()
