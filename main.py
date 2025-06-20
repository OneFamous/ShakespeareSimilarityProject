import os
import csv
import subprocess
import re
import random
import numpy as np


def read_in_shakespeare():
    '''Reads in the Shakespeare dataset and processes it into a list of tuples.
       Also reads in the vocab and play name lists from files.
    
    Returns:
        tuples: A list of (play_name, tokenized_line) tuples
        document_names: A list of unique play names from the CSV
        vocab: A list of all tokens in the vocabulary
    '''
    tuples = []
    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            play_name = row[1].strip()  
            line = row[5] 
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split() 
            line_tokens = [token.lower() for token in line_tokens if token]
            
            tuples.append((play_name, line_tokens))

    with open('vocab.txt') as f:
        vocab = [line.strip() for line in f if line.strip()]
  
    with open('play_names.txt') as f:
        document_names = list(set([tpl[0] for tpl in tuples]))
    
    print(f"Term-Document Matrix will be: {len(vocab)}x{len(document_names)} (|V| x D)")
    
    return tuples, document_names, vocab

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
  '''Returns a numpy array containing the term document matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

  Let m = len(vocab) and n = len(document_names).

  Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of words
        and each column corresponds to a document. A_ij contains the
        frequency with which word i occurs in document j.
  '''
  vocab_to_id = {word: idx for idx, word in enumerate(vocab)}
  doc_to_id = {doc: idx for idx, doc in enumerate(document_names)}

  td_matrix = np.zeros((len(vocab), len(document_names)), dtype=int)

  for play_name, tokens in line_tuples:
      if play_name not in doc_to_id:
          continue
      doc_id = doc_to_id[play_name]
      for token in tokens:
          if token in vocab_to_id:
              word_id = vocab_to_id[token]
              td_matrix[word_id, doc_id] += 1

  singletons = np.sum(np.sum(td_matrix, axis=1) == 1)
  print(f"Number of hapax legomena (singletons): {singletons}")
  return td_matrix

def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
  '''Returns a numpy array containing the term context matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    vocab: A list of the tokens in the vocabulary

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

  Let n = len(vocab).

  Returns:
    tc_matrix: A nxn numpy array where A_ij contains the frequency with which
        word j was found within context_window_size to the left or right of
        word i in any sentence in the tuples.
  '''

  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

  n = len(vocab)
  tc_matrix = np.zeros((n, n), dtype=int)

  for _, tokens in line_tuples:
    for i, target_word in enumerate(tokens):
      if target_word not in vocab_to_id:
        continue
      target_id = vocab_to_id[target_word]
      left = max(0, i - context_window_size)
      right = min(len(tokens), i + context_window_size + 1)
      for j in range(left, right):
        if i == j:
          continue
        context_word = tokens[j]
        if context_word in vocab_to_id:
          context_id = vocab_to_id[context_word]
          tc_matrix[target_id][context_id] += 1
  return tc_matrix

def create_PPMI_matrix(term_context_matrix):
  '''Given a term context matrix, output a PPMI matrix.
  
  See section 15.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.
  
  Input:
    term_context_matrix: A nxn numpy array, where n is
        the numer of tokens in the vocab.
  
  Returns: A nxn numpy matrix, where A_ij is equal to the
     point-wise mutual information between the ith word
     and the jth word in the term_context_matrix.
  '''       
  
  total_sum = np.sum(term_context_matrix)
  row_sum = np.sum(term_context_matrix, axis=1)
  col_sum = np.sum(term_context_matrix, axis=0)
  expected = np.outer(row_sum, col_sum) / total_sum
  with np.errstate(divide='ignore', invalid='ignore'):
    ppmi = np.log2((term_context_matrix * total_sum) / expected)
    ppmi[np.isnan(ppmi)] = 0.0
    ppmi[np.isinf(ppmi)] = 0.0
    ppmi[ppmi < 0] = 0.0
  return ppmi

def create_tf_idf_matrix(term_document_matrix):
  '''Given the term document matrix, output a tf-idf weighted version.

  See section 15.2.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.

  Input:
    term_document_matrix: Numpy array where each column represents a document 
    and each row, the frequency of a word in that document.

  Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
  '''

  tf = term_document_matrix
  df = np.count_nonzero(term_document_matrix, axis=1)
  idf = np.log(tf.shape[1] / (df + 1e-10))
  tf_idf = tf * idf[:, np.newaxis]
  return tf_idf

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  dot = np.dot(vector1, vector2)
  norm1 = np.linalg.norm(vector1)
  norm2 = np.linalg.norm(vector2)
  if norm1 == 0 or norm2 == 0:
    return 0.0
  return dot / (norm1 * norm2)

def compute_jaccard_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  intersection = np.sum((vector1 > 0) & (vector2 > 0))
  union = np.sum((vector1 > 0) | (vector2 > 0))
  if union == 0:
    return 0.0
  return intersection / union

def compute_dice_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''

  intersection = np.sum((vector1 > 0) & (vector2 > 0))
  total = np.sum(vector1 > 0) + np.sum(vector2 > 0)
  if total == 0:
    return 0.0
  return 2 * intersection / total

def rank_plays(target_play_index, term_document_matrix, similarity_fn):
  ''' Ranks the similarity of all of the plays to the target play.

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

  Inputs:
    target_play_index: The integer index of the play we want to compare all others against.
    term_document_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index
  '''
  
  target_vector = term_document_matrix[:, target_play_index]  # Sütun vektörü al
  similarities = []
    
  for i in range(term_document_matrix.shape[1]):
      if i == target_play_index:
          continue
      current_vector = term_document_matrix[:, i]  # Diğer sütun vektörleri
      sim = similarity_fn(target_vector, current_vector)
      similarities.append((i, sim))
    
  # Benzerliğe göre sırala (yüksekten düşüğe)
  similarities.sort(key=lambda x: x[1], reverse=True)
  return [i for i, _ in similarities]


def rank_words(target_word_index, matrix, similarity_fn):
  ''' Ranks the similarity of all of the words to the target word.

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

  Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
    similarity_fn: Function that should be used to compared vectors for two word
      ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the 
    target word indexed by word_index
  '''

  target_vector = get_row_vector(matrix, target_word_index)
  similarities = []
  for i in range(matrix.shape[0]):
    if i == target_word_index:
      continue
    vec = get_row_vector(matrix, i)
    sim = similarity_fn(target_vector, vec)
    similarities.append((i, sim))
  similarities.sort(key=lambda x: x[1], reverse=True)
  return [i for i, _ in similarities]


if __name__ == '__main__':
    tuples, document_names, vocab = read_in_shakespeare()

    print('Computing term document matrix...')
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)

    print('Computing tf-idf matrix...')
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    print('Computing term context matrix...')
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=4)

    print('Computing PPMI matrix...')
    PPMI_matrix = create_PPMI_matrix(tc_matrix)

    random_idx = random.randint(0, len(document_names)-1)
    print("\nSelected play:", document_names[random_idx])
    
    similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
    for sim_fn in similarity_fns:
        print('\nThe 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
        ranks = rank_plays(random_idx, td_matrix, sim_fn)
        
        if len(ranks) < 10:
            print(f"Warning: Only {len(ranks)} similar plays found")
        
        for idx in range(0, min(10, len(ranks))):
            doc_id = ranks[idx]
            print('%d: %s' % (idx+1, document_names[doc_id]))

    word = 'gain'
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
    
    if word in vocab_to_index: 
        for sim_fn in similarity_fns:
            print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
            ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
            for idx in range(0, min(10, len(ranks))):
                word_id = ranks[idx]
                print('%d: %s' % (idx+1, vocab[word_id]))

            print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
            ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
            for idx in range(0, min(10, len(ranks))):
                word_id = ranks[idx]
                print('%d: %s' % (idx+1, vocab[word_id]))
    else:
        print(f'\nError: Word "{word}" not in vocabulary')

         # ============================================
    # EXTRA: Tablolu çıktı üretmek için eklendi
    # ============================================

    def print_play_similarity_tables(play_index, matrix, matrix_name):
        print(f"\n--- Top 10 similar plays to \"{document_names[play_index]}\" using {matrix_name} ---")
        similarity_functions = [
            ("Cosine", compute_cosine_similarity),
            ("Jaccard", compute_jaccard_similarity),
            ("Dice", compute_dice_similarity)
        ]

        scores_by_method = {}
        for name, fn in similarity_functions:
            ranks = rank_plays(play_index, matrix, fn)
            scores = []
            for i in range(10):
                idx = ranks[i]
                score = fn(get_column_vector(matrix, play_index), get_column_vector(matrix, idx))
                scores.append((document_names[idx], score))
            scores_by_method[name] = scores

        print(f"{'Rank':<5} {'Cosine Similarity':<30} {'Jaccard Similarity':<30} {'Dice Similarity':<30}")
        print("-" * 100)
        for i in range(10):
            c = f"{scores_by_method['Cosine'][i][0]} ({scores_by_method['Cosine'][i][1]:.4f})"
            j = f"{scores_by_method['Jaccard'][i][0]} ({scores_by_method['Jaccard'][i][1]:.4f})"
            d = f"{scores_by_method['Dice'][i][0]} ({scores_by_method['Dice'][i][1]:.4f})"
            print(f"{i+1:<5} {c:<30} {j:<30} {d:<30}")


    def print_word_similarity_tables(word, matrix, matrix_name):
        print(f"\n--- Top 10 similar words to \"{word}\" using {matrix_name} ---")
        similarity_functions = [
            ("Cosine", compute_cosine_similarity),
            ("Jaccard", compute_jaccard_similarity),
            ("Dice", compute_dice_similarity)
        ]

        word_index = vocab.index(word)
        scores_by_method = {}

        for name, fn in similarity_functions:
            ranks = rank_words(word_index, matrix, fn)
            scores = []
            for i in range(10):
                idx = ranks[i]
                score = fn(get_row_vector(matrix, word_index), get_row_vector(matrix, idx))
                scores.append((vocab[idx], score))
            scores_by_method[name] = scores

        print(f"{'Rank':<5} {'Cosine Similarity':<30} {'Jaccard Similarity':<30} {'Dice Similarity':<30}")
        print("-" * 100)
        for i in range(10):
            c = f"{scores_by_method['Cosine'][i][0]} ({scores_by_method['Cosine'][i][1]:.4f})"
            j = f"{scores_by_method['Jaccard'][i][0]} ({scores_by_method['Jaccard'][i][1]:.4f})"
            d = f"{scores_by_method['Dice'][i][0]} ({scores_by_method['Dice'][i][1]:.4f})"
            print(f"{i+1:<5} {c:<30} {j:<30} {d:<30}")

    # 🔸 Oyun benzerliği tabloları:
    print_play_similarity_tables(random_idx, td_matrix, "Term-Document")
    print_play_similarity_tables(random_idx, tf_idf_matrix, "TF-IDF")

    # 🔸 Kelime benzerliği tabloları:
    if word in vocab:
        print_word_similarity_tables(word, tc_matrix, "Term-Context Frequency Matrix")
        print_word_similarity_tables(word, PPMI_matrix, "PPMI Matrix")