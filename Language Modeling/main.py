from collections import Counter

import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import codecs
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.util import ngrams
import math
import numpy as np


def Myngram(tokens, n):
    tokens = nltk.word_tokenize(tokens)
    n_grams = list(ngrams(tokens, n))
    return n_grams
    # for gram in n_grams:
    #     print(gram)


def import_text(name):
    with codecs.open(name, "r", encoding="cp1252") as f:
        text = f.read()
        return text

def extract_sentence(text):
    text = text.replace("Mr.", "Mr<period>")
    text = text.replace("Jr.", "Jr<period>")
    text = text.replace("Ms.", "Ms<period>")
    text_tokens = sent_tokenize(text, "english")
    text_tokens = [sentence.replace("Jr<period>", "Jr.") for sentence in text_tokens]
    text_tokens = [sentence.replace("Mrs<period>", "Mrs.") for sentence in text_tokens]
    text_tokens = [sentence.replace("Mr<period>", "Mr.") for sentence in text_tokens]
    return text_tokens


def removing(senteces_token):
    patternEng = r'[a-zA-Z0-9]+'
    RegtokenizerEng = RegexpTokenizer(patternEng)
    result = []
    for sentence in senteces_token:
        noPunctuationtokens = RegtokenizerEng.tokenize(sentence)
        sentence = ' '.join(noPunctuationtokens)
        result.append(sentence)
    return result


def calculateUniprobability(sentences):
    gram1 = []
    for i in sentences:
        gram1.extend(Myngram(i, 1))
    counter = Counter(gram1)
    gramNumber = sum(counter.values())
    probability = {}
    for key, values in counter.items():
        probability[key] = values / gramNumber
    return probability, counter


def calculateBigramProbability(sentences):
    gram2 = []
    for i in sentences:
        gram2.extend(Myngram(i, 2))
    counter = Counter(gram2)
    bigram_probabilities = {}
    for (prev_word, curr_word), count in counter.items():
        if prev_word not in bigram_probabilities:
            bigram_probabilities[prev_word] = {}

        total_count = sum(counter.values())
        probability = count / total_count
        bigram_probabilities[prev_word][curr_word] = probability

    return bigram_probabilities


# def calculateThreegremProbability(sentences):
#     gram3 = []
#     for i in sentences:
#         gram3.extend(Myngram(i, 3))
#     counter = Counter(gram3)
#     threegram_probabilities = {}
#     for (prev_prev_word, prev_word, curr_word), count in counter.items():
#         if (prev_prev_word, prev_word) not in threegram_probabilities:
#             threegram_probabilities[(prev_prev_word, prev_word)] = {}
#         total_count = sum(counter[(prev_prev, prev, word)] for (prev_prev, prev, word) in counter if
#                           (prev_prev, prev) == (prev_prev_word, prev_word))
#         probability = count / total_count
#         threegram_probabilities[(prev_prev_word, prev_word)][curr_word] = probability
#
#     return threegram_probabilities


def calculateThreegramProbability(sentences):
    gram3 = []
    gram2 = []
    for i in sentences:
        gram3.extend(Myngram(i, 3))
        gram2.extend(Myngram(i, 2))
    counter3 = Counter(gram3)
    counter2 = Counter(gram2)
    threegram_probabilities = {}
    for (prev_prev_word, prev_word, curr_word), count in counter3.items():
        if (prev_prev_word, prev_word) not in threegram_probabilities:
            threegram_probabilities[(prev_prev_word, prev_word)] = {}
        bigram_count = counter2[(prev_prev_word, prev_word)]
        probability = count / bigram_count
        threegram_probabilities[(prev_prev_word, prev_word)][curr_word] = probability
    return threegram_probabilities


def calculateLaplaceSmothing(sentences):
    gram3 = []
    for i in sentences:
        gram3.extend(Myngram(i, 1))
    counter = Counter(gram3)
    Unique_V = len(counter)
    Total_numberOfWords = sum(counter.values())
    LS = {}
    for key, values in counter.items():
        LS[key] = (values + 1) / (Total_numberOfWords + Unique_V)
    return LS, Total_numberOfWords


# def calculategood_smoothingBigramProbability(sentences):
#     bigram_counts = Counter()
#     unigram_counts = Counter()
#     for sentence in sentences:
#         words = sentence.split()
#         bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
#         bigram_counts.update(bigrams)
#         unigram_counts.update(words)
#     observed_counts = Counter(bigram_counts.values())
#     total_bigrams = sum(bigram_counts.values())
#     total_unigrams = sum(unigram_counts.values())
#     observed_frequencies = np.array([observed_counts[i] for i in range(1, max(observed_counts) + 1)])
#
#     r_star = (np.arange(1, max(observed_counts)) + 1) * (observed_frequencies[1:] / observed_frequencies[:-1])
#     c_star = r_star - observed_counts[1:]
#     c_star[0] = 1  # Adjust for unseen bigrams
#     bigram_probabilities = {}
#     for bigram, count in bigram_counts.items():
#         prev_word = bigram[0]
#         probability = (count + c_star[count]) / (total_unigrams + c_star[1])
#         if prev_word not in bigram_probabilities:
#             bigram_probabilities[prev_word] = {}
#         bigram_probabilities[prev_word][bigram[1]] = probability
#     return bigram_probabilities


#def calculategood_smoothingBigramProbability(sentences):


from collections import defaultdict

from collections import defaultdict


def calculate_good_turing_unigram_probability(sentences):
    # Step 1: Count the frequency of each word
    word_count = defaultdict(int)

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            word_count[word] += 1

    # Step 2: Count the frequency of each word frequency
    word_freq_count = defaultdict(int)

    for count in word_count.values():
        word_freq_count[count] += 1

    # Step 3: Calculate the Good-Turing smoothed probability for each word
    total_words = sum(word_count.values())
    unseen_words = len(word_freq_count) - len(word_count)

    word_prob = {}
    N = len(word_count)
    for word, count in word_count.items():
        c_star = (count + 1) * word_freq_count[count + 1] / word_freq_count[count] if count < N else 1
        prob = c_star / total_words
        word_prob[word] = prob

    # Step 4: Calculate the Good-Turing smoothed probability for unseen words
    prob_unseen = word_freq_count[1] / total_words if 1 in word_freq_count else 0

    return word_prob, prob_unseen


def calculategood_smoothingBigramProbability(sentences):
    # Step 1: Count the frequency of each bigram
    bigram_count = defaultdict(int)
    unigram_count = defaultdict(int)

    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            bigram_count[(current_word, next_word)] += 1
            unigram_count[current_word] += 1

    # Step 2: Count the frequency of each bigram frequency
    bigram_freq_count = defaultdict(int)

    for count in bigram_count.values():
        bigram_freq_count[count] += 1

    # Step 3: Calculate the Good-Turing smoothed probability for each bigram
    total_bigrams = sum(bigram_count.values())
    total_unigrams = sum(unigram_count.values())
    unseen_bigrams = total_unigrams ** 2 - total_bigrams

    bigram_prob = {}
    N = len(bigram_count)
    for bigram, count in bigram_count.items():
        c_star = (count + 1) * bigram_freq_count[count + 1] / bigram_freq_count[count] if count < N else 1
        prob = (c_star / total_bigrams) + (unigram_count[bigram[1]] / total_unigrams) if unigram_count[
                                                                                             bigram[1]] > 0 else 0
        bigram_prob[bigram] = prob

    # Step 4: Calculate the Good-Turing smoothed probability for unseen bigrams
    prob_unseen = (1 / total_bigrams) * (bigram_freq_count[1] / total_bigrams) if 1 in bigram_freq_count else 0

    return bigram_prob, prob_unseen


def removepunctation(test):
    patternEng = r'[a-zA-Z0-9]+'
    RegtokenizerEng = RegexpTokenizer(patternEng)
    noPunctuationtokensBeanstalk = RegtokenizerEng.tokenize(test.lower())
    return noPunctuationtokensBeanstalk


def calculate_perplexity(language_model, test_dataset):
    total_log_prob = 0.0
    word_count = 0
    Unique_V = len(language_model)
    epsilon = 1e-10
    totalWords = len(test_dataset)
    for token in test_dataset:
        if token in language_model:
            word_prob = language_model[token]
        else:
            word_prob = epsilon

        total_log_prob += math.log(word_prob + epsilon)
        word_count += 1

    avg_log_prob = total_log_prob / word_count
    perplexity = math.exp(-avg_log_prob)
    return perplexity


def calculate_perplexity_bigram(sentences, bigram_probabilities):
    total_log_probability = 0.0
    total_words = 0
    sentences = list(sentences)
    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)
        for i in range(1, len(words)):
            prev_word = words[i-1]
            curr_word = words[i]
            # Check if bigram exists in the probabilities dictionary
            if prev_word in bigram_probabilities and curr_word in bigram_probabilities[prev_word]:
                bigram_probability = bigram_probabilities[prev_word][curr_word]
            else:
                # Handle unseen bigrams by assigning a very small probability
                bigram_probability = 1e-10
            total_log_probability += math.log2(bigram_probability)
    average_log_probability = total_log_probability / total_words
    perplexity = 2 ** (-average_log_probability)
    return perplexity


def get_probable_next_word(calculateLaplaceSmothing, previous_word):
    probable_next_word = None
    max_probability = 0.0
    for word, probability in calculateLaplaceSmothing.items():
        if word != previous_word and probability > max_probability:
            max_probability = probability
            probable_next_word = word

    return probable_next_word


def get_probable_next_word_bigram(bigram_model, previous_word):
    if previous_word in bigram_model:
        return max(bigram_model[previous_word], key=bigram_model[previous_word].get)
    else:
        return None


def get_probable_next_word_threegram(threegram_model, prev_prev_word, prev_word):
    if (prev_prev_word, prev_word) in threegram_model:
        return max(threegram_model[(prev_prev_word, prev_word)], key=threegram_model[(prev_prev_word, prev_word)].get)
    else:
        return None


def generate_unigram_sentence(length,theWord,BigramProbability):
    generatedsentence = [theWord]
    for i in range(length):
        theWord = get_probable_next_word_bigram(BigramProbability, theWord)
        generatedsentence.append(theWord)
    return generatedsentence


def generate_treegram_sentence(length,theWord,secWord,TreegramProbability):
    generatedsentence = [theWord,secWord]
    theWord = "I"
    secWord = "was"
    for i in range(20):
        generated = get_probable_next_word_threegram(TreegramProbability, theWord, secWord)
        generatedsentence.append(generated)
        theWord = secWord
        secWord = generated
    return  generatedsentence


def generate_bigram_sentence(length,theWord,secWord,bigramProbability):
    generatedsentence = [theWord,secWord]
    for i in range(length):
        generated = get_probable_next_word_bigram(bigramProbability, secWord)
        generatedsentence.append(generated)
        # theWord = secWord
        secWord = generated
    return generatedsentence


import random

def generate_uni_sentence(num_words, first_word, second_word, unigram_prob):
    sentence = [first_word, second_word]

    for _ in range(num_words - 2):
        next_word = select_next_word(unigram_prob)
        sentence.append(next_word)

    return sentence


def select_next_word(unigram_prob):
    words = list(unigram_prob.keys())
    probabilities = list(unigram_prob.values())

    return random.choices(words, probabilities)[0]



if __name__ == '__main__':
    # Preprocessing
    text = import_text("brown.train.txt")
    test = import_text("brown.test.txt")

    sentence = extract_sentence(text)
    sentenceTest = extract_sentence(test)
    result = removing(sentence)
    resultTest = removing(sentenceTest)
    testTokens = removepunctation(test)

     # unigram

    Uniprobability , UniqueWords = calculateUniprobability(result)
    laplaceSmothing,Total_numberOfWords = calculateLaplaceSmothing(result)
    perplexity = calculate_perplexity(laplaceSmothing,testTokens,Total_numberOfWords)
    theWord = get_probable_next_word(laplaceSmothing,("if",))

    unigram_goodturing, uni_unseen_gt = calculate_good_turing_unigram_probability(result)


    # bigram
    corpus = "I am learning language proccessing and I am happy."
    corpus = extract_sentence(corpus)
    corpus = removing(corpus)
    BigramProbability = calculateBigramProbability(result)
    print(BigramProbability)
    theWord = get_probable_next_word_bigram(BigramProbability,"I")
    print(theWord)

    bigram_laplace, bigram_Total_numberOfWords = calculateLaplaceSmothing(result)
    bigram_perplexity = calculate_perplexity_bigram(result, BigramProbability)
    bigram_goodturing, unseen_p = calculategood_smoothingBigramProbability(result)


    # generate sentence
    sentence_uni_one = generate_uni_sentence(20, 'I', 'was', Uniprobability)
    sentence_uni_two = generate_uni_sentence(20, 'Jury', 'said', Uniprobability)
    sentence_uni_three = generate_uni_sentence(20, 'The', 'jury', Uniprobability)
    sentence_uni_four = generate_uni_sentence(20, 'These', 'actions', Uniprobability)
    sentence_uni_five = generate_uni_sentence(20, 'Four', 'additional', Uniprobability)
    sentence_uni = [sentence_uni_one, sentence_uni_two, sentence_uni_three, sentence_uni_four, sentence_uni_five]

    sentence_bi_one = generate_bigram_sentence(20, 'I', 'was', BigramProbability)
    sentence_bi_two = generate_bigram_sentence(20, 'Jury', 'said', BigramProbability)
    sentence_bi_three = generate_bigram_sentence(20, 'The', 'jury', BigramProbability)
    sentence_bi_four = generate_bigram_sentence(20, 'These', 'actions', BigramProbability)
    sentence_bi_five = generate_bigram_sentence(20, 'Four', 'additional', BigramProbability)
    sentence_bi = [sentence_bi_one, sentence_bi_two, sentence_bi_three,sentence_bi_four,sentence_bi_five]

    uni_pp_mle = []
    uni_pp_laplace = []
    uni_pp_good = []
    bi_pp_mle = []
    bi_pp_laplace = []
    bi_pp_good = []

    # perplexity of sentences
    for i in range(len(sentence_uni)):
        uni_pp_mle.append(calculate_perplexity(Uniprobability, sentence_uni[i]))
        uni_pp_laplace.append(calculate_perplexity(laplaceSmothing, sentence_uni[i]))
        uni_pp_good.append(calculate_perplexity(unigram_goodturing, sentence_uni[i]))

    for i in range(len(sentence_bi)):
        bi_pp_mle.append(calculate_perplexity_bigram(sentence_bi[i], BigramProbability))
        bi_pp_laplace.append(calculate_perplexity_bigram(sentence_bi[i], bigram_laplace))
        bi_pp_good.append(calculate_perplexity_bigram(sentence_bi[i], bigram_goodturing))

    # perplexity of models
    uni_perplexity_mle = calculate_perplexity(Uniprobability, test)
    uni_perplexity_laplace = calculate_perplexity(laplaceSmothing, test)
    uni_perplexity_good = calculate_perplexity(unigram_goodturing, test)
    bi_perplexity_mle = calculate_perplexity_bigram(test, BigramProbability)
    bi_perplexity_laplace = calculate_perplexity_bigram(test, bigram_laplace)
    bi_perplexity_good = calculate_perplexity_bigram(test, bigram_goodturing)



    with open('unigram_probabilities.txt', 'w') as file:
        file.write(str(len(Uniprobability)))
        file.write('\n')
        file.write(str(Uniprobability))

    with open('bigram_probabilities.txt', 'w') as file:
        file.write(str(len(BigramProbability)))
        file.write('\n')
        file.write(str(BigramProbability))

    with open('unigram_laplace.txt', 'w') as file:
        file.write(str(len(laplaceSmothing)))
        file.write('\n')
        file.write(str(laplaceSmothing))


    with open('bigram_laplace.txt', 'w') as file:
        file.write(str(len(bigram_laplace)))
        file.write('\n')
        file.write(str(bigram_laplace))


    with open('unigram_goodturing.txt', 'w') as file:
        file.write(str(uni_unseen_gt) +" unseen probability")
        file.write('\n')
        file.write(str(unigram_goodturing))


    with open('bigram_goodturing.txt', 'w') as file:
        file.write(str(unseen_p) +" unseen probability")
        file.write('\n')
        file.write(str(bigram_goodturing))


    with open('uni_sentence_generate.txt', 'w') as file:
        file.write(str(sentence_uni_one) + '\n')
        file.write(str(sentence_uni_two) + '\n')
        file.write(str(sentence_uni_three) + '\n')
        file.write(str(sentence_uni_four) + '\n')
        file.write(str(sentence_uni_five) + '\n')

    with open('bi_sentence_generate.txt', 'w') as file:
        file.write(str(sentence_bi_one) + '\n')
        file.write(str(sentence_bi_two) + '\n')
        file.write(str(sentence_bi_three) + '\n')
        file.write(str(sentence_bi_four) + '\n')
        file.write(str(sentence_bi_five) + '\n')



    # threegram
    # threegram_model = calculateThreegremProbability(result)
    # print(threegram_model)
    # prev_prev_word = "I"
    # prev_word = "was"
    # next_word = get_probable_next_word_threegram(threegram_model, prev_prev_word, prev_word)
    # print(next_word)
