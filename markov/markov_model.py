'''import re
import random
def preprocess(text):
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text.split()

def read_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw = f.read()
    return preprocess(raw)

def build_markov_chain(words):
    transitions = {}
    for i in range(len(words) - 1):
        curr_word = words[i]
        next_word = words[i + 1]
        if curr_word not in transitions:
            transitions[curr_word] = {}
        transitions[curr_word][next_word] = transitions[curr_word].get(next_word, 0) + 1
    return transitions

def build_probabilities(transitions):
    probabilities = {}
    for curr_word, nexts in transitions.items():
        total = sum(nexts.values())
        probabilities[curr_word] = {w: c / total for w, c in nexts.items()}
    return probabilities

def generate_text(probabilities, start_word, num_words=20):
    current = start_word
    result = [current]
    for _ in range(num_words-1):
        next_words = probabilities.get(current)
        if not next_words:
            break 
        words, probs = zip(*next_words.items())
        current = random.choices(words, probs)[0]
        result.append(current)
    return ' '.join(result)

filepath = "sample_essay.txt"
words = read_text_file(filepath)
transitions = build_markov_chain(words)
probabilities = build_probabilities(transitions)
keylist = list(transitions.keys())

# Display first few results for sanity check:
for curr_word, nexts in list(probabilities.items())[:10]:
    print(f"After '{curr_word}':")
    for next_word, prob in nexts.items():
        print(f"    {next_word}: {prob:.2f}")

num_of_words = int(input("How many number of words would you like to form: "))
prediction = generate_text(probabilities, keylist[0], num_of_words)
print("Possibel next word: ", prediction)'''

import re
import random

def preprocess(text):
    # Remove punctuation but keep spaces; convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation but keeps spaces
    text = text.lower()
    return list(text)  # returns list of characters, including spaces

def read_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw = f.read()
    return preprocess(raw)

def build_ngram_chain(chars, n=4):
    transitions = {}
    for i in range(len(chars) - n):
        curr_seq = ''.join(chars[i:i+n])
        next_char = chars[i+n]
        if curr_seq not in transitions:
            transitions[curr_seq] = {}
        transitions[curr_seq][next_char] = transitions[curr_seq].get(next_char, 0) + 1
    return transitions

def build_probabilities(transitions):
    probabilities = {}
    for curr_seq, nexts in transitions.items():
        total = sum(nexts.values())
        probabilities[curr_seq] = {c: count / total for c, count in nexts.items()}
    return probabilities

def generate_text(probabilities, start_seq, num_chars=100):
    current = start_seq
    result = [current]
    for _ in range(num_chars):
        next_chars = probabilities.get(current)
        if not next_chars:
            break
        chars, probs = zip(*next_chars.items())
        next_char = random.choices(chars, probs)[0]
        current = current[1:] + next_char 
        result.append(next_char)
    return ''.join(result)

# Usage
filepath = "sample_essay.txt"
chars = read_text_file(filepath)
transitions = build_ngram_chain(chars, n=4)
probabilities = build_probabilities(transitions)
keylist = list(transitions.keys())

# Display first few results for sanity check
for curr_seq, nexts in list(probabilities.items())[:10]:
    print(f"After '{curr_seq}':")
    for next_char, prob in nexts.items():
        print(f"    '{next_char}': {prob:.2f}")

num_chars = int(input("How many characters would you like to generate? "))
prediction = generate_text(probabilities, keylist[0], num_chars)
print("Possible character sequence:\n", prediction)