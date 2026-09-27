#!/usr/bin/env python3
import nltk
from nltk.corpus import wordnet
from fuzzywuzzy import process
import re

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Example acronym dictionary (expand as needed)
ACRONYM_DICT = {
    'AI': 'Artificial Intelligence',
    'NLP': 'Natural Language Processing',
    'FAQ': 'Frequently Asked Questions',
    # Add more acronyms and definitions here
}

# Use a small, domain-specific vocabulary for spelling correction
DOMAIN_VOCABULARY = set(['implementation', 'financial', 'resources', 'assist', 'conference', 'parties', 'WHO', 'FCTC', 'country', 'support', 'aid', 'help', 'funding', 'document', 'criteria', 'benefits', 'application', 'eligibility'])

def synonym_expand(query):
    words = query.split()
    expanded = set(words)
    for word in words:
        # Only expand nouns and verbs, and only if unambiguous
        synsets = wordnet.synsets(word)
        if len(synsets) == 1:
            for lemma in synsets[0].lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return ' '.join(expanded)

def acronym_expand(query):
    for acronym, definition in ACRONYM_DICT.items():
        pattern = r'\b' + re.escape(acronym) + r'\b'
        query = re.sub(pattern, definition, query)
    return query

def spelling_correct(query):
    words = query.split()
    corrected = []
    for word in words:
        if word.lower() not in DOMAIN_VOCABULARY:
            best_match = process.extractOne(word, DOMAIN_VOCABULARY)
            if best_match and best_match[1] > 90:
                corrected.append(best_match[0])
            else:
                corrected.append(word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

def hyde_contextual_query(query, llm=None):
    # Only use HyDE if LLM is provided and query is short
    if llm and len(query.split()) < 12:
        hypothetical_answer = llm(query)
        return query + ' ' + hypothetical_answer
    return query

def preprocess_query(query, llm=None):
    # Only apply spelling correction for obvious typos (word not in domain vocab and >2 chars off)
    def safe_spelling_correct(word):
        if word.lower() not in DOMAIN_VOCABULARY:
            best_match = process.extractOne(word, DOMAIN_VOCABULARY)
            if best_match and best_match[1] > 95 and abs(len(word) - len(best_match[0])) > 2:
                return best_match[0]
        return word
    words = query.split()
    query = ' '.join([safe_spelling_correct(w) for w in words])
    query = acronym_expand(query)
    # Only expand synonyms for nouns/verbs and only if unambiguous and relevant
    expanded = set(words)
    for word in words:
        synsets = wordnet.synsets(word)
        if len(synsets) == 1 and synsets[0].pos() in ['n', 'v']:
            for lemma in synsets[0].lemmas():
                if lemma.name().replace('_', ' ') != word:
                    expanded.add(lemma.name().replace('_', ' '))
    query = ' '.join(expanded)
    # Only use HyDE if LLM is provided and query is short
    if llm and len(query.split()) < 12:
        hypothetical_answer = llm(query)
        query = query + ' ' + hypothetical_answer
    return query

# Example usage
if __name__ == "__main__":
    test_query = "What is AI and how does NLP work?"
    print("Original:", test_query)
    print("Preprocessed:", preprocess_query(test_query))
