import re
import nltk
from collections import Counter

try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("Warning: Unable to download NLTK data. Some features may not work correctly.")

def form_sentence(prediction_history):
    if not prediction_history:
        return ""
    
    filtered_signs = []
    prev_sign = None
    
    for sign in prediction_history:
        if sign in ['nothing', 'no_hand', 'error']:
            continue
    
        if sign == 'del' and filtered_signs:
            filtered_signs.pop()  
            continue
        elif sign == 'space':
            if filtered_signs and filtered_signs[-1] != ' ':
                filtered_signs.append(' ')
            continue
        if sign != prev_sign:
            if sign.isalpha() and len(sign) == 1:
                filtered_signs.append(sign)
            prev_sign = sign
    
    raw_text = ''.join(filtered_signs)
    text = capitalize_sentences(raw_text)
    
    return text

def capitalize_sentences(text):
    if not text:
        return ""

    sentences = re.split(r'([.!?]\s*)', text)

    result = ""
    capitalize_next = True
    
    for part in sentences:
        if capitalize_next and part.strip():
            result += part[0].upper() + part[1:]
            capitalize_next = False
        else:
            result += part
            if re.search(r'[.!?]', part):
                capitalize_next = True
    
    return result

def predict_next_word(words, n=3):
    if len(words) < n:
        return None
    
    context = tuple(words[-(n-1):])
    common_ngrams = {
        ('i', 'am'): ['a', 'going', 'happy', 'here', 'not'],
        ('how', 'are'): ['you', 'they', 'we'],
        ('thank', 'you'): ['for', 'very', 'so'],
        ('nice', 'to'): ['meet', 'see', 'talk'],
        ('i', 'need'): ['help', 'to', 'a'],
        ('can', 'you'): ['help', 'please', 'show'],
    }
    
    return common_ngrams.get(context, None)

def detect_grammar_issues(text):
    corrections = [
        (r'\bi am\b', 'I am'),  
        (r'\bi\b', 'I'),        
        (r'\s{2,}', ' '),       
    ]
    
    corrected = text
    for pattern, replacement in corrections:
        corrected = re.sub(pattern, replacement, corrected)
    
    return corrected

def expand_abbreviations(text):
    """Expand common ASL abbreviations"""
    abbreviations = {
        'thx': 'thanks',
        'pls': 'please',
        'ur': 'your',
        'r': 'are',
        'u': 'you',
        'im': "I'm",
        'dont': "don't",
        'cant': "can't",
    }
    
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in abbreviations:
            words[i] = abbreviations[word.lower()]
    
    return ' '.join(words)

def clean_and_format_sentence(sentence):
    if not sentence:
        return ""
    
    sentence = expand_abbreviations(sentence)
    sentence = detect_grammar_issues(sentence)
    sentence = capitalize_sentences(sentence)
    

    if sentence and not re.search(r'[.!?]$', sentence):
        sentence += '.'
    
    return sentence.strip()     