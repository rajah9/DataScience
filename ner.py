import logging

from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

from LogitUtil import logit

_IS_PROD = True

# download('punkt')
# download('averaged_perceptron_tagger')
# download('maxent_ne_chunker')
# download('words')

class Ner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if _IS_PROD:
            self.logger.handlers = []
            self.logger.addHandler(logging.NullHandler())


class Nltk_ner(Ner):
    @logit(showArgs=False, showRetVal=True)
    def tree(self, sent:str) -> Tree:
        ne_tree = ne_chunk(pos_tag(word_tokenize(sent)))
        return ne_tree

    @logit(showArgs=False, showRetVal=True)
    def preprocess(self, sent) -> list:
        tokens = word_tokenize(sent)
        sent = pos_tag(tokens)
        return sent

import spacy


#import en_core_web_sm # NOT pip installable; have to run first: python -m spacy download en_core_web_sm
# tried: python -m spacy download en BUT got "Download successful but linking failed"

class Spacy_ner(Ner):
    def __init__(self):
        super().__init__()
        model_to_load = "en" # "en_core_web_sm"
        model_to_load = r'C:\Program Files\Python\Python37\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.2.5'
        self.nlp = spacy.load(model_to_load)

    def preprocess(self, sent:str) -> list:
        doc = self.nlp(sent)
        return doc

