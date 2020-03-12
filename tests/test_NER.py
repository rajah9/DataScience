from unittest import TestCase
from ner import Nltk_ner, Spacy_ner
import pprint
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

demo = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'
sents = [
    'Is David coming into your conference room or is he calling us',
    'Did you get my invite for the 9am meeting tomorrow with the Club?',
    'Morning, Looks like we got some time with Kevin today!',
    'Per Chief Legal Officer Christopher Powell, the certificate of formation',
    'against her manager Erick Strati for bullying me and unethical',
]

class TestNltk_ner(TestCase):
    def setUp(self) -> None:
        self.ner = Nltk_ner()
        self.sentence = sents[4]

    def test_nltk_tree(self):
        actual = self.ner.tree(self.sentence)
        self.assertIsNotNone(actual)

    def test_preprocess(self):
        actual = self.ner.preprocess(self.sentence)
        self.assertIsNotNone(actual)

class TestSpacy_ner(TestCase):
    def setUp(self) -> None:
        self.ner = Spacy_ner()
        self.sentence = sents[4]

    def test_preprocess(self):
        actual = self.ner.preprocess(self.sentence)
        text_label = [(X.text, X.label_) for X in actual.ents]
        logger.debug(text_label)
        test_norp = 'European'
        expected_label = 'NORP'

