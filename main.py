import nltk
import wikipedia

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

EXIT_WORDS = ['exit', 'quit', 'bye', 'goodbye']

LEMMATIZER = WordNetLemmatizer()


def lemma_me(sent):
    sentence_tokens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sentence_tokens)

    sentence_lemmas = []
    for token, pos_tag in zip(sentence_tokens, pos_tags):
        if pos_tag[1][0].lower() in ['n', 'v', 'a', 'r']:
            lemma = LEMMATIZER.lemmatize(token, pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)

    return sentence_lemmas


VECTORIZER = TfidfVectorizer(tokenizer=lemma_me)


class Engine:

    def __init__(self, topic):
        content = wikipedia.page(topic).content
        self.content_tokens = nltk.sent_tokenize(content)
        self._ft = VECTORIZER.fit_transform(self.content_tokens)

    def get_answer(self, question):
        question_vector = VECTORIZER.transform([question])
        values = cosine_similarity(question_vector, self._ft)
        index = values.argsort()[0][-1]
        values_flat = values.flatten()
        values_flat.sort()
        coeff = values_flat[-1]
        if coeff > 0.3:
            return self.content_tokens[index]


def main():
    topic = input("Topic: ").lower()
    engine = Engine(topic)

    while True:
        question = input("Question: ").lower()

        if question in EXIT_WORDS:
            print("\nBye! Talk to you later...\n")
            break

        output = engine.get_answer(question)
        if output:
            print(f"\nBot: {output}\n")
        else:
            print("\nI don't know.\n")


if __name__ == "__main__":
    main()
