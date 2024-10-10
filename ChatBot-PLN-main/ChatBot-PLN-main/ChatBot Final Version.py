

from textblob import TextBlob
import json
import nltk
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

with open("conversations.json", "r", encoding = 'utf8') as file:
    data = json.load(file)


vectorizer = TfidfVectorizer()

model = KMeans(n_clusters=2)

conversations = []
for item in data:
    conversation = item["conversation"].split(";")
    conversations.append({"question": conversation[0], "answer": conversation[1]})

questions = [conversation["question"] for conversation in conversations]
X = vectorizer.fit_transform(questions)

model.fit(X)

def get_intent(question):
    x = vectorizer.transform([question])
    cluster = model.predict(x)[0]
    return cluster

print(conversations[:2])

with open("conversations.json", "w", encoding = 'utf8') as file:
    json.dump(conversations, file)

for i, conversation in enumerate(conversations):
    x = vectorizer.transform([conversation["question"]])
    cluster = model.predict(x)[0]
    conversations[i]["cluster"] = cluster

original_question = input("Usuário: ")

nltk.download('rslp')
nltk.download("stopwords")
stemmer = RSLPStemmer()
stopwords = set(stopwords.words("portuguese"))

def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
    text = [stemmer.stem(word) for word in text]
    return text

preprocessed_question = preprocess_text(original_question)

cluster = get_intent(preprocessed_question)
    
related_conversations = [conversation for conversation in conversations if conversation["cluster"] == cluster]

best_match = max(related_conversations, key=lambda conversation: TextBlob(preprocessed_question).similarity(preprocess_text(conversation["question"])))
answer = best_match["answer"]

def handle_unknown(original_question):
    question = preprocess_text(original_question)
    tokens = nltk.word_tokenize(question)
    tagged = nltk.pos_tag(tokens)
    keywords = [word for word, pos in tagged if pos in ["NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP"]]
    related_conversations = [conversation for conversation in conversations if any(keyword in conversation["question"] for keyword in keywords)]
    best_match = max(related_conversations, key=lambda conversation: TextBlob(question).similarity(conversation["question"]))
    return best_match["answer"]
if not answer:
    answer = handle_unknown(original_question)

print("Resposta: ", answer)
conversations.append({"question": original_question, "answer": answer})

with open("conversations.json", "w", encoding = 'utf8') as file:
    json.dump(conversations, file)
