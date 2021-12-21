from flask import Flask, render_template, request
import random
import json
from keras.models import load_model
from nltk.collocations import QuadgramCollocationFinder
import numpy as np
import pickle
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import requests
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model')
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model_sent = load_model('cnn_w2v.h5')

class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
max_seq_len = 500

worry_gifs = ["Everything will be OK; just relax and take a deep breath.", "Good things are coming down the road, just don’t stop walking.", "You can’t calm the storm, so stop trying. What you can do is calm yourself, the storm will pass.", "Tides don’t last forever and when they go, they leave behind beautiful seashells.",
              "Take a deep breath, and relax, it’s all going to turn out better than you expected."]

jokes = ["Why can't a bicycle stand on its own?\nIt's two tired😴🚲", "If we weigh 100 Kgs on Earth, we'll weigh 38 Kgs on Mars, and just 16.6 Kgs on the Moon! It goes to prove, we're not overweight, just on the wrong planet.",
         "What movie should you watch on a dinner date? 🍴\nKabhi Sushi Kabhi Rum🍣🍹"]

twisters = ["How much wood would a woodchuck\nchuck\nIf a woodchuck would chuck wood? A woodchuck would chuck all the\nwood he could chuck\nIf a woodchuck would chuck wood", "A big black bug bit a big black bear and made the big black bear bleed blood", "Shy Shelly says she shall sew sheets💪",
            "Sheela sat slowly sewing some silk salwars👖", "Pinky's papa picked a pink papaya to pickle", "I thought a thought.\nBut the thought I thought wasn't the thought I thought I thought.\nIf the thought I thought I thought had been the thought I thought,I wouldn't have thought so much."]

starter_inspirations = ["Cheer up! Everything will be alright. Try having something delicious?", "To fall in love with yourself is the first secret to happiness. -Robert Morely", "Be gentle with yourself, learn to love yourself, to forgive yourself, for only as we have the right attitude toward ourselves can we have the right attitude toward others. -Wilfred Peterson",
                        "Love yourself!", "To fall in love with yourself is the first secret to happiness. -Robert Morely", "Be gentle with yourself, learn to love yourself, to forgive yourself, for only as we have the right attitude toward ourselves can we have the right attitude toward others. -Wilfred Peterson", "Love yourself!", "Try doing nothing for 15 seconds", "If you’re searching for that one person that will change your life, take a look in the mirror."]

songs = ["https://youtu.be/I0czvJ_jikg", "https://youtu.be/Xn676-fLq7I", "https://youtu.be/_Yhyp-_hX2s",
         "https://youtu.be/G-UnzRM24IM", "https://youtu.be/nkqVm5aiC28", "https://youtu.be/1k8craCGpgs"]

cheer_up = ['Do you know, you are the most perfect you there is. \N{smiling face with halo}', 'Do you know, you are one of the smartest people I know. \N{smiling face with halo}',
            'Do you know, you have the best smile. \N{smiling face with halo}', "You are the best ever, even better than a chocolate!🍫", "The word legend was coined in your praise!😎", "You are the rising sun on a cloudy day✨"]

books = ["https://1lib.in/book/3408539/b1502b", "https://1lib.in/book/737677/96de75",
         "https://1lib.in/book/2884726/51cc4b", "https://1lib.in/book/2765386/3121a6"]

movies = [" https://www.youtube.com/watch?v=reRcVxAWT1g  ", " https://www.youtube.com/watch?v=xroy2VFphi4 ", " https://www.youtube.com/watch?v=lTxn2BuqyzU ",
          " https://www.youtube.com/watch?v=XZHim74k7CA", "https://www.youtube.com/watch?v=sAea71tH8Zc"]  # biscuit #piper watermelon , night before math exam , camera

#---------------------------------#
lazy_words = ["lazy", "boring", "bored"]
bodyshame_words = ["obesity", "overweight", " fat", "chubby"]

exercises = ['Go for a walk! ',
             "How about a run? Don't forget to stretch! ",
             'Stand up, walk around, and streeeeetch! ',
             'Try some yoga!',
             'Head outside and explore! The world is your oyster.',
             'Try some pushups. Down, up 1!', ]

data_train = pd.read_csv('data/data_train.csv', encoding='utf-8')
data_test = pd.read_csv('data/data_test.csv', encoding='utf-8')
data = data_train.append(data_test, ignore_index=True)


def clean_text(data):

    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)

    # tekenization using nltk
    data = word_tokenize(data)

    return data


texts = [' '.join(clean_text(text)) for text in data.Text]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


def sentiment(sent_msg):
    seq = tokenizer.texts_to_sequences(sent_msg)
    padded = pad_sequences(seq, maxlen=max_seq_len)
    pred = model_sent.predict(padded)
    return pred


def sentiment_response(sent_msg):
    pred = sentiment(sent_msg)
    emotion_pred = class_names[np.argmax(pred)]

    if emotion_pred == 'joy':
        emotion_cond = "Happy 😄"
    elif emotion_pred == 'fear':
        emotion_cond = "Fear 😰"
    elif emotion_pred == 'anger':
        emotion_cond = "Angry 😡"
    elif emotion_pred == 'sadness':
        emotion_cond = "Sad 😢"
    elif emotion_pred == 'neutral':
        emotion_cond = "Neutral 😐"
    else:
        emotion_cond = "Neutral 😐"

    return emotion_cond


def contains_depression_traces(message):
    depression_keywords = ["kill myself", "cut myself", "I want to die",
                           "hate myself", "end my life", "self harm", "i don't want to live", "harm", "die"]
    sentiment_analyze = SentimentIntensityAnalyzer()
    sentiment_dict = sentiment_analyze.polarity_scores(message)
    # if score is lesser tha -0.95 , message may be serious
    if sentiment_dict['neg'] >= 0.95:
        return True
    is_depressing = sentiment_dict['neg'] > .75 or (
        sentiment_dict['neg'] > .5 and sentiment_dict['pos'] < .5)

    for word in depression_keywords:  # rechecking if message includes harmful keywords
        if word in message:
            return is_depressing
    return False


def depression_response():
    intro = "Hey, Are you okay? If you're struggling with anything, you're not alone and help is available. Here are some resources:"
    r1 = "  -  National Suicide Prevention Lifeline: call 9152987821 or visit http://www.aasra.info/"
    r2 = "  -  Please remember you are always loved: "

    return "{intro}\n\n{r1}\n{r2}".format(intro=intro, r1=r1, r2=r2)


def contains_stress_traces(message):
    stress_keywords = ['stress', 'anxiety', 'anxious', 'stressed',
                       'scared', 'afraid', 'fear', 'need help', 'need support']
    sentiment_analyze = SentimentIntensityAnalyzer()
    sentiment_dict = sentiment_analyze.polarity_scores(message)
    # if score between -0.85 and -0.55
    if sentiment_dict['compound'] >= -0.85 and sentiment_dict['compound'] < -0.55:
        return True
    is_stressed = sentiment_dict['neg'] > .75 or (
        sentiment_dict['neg'] > .5 and sentiment_dict['pos'] < .5)
    for word in stress_keywords:  # rechecking if message includes stress keywords
        if word in message:
            return is_stressed
    return False


def stress_response():
    intro = "Hey, you sound stressed"
    r2 = "\nWould you like to listen to some songs?"
    r3 = "\nCheck this out"
    r4 = random.choice(songs)
    return "{intro}\n{r2}\n{r3}\n{r4}".format(intro=intro, r2=r2, r3=r3, r4=r4)


def contains_worry_traces(message):
    worry_keywords = ["sad", "depressed", "stress", "stressed", "unhappy", "miserable", "angry", "depressing", "cry", "crying", "worry", "worried", "tensed ",
                      "trouble", "distress", "strain", "upset ", "anxiety ", "irritated", "irritation", "pressure ", "difficulty ", "suffering", "burden ", "afraid", "tired"]
    sentiment_analyze = SentimentIntensityAnalyzer()
    sentiment_dict = sentiment_analyze.polarity_scores(message)
    is_worried = sentiment_dict['neg'] > .75 or (
        sentiment_dict['neg'] > .5 and sentiment_dict['pos'] < .5)
    for word in worry_keywords:  # rechecking if message includes worry keywords
        if word in message:
            return is_worried
    return False


def worry_response():
    intro = "Hey, it's okay life happens sometimes"
    r1 = random.choice(worry_gifs)
    return "{intro}\n{r1}".format(intro=intro, r1=r1)


def contains_happy_traces(message):
    happy_words = ["happy", "joy", "yohooo", "yay!", "yay",
                   "cheerful", "cheers", "nice", "cool", "helpful", "great", "wow"]
    sentiment_analyze = SentimentIntensityAnalyzer()
    sentiment_dict = sentiment_analyze.polarity_scores(message)
    is_happy = sentiment_dict['pos'] > .75 or (
        sentiment_dict['pos'] > .5 and sentiment_dict['neg'] < .5)
    for word in happy_words:
        if word in message:
            return is_happy
    return False


def happy_response():
    intro = "I am glad you are happy!"
    # importing modules

    r1 = "Try typing $joy for happieeer suggestions"
    return "\n{intro}\n{r1}".format(intro=intro, r1=r1)


def get_quote():
    # get response from api
    response = requests.get("https://zenquotes.io/api/random")
    # converts response to json #response.text --> from api documentation
    json_data = json.loads(response.text)
    quote = json_data[0]['q'] + "\n-" + json_data[0]['a']
    return(quote)


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    for word in intents:
        if userText.startswith(word):
            return 'Hello! How are you, lovely person'
    if userText.startswith("$joy"):
        quote = get_quote()
        joy_list = [random.choice(songs), quote, random.choice(cheer_up), random.choice(
            exercises), random.choice(books), random.choice(movies), random.choice(jokes), random.choice(twisters)]
        response = "Try this:\n" + random.choice(joy_list)
        return response

    elif ("bye" in userText) or ("goodbye" in userText):
        return chatbot_response(userText)

    elif "no friends" in userText or ("no friend" in userText) or ("lonely" in userText):
        response = emoji.emojize(
            "Hey, I am your friend. I'm always here for you :red_heart:")
        return response

    # iterates over lazy_words to check if there is any word in message that is also in lazy_words
    elif any(word in userText for word in lazy_words):
        response = random.choice(exercises)
        # gives a random excercise suggestion
        return response

    elif ("compliment" in userText) or ("compliments" in userText):
        compliments = ['You are more fun than bubble wrap <3',
                       'You are the most perfect you there is. <3',
                       'You are enough. <3',
                       'You are one of the smartest people I know. <3',
                       'You look great today <3',
                       'You have the best smile <3',
                       'You light up the whole server <3']
        response = random.choice(compliments)
        return response

    elif ("movie" in userText) or ("movies" in userText):
        response = " Try this movie:\n" + random.choice(movies)
        return response

    elif ("book" in userText) or ("books" in userText):
        response = " Try this book:\n" + random.choice(books)
        return response

    elif ("song" in userText) or ("song" in userText):
        response = " Try this song:\n" + random.choice(songs)
        return response

    elif ("exercise" in userText) or ("overweight" in userText.lower()):
        return (random.choice(exercises))

    elif ("be my friend" in userText) or ("i love you" in userText):
        r1 = emoji.emojize(
            "Aww, You are the idli to my chutney :red_heart:")
        r2 = emoji.emojize("Aww, You are the aloo to my samosa :red_heart:"
                           )
        r3 = emoji.emojize("Aww, You are the paneer to my pakora :red_heart:")
        r4 = emoji.emojize(
            "Aww , You are the sambhar to my dosa :red_heart:")
        r5 = emoji.emojize(
            "Can we be friends? :red_heart:")
        responses = [r1, r2, r3, r4, r5]
        return (random.choice(responses))

    elif ("sleep" in userText):
        response = ["https://youtu.be/JEoxUG898qY",
                    "https://youtu.be/cI4ryatVkKw"]
        return (random.choice(response))

    elif ("cheer" in userText):
        return (random.choice(cheer_up))

    elif ("joke" in userText) or ("jokes" in userText) or ("make me laugh" in userText):
        response = "Try this:\n" + random.choice(jokes)
        return (response)

    elif ("tongue twister" in userText):
        response = "Try this:\n" + random.choice(twisters)
        return (response)

    elif any(word in userText for word in bodyshame_words):
        return (random.choice(cheer_up))

    elif contains_happy_traces(userText):
        response = happy_response()
        return (response)

    else:
        return chatbot_response(userText)


@app.route("/get2")
def get_bot_sentiment():
    userText = request.args.get('msg')
    sent_msg = [userText]
    return str('Based on my calculation, you currently feel "'+sentiment_response(sent_msg)+'"')


# def get_bot_response2():
#     userText = request.args.get('msg')
#     nltk.download('vader_lexicon')
#     from nltk.sentiment.vader import SentimentIntensityAnalyzer
#     sid = SentimentIntensityAnalyzer()
#     score = ((sid.polarity_scores(str(userText))))['compound']
#     if(score > 0):
#         label = 'Sentiment: Positive'
#     elif(score == 0):
#         label = 'Sentiment: Neutral'
#     else:
#         label = 'Sentiment: Negative'
#     return label
if __name__ == "__main__":
    app.run()
