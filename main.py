import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware

class Item(BaseModel):
    sentence: str

app = FastAPI()

# Enable CORS to handle preflight OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the frontend's domain if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define a root endpoint
@app.get("/")
def root():
    return {"message": "sign language api"}

# Explicitly handle OPTIONS request for /a2sl
@app.options("/a2sl")
def handle_options():
    return {"Allow": "POST, OPTIONS"}

@app.post("/a2sl")
def a2sl(item: Item):
    text = item.sentence
    text = text.lower()

    # Tokenize input text into words
    words = word_tokenize(text)

    # Perform part of speech tagging on the words
    tagged = nltk.pos_tag(words)

    # Create a dictionary to store tense information
    tense = {
        "future": len([word for word in tagged if word[1] == "MD"]),
        "present": len([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]]),
        "past": len([word for word in tagged if word[1] in ["VBD", "VBN"]]),
        "present_continuous": len([word for word in tagged if word[1] in ["VBG"]]),
    }

    stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn',
                      'do', "you've", 'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't",
                      'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's',
                      "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a',
                      'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',
                      'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn',
                      "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn',
                      'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being',
                      'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])

    lr = WordNetLemmatizer()

    filtered_text = []
    for w, p in zip(words, tagged):
        if w not in stop_words:
            if p[1] == 'VBG' or p[1] == 'VBD' or p[1] == 'VBZ' or p[1] == 'VBN' or p[1] == 'NN':
                filtered_text.append(lr.lemmatize(w, pos='v'))
            elif p[1] == 'JJ' or p[1] == 'JJR' or p[1] == 'JJS' or p[1] == 'RBR' or p[1] == 'RBS':
                filtered_text.append(lr.lemmatize(w, pos='a'))
            else:
                filtered_text.append(lr.lemmatize(w))

    words = filtered_text  # Update words with filtered text

    temp = []

    # Loop through the words and replace "I" with "Me"
    for w in words:
        if w == 'I':
            temp.append('Me')
        else:
            temp.append(w)

    words = temp  # Update words

    probable_tense = max(tense, key=tense.get)

    # Check the probable tense and make adjustments
    if probable_tense == "past" and tense["past"] >= 1:
        words = ["Before"] + words
    elif probable_tense == "future" and tense["future"] >= 1:
        if "Will" not in words:
            words = ["Will"] + words
    elif probable_tense == "present":
        if tense["present_continuous"] >= 1:
            words = ["Now"] + words

    words = [word.lower() for word in words]  # Ensure all words are lowercase

    # List of animation videos for sign language
    videos = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "after", "again", "against", 
              "age", "all", "alone", "also", "and", "ask", "at", "b", "be", "beautiful", "before", 
              "best", "better", "busy", "but", "bye", "c", "can", "cannot", "change", "college", 
              "come", "computer", "d", "day", "distance", "do not", "does not", "e", "eat", 
              "engineer", "f", "fight", "finish", "from", "g", "glitter", "go", "god", "gold", 
              "good", "great", "h", "hand", "hands", "happy", "hello", "help", "her", "here", "his", 
              "home", "homepage", "how", "i", "invent", "it", "j", "k", "keep", "l", "language", 
              "laugh", "learn", "m", "me", "more", "my", "n", "name", "next", "not", "now", "o", 
              "of", "on", "our", "out", "p", "pretty", "q", "r", "right", "s", "sad", "safe", "see", 
              "self", "sign", "so", "sound", "stay", "study", "t", "talk", "television", "thank you", 
              "thank", "that", "they", "this", "those", "time", "to", "type", "u", "us", "w", "walk", 
              "wash", "way", "we", "welcome", "what", "when", "where", "which", "who", "whole", 
              "whose", "why", "will", "with", "without", "words", "work", "world", "wrong", "x", 
              "y", "you", "your", "yourself", "z"]

    filtered_text = []  # Initialize a list to store filtered and title-cased words

    # Loop through words in the sentence
    for w in words:
        # If the word doesn't have an animation, split it into characters
        if w not in videos:
            for c in w:
                filtered_text.append(c)
        else:
            filtered_text.append(w)

    words = [word.title() for word in filtered_text]  # Title-case all the words

    return {"words": words}  # Return the result as a JSON object


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
