from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
import json
import random
import json
import random
import numpy as np
import pandas as pd

nl_model = load("models/nlm")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("datasets/findprakriti.csv")

X = df.drop("Dosha", axis="columns")
y = df["Dosha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("intents.json", "r") as f:
    intents = json.load(f)

intentList = {
    2: "greeting",
    1: "goodbye",
    6: "thanks",
    0: "about",
    4: "name",
    3: "help",
    5: "prakriti",
}

df = pd.read_csv("datasets/initial_sentences_intent.csv")
X_train = df.Sentence


def get_intent(msg_count):
    ans = np.argmax(nl_model.predict(msg_count, verbose=0))
    return intentList.get(ans)


vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(X_train.values)
print("len(X_train.values):", len(X_train.values))

questions = {
    0: {
        "question": "How is Your usual appearance of hair?",
        "options": {
            "1": "Dry;Black;Knotted;Brittle",
            "2": "Straight;Oily",
            "3": "Thick;Curly",
        },
    },
    1: {
        "question": "What is your typical body weight?",
        "options": {
            "1": "Low- difficulties in gaining weight",
            "2": "Moderate- no difficulties in gaining or losing weight",
            "3": "Heavy- difficulties in losing weight",
        },
    },
    2: {
        "question": "What is your height?",
        "options": {
            "1": "Short",
            "2": "Average",
            "3": "Tall",
        },
    },
    3: {
        "question": "What is your typical bone structure like?",
        "options": {
            "1": "Small- prominent joints",
            "2": "Medium bone structure",
            "3": "Large - broad shoulders;heavy bone structure",
        },
    },
    4: {
        "question": "What is your typical complexion?",
        "options": {
            "1": "Dark- tans easily",
            "2": "Fair- sunburns easily",
            "3": "White- pale;White- tans easily",
        },
    },
    5: {
        "question": "Describe the general feel of your skin",
        "options": {
            "1": "Dry;Thin;Cool to touch;Rough",
            "2": "Smooth;Warm;Oily T-zone",
            "3": "Thick;moist-greasy;Cold",
        },
    },
    6: {
        "question": "Describe the general texture of your skin",
        "options": {
            "1": "Dry;Pigments and aging",
            "2": "Freckles;Many moles;redness and rashes",
            "3": "Oily",
        },
    },
    7: {
        "question": "What is your typical hair color?",
        "options": {
            "1": "Black;Dull",
            "2": "Red;Light brown;Yellow",
            "3": "Brown",
        },
    },
    8: {
        "question": "Describe the general appearance of your hair",
        "options": {
            "1": "Dry;Black;Knotted;Brittle",
            "2": "Straight;Oily",
            "3": "Thick;Curly",
        },
    },
    9: {
        "question": "Describe the typical shape of your face",
        "options": {
            "1": "Long;Angular;Thin",
            "2": "Heart-shaped;Pointed chin",
            "3": "Large;Round;Full",
        },
    },
    10: {
        "question": "Describe your eyes",
        "options": {
            "1": "Small;Active;Darting;Dark eyes",
            "2": "Medium-sized;Penetrating;Light-sensitive eyes",
            "3": "Big;Round;Beautiful;Glowing eyes",
        },
    },
    11: {
        "question": "Describe your eyelashes",
        "options": {
            "1": "Scanty eyelashes",
            "2": "Moderate eyelashes",
            "3": "Thick/Fused eyelashes",
        },
    },
    12: {
        "question": "How often do you usually blink your eyes?",
        "options": {
            "1": "Excessive Blinking",
            "2": "Moderate Blinking",
            "3": "More or less stable",
        },
    },
    13: {
        "question": "Describe the typical condition of your cheeks",
        "options": {
            "1": "Wrinkled;Sunken",
            "2": "Smooth;Flat",
            "3": "Rounded;Plump",
        },
    },
    14: {
        "question": "Describe your nose",
        "options": {
            "1": "Crooked;Narrow",
            "2": "Pointed;Average",
            "3": "Rounded;Large open nostrils",
        },
    },
    15: {
        "question": "Describe your typical teeth and gums",
        "options": {
            "1": "Irregular;Protruding teeth;Receding gums",
            "2": "Medium-sized teeth;Reddish gums",
            "3": "Big;White;Strong teeth;Healthy gums",
        },
    },
    16: {
        "question": "How are your lips usually?",
        "options": {
            "1": "Tight;Thin;Dry lips which chaps easily",
            "2": "Soft;Medium-sized",
            "3": "Large;Soft;Pink;Full",
        },
    },
    17: {
        "question": "Describe your nails",
        "options": {
            "1": "Dry;Rough;Brittle;Break",
            "2": "Sharp;Flexible;Pink;Lustrous",
            "3": "Thick;Oily;Smooth;Polished",
        },
    },
    18: {
        "question": "What is your usual appetite like?",
        "options": {
            "1": "Irregular;Scanty",
            "2": "Strong;Unbearable",
            "3": "Slow but steady",
        },
    },
    19: {
        "question": "What tastes do you usually like?",
        "options": {
            "1": "Sweet;Sour;Salty",
            "2": "Bitter;Astringent",
            "3": "Pungent",
        },
    },
}

def get_percentages(responses):
    print(responses, 'responses')
    count_vata = responses.count(0)  
    count_pitta = responses.count(1)
    count_kapha = responses.count(2)
    
    total_responses = len(responses)

    print(count_vata, count_pitta, count_kapha, total_responses)
    
    percent_vata = (count_vata * 100) / total_responses
    percent_pitta = (count_pitta * 100) / total_responses
    percent_kapha = (count_kapha * 100) / total_responses

    print(percent_vata, percent_pitta, percent_kapha)
    return { 
        'percent_0': percent_vata,
        'percent_1': percent_pitta,
        'percent_2': percent_kapha
    }


def get_ans(li):
    val = model.predict(li).item()
    praks = {
        0: "vata",
        1: "pitta",
        2: "kapha",
        3: "vata+pitta",
        4: "vata+kapha",
        5: "pitta+kapha",
    }
    return praks.get(val)


flag = False
i = 0
lis = []
limit = len(questions.items())
print(lis, limit)


def get_response(msg):
    global flag, i, limit, lis
    ans = {}

    if flag:
        if i != 0:
            try:
                user_input = int(msg)
                if user_input not in {1, 2, 3}:
                    raise ValueError(
                        "Invalid Input, Please select the one out of 1,2 and 3"
                    )
                lis.append(int(msg) - 1)
            except ValueError:
                error_message = (
                    "Invalid input for question. Please enter a valid integer."
                )
                print(error_message)
                ans = json.dumps({"answer": error_message})

        if i != 0 and len(lis) < i:
            question_obj = questions.get(i - 1)
            if question_obj:
                question_text = question_obj["question"]
                options = question_obj["options"]
                ans = json.dumps({"question": question_text, "options": options})
                return ans

        question_obj = questions.get(i)
        if question_obj:
            question_text = question_obj["question"]
            options = question_obj["options"]
            ans = json.dumps({"question": question_text, "options": options})
        if i == limit:
            flag = False
            praki = get_ans([lis])
            ps = get_percentages(lis)
            ans = json.dumps({
                'answer': 'Your prakriti is ' + praki,
                'ps': {
                    'vata': ps['percent_0'],
                    'pitta': ps['percent_1'],
                    'kapha': ps['percent_2']
                }
            })
        i += 1
        print(lis, limit, i, flag)
        return ans

    msg_list = []
    msg_list.append(msg)
    msg_count = vectorizer.transform(msg_list)
    tag = get_intent(msg_count)
    if tag == "prakriti":
        flag = True
    for intent in intents["intent"]:
        if tag == intent["tag"]:
            ans = random.choice(intent["responses"])
            return json.dumps({"answer": ans})
