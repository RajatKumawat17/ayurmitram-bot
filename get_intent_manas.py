from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
import json
import random
import json
import random
import numpy as np
import pandas as pd
nl_model = load('./models/nlm')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("./datasets/findprakriti.csv")

X = df.drop('Dosha', axis ='columns')
y = df['Dosha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2);

model = RandomForestClassifier()
model.fit(X_train,y_train)

with open('intents.json', 'r') as f:
    intents = json.load(f)

intentList = {
    2: 'greeting',
    1: 'goodbye',
    6: 'thanks',
    0: 'about',
    4: 'name',
    3: 'help',
    5: 'prakriti'
}

df = pd.read_csv("./datasets/initial_sentences_intent.csv")
X_train=df.Sentence

def get_intent(msg_count):
    ans = np.argmax(nl_model.predict(msg_count,verbose = 0))
    return intentList.get(ans)

vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(X_train.values)
print(len(X_train.values))

questions = {
    0: {
        'question': "How is the usual appearance of hair?",
        'options': {
            '1': 'Dry;Black;Knotted;Brittle',
            '2': 'Straight;Oily',
            '3': 'Thick;Curly',
        }
    },
    1: {
        'question': "What is your typical body weight?",
        'options': {
            '1': 'Low- difficulties in gaining weight',
            '2': 'Moderate- no difficulties in gaining or losing weight',
            '3': 'Heavy- difficulties in losing weight',
        }
    },
    2: {
        'question': "What is your usual height?",
        'options': {
            '1': 'Short',
            '2': 'Average',
            '3': 'Tall',
        }
    },
    3: {
        'question': "What is your typical bone structure like?",
        'options': {
            '1': 'Small- prominent joints',
            '2': 'Medium bone structure',
            '3': 'Large - broad shoulders;heavy bone structure',
        }
    },
    4: {
        'question': "What is your typical complexion?",
        'options': {
            '1': 'Dark- tans easily',
            '2': 'Fair- sunburns easily',
            '3': 'White- pale;White- tans easily',
        }
    },
    5: {
        'question': "Describe the general feel of your skin",
        'options': {
            '1': 'Dry;Thin;Cool to touch;Rough',
            '2': 'Smooth;Warm;Oily T-zone',
            '3': 'Thick;moist-greasy;Cold',
        }
    },
    6: {
        'question': "Describe the general texture of your skin",
        'options': {
            '1': 'Dry;Pigments and aging',
            '2': 'Freckles;Many moles;redness and rashes',
            '3': 'Oily',
        }
    },
    7: {
        'question': "What is your typical hair color?",
        'options': {
            '1': 'Black;Dull',
            '2': 'Red;Light brown;Yellow',
            '3': 'Brown',
        }
    },
    8: {
        'question': "Describe the general appearance of your hair",
        'options': {
            '1': 'Dry;Black;Knotted;Brittle',
            '2': 'Straight;Oily',
            '3': 'Thick;Curly',
        }
    },
    9: {
        'question': "Describe the typical shape of your face",
        'options': {
            '1': 'Long;Angular;Thin',
            '2': 'Heart-shaped;Pointed chin',
            '3': 'Large;Round;Full',
        }
    },
    10: {
        'question': "Describe your eyes generally",
        'options': {
            '1': 'Small;Active;Darting;Dark eyes',
            '2': 'Medium-sized;Penetrating;Light-sensitive eyes',
            '3': 'Big;Round;Beautiful;Glowing eyes',
        }
    },
    11: {
        'question': "Describe your eyelashes typically",
        'options': {
            '1': 'Scanty eyelashes',
            '2': 'Moderate eyelashes',
            '3': 'Thick/Fused eyelashes',
        }
    },
    12: {
        'question': "How often do you usually blink your eyes?",
        'options': {
            '1': 'Excessive Blinking',
            '2': 'Moderate Blinking',
            '3': 'More or less stable',
        }
    },
    13: {
        'question': "Describe the typical condition of your cheeks",
        'options': {
            '1': 'Wrinkled;Sunken',
            '2': 'Smooth;Flat',
            '3': 'Rounded;Plump',
        }
    },
    14: {
        'question': "Describe your nose typically",
        'options': {
            '1': 'Crooked;Narrow',
            '2': 'Pointed;Average',
            '3': 'Rounded;Large open nostrils',
        }
    },
    15: {
        'question': "Describe your typical teeth and gums",
        'options': {
            '1': 'Irregular;Protruding teeth;Receding gums',
            '2': 'Medium-sized teeth;Reddish gums',
            '3': 'Big;White;Strong teeth;Healthy gums',
        }
    },
    16: {
        'question': "How are your lips usually?",
        'options': {
            '1': 'Tight;Thin;Dry lips which chaps easily',
            '2': 'Soft;Medium-sized',
            '3': 'Large;Soft;Pink;Full',
        }
    },
    17: {
        'question': "Describe your nails typically",
        'options': {
            '1': 'Dry;Rough;Brittle;Break',
            '2': 'Sharp;Flexible;Pink;Lustrous',
            '3': 'Thick;Oily;Smooth;Polished',
        }
    },
    18: {
        'question': "What is your usual appetite like?",
        'options': {
            '1': 'Irregular;Scanty',
            '2': 'Strong;Unbearable',
            '3': 'Slow but steady',
        }
    },
    19: {
        'question': "What tastes do you usually like?",
        'options': {
            '1': 'Sweet;Sour;Salty',
            '2': 'Bitter;Astringent',
            '3': 'Pungent',
        }
    },
}
