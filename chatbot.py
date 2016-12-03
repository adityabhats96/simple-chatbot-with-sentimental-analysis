from flask import Flask,url_for ,redirect ,request

from nltk.chat.util import Chat, reflections
import random
from nltk import compat
import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import precision, recall, f_measure

final_string=""

pairs = (
    (r'We (.*)',
        ("What do you mean, 'we'?",
        "Don't include me in that!",
        "I wouldn't be so sure about that.")),

    (r'You should (.*)',
        ("Don't tell me what to do, buddy.",
        "Really? I should, should I?")),

    (r'You\'re(.*)',
        ("More like YOU'RE %1!",
        "Hah! Look who's talking.",
        "Come over here and tell me I'm %1.")),

    (r'You are(.*)',
        ("More like YOU'RE %1!",
        "Hah! Look who's talking.",
        "Come over here and tell me I'm %1.")),

    (r'How is NITK(.*)',
        ("Its a good college. One of the best in town.",
        "Amazing. Its the best college you can hope for.")),

    (r'(.*) eat',
        ("You could go to Nandini",
        "You could go to atithi",  
        "You could go to amul")),

    (r'(.*) Nandini',
        ("Nandini is next to the mech dept, you get some good maggi there.")),

    (r'(.*) Amul',
        ("Amul is next to the CS dept, you get some good samosas there.")),

    (r'(.*) Atithi',
        ("Atithi is next to Amul, you get non-veg food there, some good kebabs.")),

    (r'How (.*) IT department',
        ("Go check it out yourself",
        "Is this why you come to college",  
        "I guess its the same as every other department")),

    (r'I can\'t(.*)',
        ("You do sound like the type who can't %1.",
        "Hear that splashing sound? That's my heart bleeding for you.",
        "Tell somebody who might actually care.")),

    (r'I think (.*)',
        ("I wouldn't think too hard if I were you.",
        "You actually think? I'd never have guessed...")),

    (r'I (.*)',
        ("I'm getting a bit tired of hearing about you.",
        "How about we talk about me instead?",
        "Me, me, me... Frankly, I don't care.")),

    (r'How (.*)',
        ("How do you think?",
        "Take a wild guess.",
        "I'm not even going to dignify that with an answer.")),

    (r'What (.*)',
        ("Do I look like an encyclopedia?",
        "Figure it out yourself.")),

    (r'Why (.*)',
        ("Why not?",
        "That's so obvious I thought even you'd have already figured it out.")),

    (r'(.*)shut up(.*)',
        ("Make me.",
        "Getting angry at a feeble chatbot? Somebody's losing it.",
        "Say that again, I dare you.")),

    (r'Shut up(.*)',
        ("Make me.",
        "Getting angry at a feeble NLP assignment? Somebody's losing it.",
        "Say that again, I dare you.")),

    (r'Hello(.*)',
        ("Oh good, somebody else to talk to. Joy.",
        "'Hello'? How original...")),

    (r'Where(.*)NITK',
        ("Near Suratkal on the NH-66",
        "Between Mangalore and Padubidri")),

    (r'(.*)',
        ("I'm getting bored here. Become more interesting.",
        "Either become more thrilling or get lost, buddy.",
        "Change the subject before I die of fatal boredom."))
)



def rude_chat():
    rude_chatbot.conversation()

def demo():
    rude_chat()

class Chatbot(object):
    def __init__(self, pairs, reflections={}):
        """
        Initialize the chatbot.  Pairs is a list of patterns and responses.  Each
        pattern is a regular expression matching the user's statement or question,
        e.g. r'I like (.*)'.  For each such pattern a list of possible responses
        is given, e.g. ['Why do you like %1', 'Did you ever dislike %1'].  Material
        which is matched by parenthesized sections of the patterns (e.g. .*) is mapped to
        the numbered positions in the responses, e.g. %1.

        :type pairs: list of tuple
        :param pairs: The patterns and responses
        :type reflections: dict
        :param reflections: A mapping between first and second person expressions
        :rtype: None
        """

        self._pairs = [(re.compile(x, re.IGNORECASE),y) for (x,y) in pairs]
        self._reflections = reflections
        self._regex = self._compile_reflections()


    def _compile_reflections(self):
        sorted_refl = sorted(self._reflections.keys(), key=len,
                reverse=True)
        return  re.compile(r"\b({0})\b".format("|".join(map(re.escape,
            sorted_refl))), re.IGNORECASE)

    def _substitute(self, str):
        """
        Substitute words in the string, according to the specified reflections,
        e.g. "I'm" -> "you are"

        :type str: str
        :param str: The string to be mapped
        :rtype: str
        """

        return self._regex.sub(lambda mo:
                self._reflections[mo.string[mo.start():mo.end()]],
                    str.lower())

    def _wildcards(self, response, match):
        pos = response.find('%')
        while pos >= 0:
            num = int(response[pos+1:pos+2])
            response = response[:pos] + \
                self._substitute(match.group(num)) + \
                response[pos+2:]
            pos = response.find('%')
        return response
    def responds(self, str):
        """
        Generate a response to the user input.

        :type str: str
        :param str: The string to be mapped
        :rtype: str
        """

        # check each pattern
        for (pattern, response) in self._pairs:
            match = pattern.match(str)

            # did the pattern match?
            if match:
                resp = random.choice(response)    # pick a random response
                resp = self._wildcards(resp, match) # process wildcards

                # fix munged punctuation at the end
                if resp[-2:] == '?.': resp = resp[:-2] + '.'
                if resp[-2:] == '??': resp = resp[:-2] + '?'
                return resp

    # Hold a conversation with a chatbot

    def conversation(self, quit="quit"):
        input = ""
        while input != quit:
            input = quit
            try: input = compat.raw_input(">")
            except EOFError:
                print(input)
            if input:
                while input[-1] in "!.": input = input[:-1]
                print(self.responds(input))

rude_chatbot = Chatbot(pairs, reflections)

POLARITY_DATA_DIR = "/home/aditya/Projects/ACD/sentiment_analysis_python-master/sentiment_analysis_python-master/polarityData/rt-polaritydata"
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')
testfile=os.path.join(POLARITY_DATA_DIR,'train.txt')

Sentences=""

#this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select):
    posFeatures = []
    negFeatures = []
    Words=[]
    #http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    #breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    with open(RT_POLARITY_POS_FILE, 'r',encoding = "ISO-8859-1") as posSentences:
        for i in posSentences:
            posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
    with open(RT_POLARITY_NEG_FILE, 'r',encoding = "ISO-8859-1") as negSentences:
        for i in negSentences:
            negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)
 

    #selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    #trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)  

    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set) 
    
    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        #predicted = classifier.classify(features)
        #print(predicted)
        #testSets[predicted].add(i)
        '''if(len(testSets['pos'])>len(testSets['neg'])):
            print("positive sentiment")
        else:
            print("negative sentiment")'''
    
    #with open(testfile, 'r') as Sentences:
    for i in Sentences.split():
        print(i)
        Word = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            
        Word=feature_select(Word)
        Words.append(Word)
        predicted = classifier.classify(Word)
        testSets[predicted].add(i)
    #print(features)
    print(Words)
    predicted=classifier.classify(features)
    testSets[predicted]
    #print(len(testSets['pos']))
    #print(len(testSets['neg']))
    if(len(testSets['pos'])>len(testSets['neg'])):
        return "Hey!Even though i was rude to you, its good to know that you're in a good mood today."
    else:
        return "Look who's going into a bad mood over a simple chatbot."
    '''if(len(testSets['pos'])>len(testSets['neg'])):
        print("positive sentiment")
    else:
        print("negative sentiment")'''

    
    #print(Words)
    
    #print(testFeatures)
    #print(testSets['pos'])
    #print(testSets['neg'])
    #print(len(testSets['pos']))
    #prints metrics to show how well the feature selection did
    #print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    #print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    #print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    #print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    #print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    #print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])

    #classifier.show_most_informative_features(10)

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])

#tries using all words as the feature selection mechanism
#print 'using all words as features'


app = Flask(__name__)

@app.route("/aditya" , methods=['GET'])  # consider to use more elegant URL in your JS
def get_x(): 
 		 
   return redirect(url_for('static',filename='test.html'))

@app.route("/process" ,methods=['POST'])
def process():
    flag=0
    global final_string,Sentences
    text = request.form.get('TEXT')
    final_string = final_string + " " + text
    print ("\nBefore quit" +final_string)
    if(text=="quit"):
        Sentences = final_string
        print ("\nAfter quit" + Sentences)
        response = evaluate_features(make_full_dict)
        flag=1

    rude_chatbot = Chatbot(pairs, reflections)

    if(flag==1):
        return str(response)
    else:
        return str(rude_chatbot.responds(text))

if __name__ == "__main__":
    # here is starting of the development HTTP server
    app.run(debug='True')




