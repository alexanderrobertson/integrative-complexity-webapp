import flask
import pickle
import eli5
import os

from collections import Counter, defaultdict
import re
import pandas as pd
import spacy


# Load the dif/int wordlists from Conway and my own and combine them.
# This really should not be happening!
# I should just save the combined ones and load those.
diff = pd.read_csv('ic_code/data/expanded_diff_list.csv')
integ = pd.read_csv('ic_code/data/expanded_integ_list.csv')
diff2 = pd.read_csv('ic_code/data/conway_dif_list.csv')
integ2 = pd.read_csv('ic_code/data/conway_int_list.csv')
diff = pd.concat([diff, diff2])
integ = pd.concat([integ, integ2])
diff.reset_index(inplace=True, drop=True)
integ.reset_index(inplace=True, drop=True)

# Load the smallest English model for POS tagging.
# Slightly less accurate but only 50mb
nlp = spacy.load('en_core_web_sm', disable=['ner',])

# Set the model type. 
model_type = 'Vocab+FullPOS'

# Load the pre-trained model.
with open(f'ic_code/models/{model_type}_xbgoost.pickle', 'rb') as f:
    model = pickle.load(f)

# Set up Flask app.
app = flask.Flask(__name__, template_folder='templates')

# The root page.
@app.route('/', methods=['GET', 'POST'])
def home():

    if flask.request.method == 'GET':
        #If just loading the page, show blank and ask for input.
        return flask.render_template('home.html')

    if flask.request.method == 'POST':

        # Otherwise, get the input and process it.
        tocode = flask.request.form.get('tocode')

        # Turn it into a dataframe with one row.
        df = process_text(str(tocode))

        # No real need to filter it because only one model
        # But ELI5 expects a vectoriser-type object.
        vec = FilterFeatures(df, model_type, 0)
        
        # Clean up the feature names for rendering in HTML.
        names = [i.replace('vocab_', 'v_') for i in vec.feature_names]
        names = [i.replace('tag_norm_', 't_') for i in names]

        # Get the actual explanation of predictions.
        explain = eli5.explain_prediction(model, str(tocode), vec=vec, feature_names=names)

        p = model.predict(vec.f)[0]

        # Convert it to HTML
        html = eli5.formatters.html.format_as_html(explain, show_feature_values=True,
                                                   show=('transition_features', 
                                                         'targets',
                                                         'feature_importances',
                                                         'decision_tree'))
        # Clean up the HTML a bit more to save space.
        html = html.replace('Contribution<sup>?</sup>', 'Contrib.')
        html = html.replace('top features', '')
        html = re.sub(pattern=r', score <b>-?\d\.\d\d\d</b>', string=html, repl='')
        html = re.sub(pattern=r'y=', string=html, repl='IC=')

        # Render it.
        return flask.render_template('home.html',
                                     data=flask.Markup(html),
                                     ta_content=tocode,
                                     prediction=p)

@app.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

def process_text(text, tags=True, vocab=True):
    # Turns a single string into a DF where the first row is feature values
    # and column names are the feature names.
    # Calls the two functions below.
    all_data = []
    
    data_to_add = {}

    parsed = nlp(text)

    if tags:
        for k1,v1 in get_tag_stats(parsed).items():
            for k2,v2 in v1.items():
                data_to_add[f"{k1}_{k2}"] = v2
    if vocab:
        for k,v in vocab_feature(parsed).items():
            data_to_add[k] = v

        
    all_data.append(data_to_add)
        
    all_data = pd.DataFrame(all_data)
    
    return all_data

def get_tag_stats(parsed_doc):
    tag_tags = ["ADD", "AFX", "BES", "CC", "CD", "DT", "EX", "FW", "GW",
                "HVS", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP",
                "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP",
                "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH",
                "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$",
                "WRB", "XX"]

    tag = Counter({k: 0 for k in tag_tags})
    tag.update(Counter([i.tag_ for i in parsed_doc]))
    tag_norm = {k: (v / sum(tag.values())) for k, v in tag.items()}
    tag_norm = {k:v for k,v in tag_norm.items() if k in tag_tags}

    return {'tag_norm': tag_norm}


def vocab_feature(parsed):
    a = ' '.join([i.lemma_ for i in parsed])
    
    vocab_features = defaultdict(int)
    
    int_f = 0
    dif_f = 0

    for w in set(integ.lemma):
        f_name = f"vocab_int_{w.replace(' ', '_')}"
        if w in a:
            vocab_features[f_name] = 1
            int_f = 1
        else:
            vocab_features[f_name] = 0


    for w in set(diff.lemma):
        f_name = f"vocab_dif_{w.replace(' ', '_')}"
        if w in a:
            vocab_features[f_name] = 1
        else:
            vocab_features[f_name] = 0
  
    if all([dif_f > 0, int_f > 0]):
        vocab_features['vocab_d1i1'] = 1
    else:
        vocab_features['vocab_d1i1'] = 1
        
    if dif_f > 0:
        vocab_features['vocab_d1'] = 1
    else:
        vocab_features['vocab_d1'] = 0

    if int_f > 0:
        vocab_features['vocab_i1'] = 1
    else:
        vocab_features['vocab_i1'] = 0        
        
    
    return dict(vocab_features)


class FilterFeatures:
    def __init__(self, df, f_name, index=0):
        self.f_name = f_name
        self.index = index
        
        if f_name == 'Vocab+FullPOS':
            a = df[[i for i in df.columns if i.startswith(f'tag_norm')]]
            b = df[[i for i in df.columns if 'vocab_' in i]]
            self.f = pd.concat([a, b], axis=1)
            
            
        self.feature_names = list(self.f.columns)
        
    def transform(self, _):
        self.feature_names = list(self.f.columns)
        return pd.DataFrame(self.f.iloc[self.index]).T
        

if __name__ == '__main__':
    app.run(debug=False)