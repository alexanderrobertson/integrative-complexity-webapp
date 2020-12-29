from collections import Counter, defaultdict
import re
import pandas as pd
import spacy

diff = pd.read_csv('/home/alexander/Dropbox/bell/reddit/ic_code/data/expanded_diff_list.csv')
integ = pd.read_csv('/home/alexander/Dropbox/bell/reddit/ic_code/data/expanded_integ_list.csv')

diff2 = pd.read_csv('/home/alexander/Dropbox/bell/reddit/ic_code/data/conway_dif_list.csv')
integ2 = pd.read_csv('/home/alexander/Dropbox/bell/reddit/ic_code/data/conway_int_list.csv')

diff = pd.concat([diff, diff2])
integ = pd.concat([integ, integ2])

diff.reset_index(inplace=True, drop=True)
integ.reset_index(inplace=True, drop=True)

nlp = spacy.load('en_core_web_lg', disable=['ner', ])

def process_dataframe(dataframe, text_col, tags=True, vocab=True, subtree=True):
    all_data = []
    
    for t in dataframe[text_col]:
    
        data_to_add = {}
    
        parsed = nlp(t)
    
        if tags:
            for k1,v1 in get_tag_stats(parsed).items():
                for k2,v2 in v1.items():
                    data_to_add[f"{k1}_{k2}"] = v2

        if vocab:
            for k,v in vocab_feature(t).items():
                data_to_add[k] = v
        
        if subtree:    
            for k,v in get_subtree_stats(parsed).items():
                data_to_add[k] = v

        all_data.append(data_to_add)
        
    all_data = pd.DataFrame(all_data)
    
    return all_data


def process_text(text, tags=True, vocab=True, subtree=True):
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

    if subtree:
        for k,v in get_subtree_stats(parsed).items():
            data_to_add[k] = v
        
    all_data.append(data_to_add)
        
    all_data = pd.DataFrame(all_data)
    
    return all_data


def get_subtree_stats(parsed_doc):
    accepted = ['det', 'nsubj', 'aux', 'advmod', 'cc', 'amod', 'punct', 'pobj', 'prep_pobj', 'mark', 'dobj', 'poss', 'neg', 'det_pobj', 'compound', 'prep_det_pobj', 'auxpass', 'conj', 'acomp', 'det_nsubj', 'poss_pobj', 'nsubjpass', 'prep_poss_pobj', 'det_dobj', 'amod_pobj', 'prep_amod_pobj', 'det_amod_pobj', 'prep_det_amod_pobj', 'prt', 'advmod_amod', 'advmod_advmod', 'advmod_acomp', 'amod_dobj', 'poss_dobj', 'compound_pobj', 'expl', 'nummod', 'prep_compound_pobj', 'case', 'det_pobj_prep_pobj', 'poss_nsubj', 'prep', 'prep_det_pobj_prep_pobj', 'dative', 'det_amod_dobj', 'pobj_cc_conj', 'amod_cc_conj', 'predet', 'amod_nsubj', 'prep_pobj_cc_conj', 'attr', 'det_amod_attr', 'det_dobj_prep_pobj', 'appos', 'compound_compound', 'det_compound_pobj', 'det_amod_nsubj', 'poss_amod_pobj', 'prep_poss_amod_pobj', 'pcomp', 'prep_det_compound_pobj', 'poss_case', 'nsubj_relcl', 'det_nsubj_prep_pobj', 'det_attr', 'acomp_prep_pobj', 'npadvmod', 'pobj_prep_pobj', 'preconj', 'prep_pobj_prep_pobj', 'det_pobj_prep_det_pobj', 'det_nsubjpass', 'amod_conj', 'oprd', 'det_conj', 'aux_xcomp', 'nsubj_cc_conj', 'dative_pobj', 'det_dobj_prep_det_pobj', 'det_npadvmod', 'compound_nsubj', 'det_poss_case', 'det_compound_nsubj', 'advmod_prep_pobj', 'agent_pobj', 'conj_prep_pobj', 'acomp_cc_conj', 'conj_cc_conj', 'aux_xcomp_dobj', 'intj', 'det_amod_pobj_prep_pobj', 'poss_amod_dobj', 'csubj_acomp', 'nsubj_relcl_prep_pobj', 'nsubj_ccomp', 'advmod_acomp_prep_pobj', 'dobj_prep_pobj', 'poss_nsubjpass', 'amod_compound', 'nsubj_prep_pobj', 'det_compound_dobj', 'prep_pcomp', 'compound_conj', 'poss_case_pobj', 'amod_nsubj_prep_compound_pobj', 'neg_advmod', 'mark_nsubj_advcl', 'compound_dobj', 'csubj_amod_dobj', 'det_amod_amod_pobj', 'conj_pobj', 'prep_poss_case_pobj', 'nsubj_appos', 'amod_amod_pobj', 'poss_conj', 'advmod_conj', 'det_pobj_prep_poss_pobj', 'pcomp_dobj', 'poss_attr', 'poss_amod_nsubj', 'prep_nummod_pobj', 'nummod_pobj', 'prep_amod_amod_pobj', 'det_attr_prep_pobj', 'conj_dobj', 'prep_amod', 'advmod_cc_conj', 'nsubj_relcl_dobj', 'det_amod_conj', 'det_amod_dobj_prep_pobj', 'prep_pcomp_dobj', 'quantmod', 'csubj_dobj', 'prep_det_amod_amod_pobj', 'preconj_advmod', 'prep_prep_pobj', 'nmod', 'det_appos', 'amod_dobj_prep_pobj', 'nsubj_aux_relcl', 'dobj_cc_conj', 'aux_relcl', 'det_dobj_prep_poss_pobj', 'det_pobj_prep_amod_pobj', 'ccomp', 'prep_prep_det_pobj', 'oprd_prep_pobj', 'xcomp_dobj', 'mark_nsubj_advcl_prep_pobj', 'acomp_prep_det_pobj', 'det_conj_prep_pobj', 'dobj_prep_det_pobj', 'conj_det_pobj', 'nsubj_aux_ccomp', 'aux_xcomp_acomp', 'det_pobj_cc_conj', 'prep_compound_compound_pobj', 'compound_compound_pobj', 'advmod_nsubj_advcl', 'compound_appos', 'dobj_nsubj_ccomp', 'advmod_acomp_cc_conj', 'pobj_prep_det_pobj', 'xcomp', 'aux_relcl_dobj', 'amod_cc_conj_pobj', 'poss_dobj_prep_pobj', 'advmod_advmod_cc', 'poss_appos', 'nsubj_relcl_prep_det_pobj', 'prep_det_advmod_amod_pobj', 'det_advmod_amod_pobj', 'compound_compound_nsubj', 'aux_xcomp_prep_pobj', 'prep_pobj_prep_det_pobj', 'cc_advmod', 'det_nsubj_prep_poss_pobj', 'det_nsubj_cc_conj', 'prep_det_pobj_cc_conj', 'prep_det_poss_case_pobj', 'det_poss_case_pobj', 'dep', 'prep_amod_cc_conj_pobj', 'det_nsubj_prep_det_pobj', 'acl', 'amod_npadvmod', 'poss_pobj_prep_pobj', 'attr_prep_pobj', 'nsubj_prep_det_pobj', 'aux_advcl_dobj', 'advmod_amod_cc_conj', 'advmod_prep_det_pobj', 'det_amod_attr_prep_pobj', 'conj_det_dobj', 'advmod_amod_nsubj', 'prep_pobj_prep_poss_pobj', 'pobj_prep_poss_pobj', 'poss_compound_pobj', 'nsubj_aux_relcl_dobj', 'det_pobj_prep_compound_pobj', 'det_amod_amod_dobj', 'det_pobj_appos', 'amod_dobj_prep_compound_pobj', 'compound_compound_dobj', 'amod_attr', 'advmod_dobj', 'aux_xcomp_det_dobj', 'prep_poss_compound_pobj', 'amod_nsubjpass', 'det_amod_nsubjpass', 'agent_det_pobj', 'pobj_nsubj_relcl', 'poss_dobj_cc_conj', 'ROOT', 'advmod_cc', 'det_conj_prep_det_pobj', 'acl_prep_pobj', 'dative_det_pobj', 'nummod_npadvmod', 'attr_prep_det_pobj', 'mark_nsubj_advcl_acomp', 'dobj_prep_pobj_cc_conj', 'nsubjpass_cc_conj', 'conj_dobj_prep_pobj', 'poss_nsubj_prep_pobj', 'mark_nsubj_ccomp_acomp', 'det_amod_cc_conj_pobj', 'pcomp_acomp', 'prep_pobj_prep_amod_pobj', 'pobj_prep_amod_pobj', 'prep_prep_poss_pobj', 'det_attr_prep_det_pobj', 'advmod_advmod_advmod', 'nsubj_ccomp_acomp', 'acomp_prep_poss_pobj', 'aux_xcomp_dobj_prep_pobj', 'prep_pcomp_det_dobj', 'pcomp_det_dobj', 'conj_prep_det_pobj', 'advmod_nsubj_advcl_dobj', 'det_amod_npadvmod', 'prep_amod_pobj_prep_pobj', 'amod_pobj_prep_pobj', 'agent_amod_pobj', 'nmod_cc_conj', 'prep_det_pobj_appos', 'poss_pobj_cc_conj', 'poss_amod_amod_pobj', 'prep_poss_pobj_prep_pobj', 'conj_poss_pobj', 'conj_amod_dobj', 'pobj_cc_conj_prep_pobj', 'prep_pobj_cc_conj_pobj', 'compound_pobj_cc_conj', 'nsubj_nsubj_relcl', 'aux_xcomp_advmod', 'dobj_nsubj_relcl_prep_pobj', 'det_poss_case_nsubj', 'det_oprd', 'prep_pcomp_acomp', 'det_poss_case_dobj', 'dobj_det_nsubj_relcl', 'mark_nsubjpass_auxpass_advcl', 'det_compound_pobj_prep_pobj', 'det_advmod_amod_attr', 'advmod_oprd', 'nsubj_relcl_acomp', 'prep_det_pobj_cc', 'prep_pobj_nsubj_relcl', 'pobj_nsubj_relcl_prep', 'pobj_cc_amod_conj', 'prep_amod_compound_pobj', 'amod_compound_pobj', 'poss_case_dobj', 'prep_poss_pobj_cc_conj', 'dobj_nsubj_relcl_aux_xcomp', 'predet_det_pobj', 'predet_det_nsubj', 'det_compound_compound_pobj', 'det_attr_prep_poss_pobj']

    data = {f'sub_{j}':0 for j in accepted}
    
    for w in parsed_doc:
        subtree_span = parsed_doc[w.left_edge.i : w.right_edge.i + 1]

        j = '_'.join([i.dep_ for i in subtree_span])
        if j in accepted:
            data[f'sub_{j}'] = 1
            
    return data


def get_tag_stats(parsed_doc):

    # pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ",
    #             "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    #             "SCONJ", "SYM", "VERB", "X",]
    # pos = Counter({k: 0 for k in pos_tags})
    # pos.update(Counter([i.pos_ for i in parsed_doc]))
    # pos_norm = {k: (v / sum(pos.values())) for k, v in pos.items()}
    # # pos_norm = {k: 1 for k, v in pos.items()}
    # pos_norm = {k:v for k,v in pos_norm.items() if k in pos_tags}


    # dep_tags = ["acl", "acomp", "advcl", "advmod", "agent", "amod",
    #             "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp",
    #             "compound", "conj", "csubj", "csubjpass", "dative", "dep",
    #             "det", "dobj", "expl", "intj", "mark", "meta", "neg",
    #             "nounmod", "npmod", "nsubj", "nsubjpass", "nummod", "oprd",
    #             "parataxis", "pcomp", "pobj", "poss", "preconj", "predet",
    #             "prep", "prt", "quantmod", "relcl", "xcomp",
    #             "npadvmod", "nmod", ]
    
    # dep = Counter({k: 0 for k in dep_tags})
    # dep.update(Counter([i.dep_ for i in parsed_doc]))
    # dep_norm = {k: (v / sum(dep.values())) for k, v in dep.items()}
    # # dep_norm = {k: 1 for k, v in dep.items()}
    # dep_norm = {k:v for k,v in dep_norm.items() if k in dep_tags}


    tag_tags = ["ADD", "AFX", "BES", "CC", "CD", "DT", "EX", "FW", "GW",
                "HVS", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP",
                "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP",
                "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH",
                "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$",
                "WRB", "XX"]

    tag = Counter({k: 0 for k in tag_tags})
    tag.update(Counter([i.tag_ for i in parsed_doc]))
    tag_norm = {k: (v / sum(tag.values())) for k, v in tag.items()}
    # tag_norm = {k: 1 for k, v in tag.items()}
    tag_norm = {k:v for k,v in tag_norm.items() if k in tag_tags}

    return {'tag_norm': tag_norm,
            # 'pos_norm': pos_norm,
            # 'dep_norm': dep_norm,
            
            }


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
  
    # dif_f = sum([v*diff[diff.lemma == ' '.join(k.split('_')[2:])].score.values[0] for k,v in vocab_features.items() if k.startswith('vocab_dif')])
    # int_f = sum([v*integ[integ.lemma == ' '.join(k.split('_')[2:])].score.values[0] for k,v in vocab_features.items() if k.startswith('vocab_int')])
    
    # vocab_features['dif_score'] = round(dif_f)
    # vocab_features['int_score'] = round(int_f)
    
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
        
        
        if f_name == 'vocab':
            self.f = df[[i for i in df.columns if 'vocab_' in i]]
            
        if f_name == 'Dep. subtrees':
            self.f = df[[i for i in df.columns if all({i[0:4] == 'sub_', i[0:9] != 'sub_sent_'})]]
        if f_name in {'tag', 'dep', 'pos', 'sents_with_tag', 'sents_with_dep', 'sents_with_pos'}:
            self.f = df[[i for i in df.columns if i.startswith(f'{f_name}_norm')]]

        if f_name == 'LIWC semantic':
            self.f = df[['liwc_Tone','liwc_affect','liwc_posemo','liwc_negemo','liwc_anx','liwc_anger','liwc_sad','liwc_social','liwc_family','liwc_friend','liwc_female','liwc_male','liwc_cogproc','liwc_insight','liwc_cause','liwc_discrep','liwc_tentat','liwc_certain','liwc_differ','liwc_percept','liwc_see','liwc_hear','liwc_feel','liwc_drives','liwc_affiliation','liwc_achieve','liwc_power','liwc_reward','liwc_risk']]
        if f_name == 'LIWC syntax':
            self.f = df[['liwc_Analytic','liwc_function','liwc_pronoun','liwc_ppron','liwc_i','liwc_we','liwc_you','liwc_shehe','liwc_they','liwc_ipron','liwc_article','liwc_prep','liwc_auxverb','liwc_adverb','liwc_conj','liwc_negate','liwc_verb','liwc_adj','liwc_compare','liwc_interrog','liwc_number','liwc_quant','liwc_focuspast','liwc_focuspresent','liwc_focusfuture','liwc_relativ','liwc_motion','liwc_space','liwc_time','liwc_informal','liwc_swear','liwc_netspeak','liwc_assent','liwc_nonflu','liwc_filler']]

        if f_name == 'Vocab+FullPOS':
            a = df[[i for i in df.columns if i.startswith(f'tag_norm')]]
            b = df[[i for i in df.columns if 'vocab_' in i]]
            self.f = pd.concat([a, b], axis=1)
            
            
        self.feature_names = list(self.f.columns)
        
    def transform(self, _):
        self.feature_names = list(self.f.columns)
        return pd.DataFrame(self.f.iloc[self.index]).T
        

