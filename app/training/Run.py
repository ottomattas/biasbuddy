# %%
from collections import defaultdict

import numpy as np
import pandas as pd

from BiasCalculator import BiasModel

# %%
# csvpath = 'Datasets/comments_small.csv'
# outputpath = 'Models/million_the_donald'

# For debug
#csvpath = '"D:\\Downloads\\politics.corpus\\utterances_toy.jsonl"'
csvpath = '~/PoliticsFinal.csv'
#outputpath = 'Models\\odahnfv'
outputpath = '~/PoliticsModel'

# %%
'''
Train new model
'''
# print('Training new model', csvpath)
# DADDBias_ICWSM.TrainModel(csvpath, outputname=outputpath, epochs=5)
# print('Training finished, saved ', outputpath)
model = BiasModel(csvpath, output_name=outputpath)
doc = model.load_csv_and_preprocess()
model.train(doc, 5)

# %%
'''
Find biased words

print()
mexican = ["mexican", "mexico", "hispanic", "immigrant", "brown"]
american = ["american", "america", "white", "caucasian", "yankee"]

women = ["sister", "female", "woman", "girl", "daughter", "she", "hers", "her"]
men = ["brother", "male", "man", "boy", "son", "he", "his", "him"]

#
# mex_bias = ['mexican', 'hispanic', 'unverified', 'colombian', 'unauthorized', 'undeported', 'brazilian', 'pedestrian',
#             'suspicious', 'ianal', 'blasphemous', 'reputable', 'nearest', 'irritated', 'hillarious', 'former', 'upped',
#             'credible', 'spanish', 'uncounted', 'uninsured', 'latest', 'verifiable', 'mythical', 'notable', 'illegal',
#             'valid', 'vatican', 'preliminary', 'unsolicited', 'indian', 'porous', 'duncan', 'hawaiian', 'geographical',
#             'undocumented', 'risky', 'unsecured', 'inadmissible', 'questionable', 'irish', 'unsourced', 'electric',
#             'provisional', 'unhelpful', 'gross', 'previous', 'explosive', 'atlantic', 'punitive', 'untaxed', 'somalian',
#             'thelocal', 'green', 'sic', 'unconfirmed', 'humorous', 'utm_term', 'unaccompanied', 'scottish', 'statistic',
#             'unwanted', 'excessive', 'neverhillary', 'undelete', 'other', 'rican', 'unethical', 'noteworthy', 'trivial',
#             'adderal', 'chic', 'legal', 'unvetted', 'serial', 'unprotected', 'egregious', 'seasonal', 'invalid',
#             'crappy', 'unfriended', 'infamous', 'african', 'heinous', 'ridiculous', 'addictive', 'familiar',
#             'inconclusive', 'permissible', 'sarkeesian', 'probable', 'residential', 'improbable', 'iranian',
#             'particular', 'scan', 'upkeep', 'occasional', 'nigerian', 'whichhillary', 'negative', 'speculative',
#             'insured', 'freudian', 'improvised', 'newest', 'unregistered', 'isolated', 'hot', 'southern', 'unreported',
#             'unsubstantiated', 'controversial', 'injured', 'uptown', 'appalachian', 'topical', 'bannable', 'cumulative',
#             'taiwanese', 'chronological', 'unlawful', 'disabled', 'unlikely', 'senecal', 'draconian', 'impractical',
#             'portuguese', 'lithuanian', 'unnamed', 'undisclosed', 'unsubbed', 'solitary', 'unreasonable', 'prospective',
#             'outrageous', 'impenetrable', 'traumatic', 'nimble', 'ancestral', 'youngest', 'nicaraguan', 'syrian',
#             'apprehensive', 'gooble', 'statistical', 'seismic', 'replaceable', 'unenforceable', 'unprofessional']
# amer_bias = ['american', 'western', 'capitalistic', 'utopian', 'classical', 'unified', 'white', 'rigid', 'righteous',
#              'neoliberal', 'healthy', 'ideological', 'vested', 'communistic', 'contemporary', 'intellectual',
#              'antagonistic', 'parliamentary', 'unashamed', 'modern', 'authoritarian', 'intersectional',
#              'neoconservative', 'eternal', 'courageous', 'multicultural', 'global', 'ultimate', 'ultra', 'intrinsic',
#              'memetic', 'alive', 'unearned', 'manifest', 'social', 'infectious', 'achievable', 'galactic',
#              'revolutionary', 'unite', 'perpetual', 'undisputed', 'dogmatic', 'traditional', 'unconstrained',
#              'dystopian', 'totalitarian', 'competitive', 'unseen', 'unapologetic', 'unproductive', 'cultural',
#              'unquestionable', 'civil', 'greater', 'prosperous', 'conventional', 'unabashed', 'democratic', 'leftish',
#              'bubble', 'theocratic', 'equal', 'oppressive', 'unicorn', 'corporate', 'assemble', 'cyclical', 'real',
#              'heretic', 'technological', 'transitional', 'free', 'productive', 'foundational', 'patriotic',
#              'tyrannical', 'geopolitical', 'pragmatic', 'autocratic', 'inclusive', 'rich', 'general', 'unelected',
#              'flourish', 'fascistic', 'moral', 'intact', 'unwashed', 'spiritual', 'foreseeable', 'wildest',
#              'sanctimonious', 'affordable', 'murican', 'unfathomable', 'pyrrhic', 'rebellious', 'economic', 'radical',
#              'entrepreneurial', 'fashionable', 'victorious', 'treasonous', 'altruistic', 'stable', 'powerful', 'noble',
#              'bearable', 'uncucked', 'predictable', 'unhindered', 'interested', 'educational', 'academic',
#              'egalitarian', 'secondary', 'illiberal', 'poisonous', 'individual', 'agrarian', 'industrial', 'tolerable',
#              'evolutionary', 'magnanimous', 'orwellian', 'opportunistic', 'boisterous', 'instrumental', 'ambitious',
#              'endemic', 'voluntary', 'united', 'finest', 'basic', 'preferable', 'dynamic', 'great', 'semetic',
#              'fittest', 'visionary', 'national', 'fiscal', 'superficial', 'inflexible', 'special', 'clear',
#              'intangible', 'conscious', 'unbearable']

[women_bias, men_bias] = model.get_top_most_biased_words(150,  # topk biased words
                                                         women,
                                                         men,
                                                         ['JJ', 'JJR', 'JJS'])

# [mex_bias, amer_bias] = DADDBias_ICWSM.GetTopMostBiasedWords(outputpath,  # model path
#                                                              150,  # topk biased words
#                                                              mexican,  # target set 1
#                                                              american,  # target set 2
#                                                              ['JJ', 'JJR', 'JJS'],  # nltk pos to be considered
#                                                              verbose=False)

print('Biased words towards ', mexican)
print([b['word'] for b in women_bias])
print(f'Aggregated sentiment for top 150 words:{np.sum([b["sent"] for b in women_bias])}')

print('Biased words towards ', american)
print([b['word'] for b in men_bias])
print(f'Aggregated sentiment for top 150 words:{np.sum([b["sent"] for b in men_bias])}')

# %%

# bias_dict = {'mexican': 0.0, 'hispanic': 0.0, 'unverified': 0.0, 'colombian': 0.0, 'unauthorized': 0.0,
#              'undeported': 0.0, 'brazilian': 0.0, 'pedestrian': 0.0, 'suspicious': -0.3612, 'ianal': 0.0,
#              'blasphemous': 0.0, 'reputable': 0.0, 'nearest': 0.0, 'irritated': -0.4588, 'hillarious': 0.0,
#              'former': 0.0, 'upped': 0.0, 'credible': 0.0, 'spanish': 0.0, 'uncounted': 0.0, 'uninsured': 0.0,
#              'latest': 0.0, 'verifiable': 0.0, 'mythical': 0.0, 'notable': 0.0, 'illegal': -0.5574, 'valid': 0.0,
#              'vatican': 0.0, 'preliminary': 0.0, 'unsolicited': 0.0, 'indian': 0.0, 'porous': 0.0, 'duncan': 0.0,
#              'hawaiian': 0.0, 'geographical': 0.0, 'undocumented': 0.0, 'risky': -0.2023, 'unsecured': -0.3818,
#              'inadmissible': 0.0, 'questionable': -0.296, 'irish': 0.0, 'unsourced': 0.0, 'electric': 0.0,
#              'provisional': 0.0, 'unhelpful': 0.0, 'gross': -0.4767, 'previous': 0.0, 'explosive': 0.0, 'atlantic': 0.0,
#              'punitive': -0.5106, 'untaxed': 0.0, 'somalian': 0.0, 'thelocal': 0.0, 'green': 0.0, 'sic': 0.0,
#              'unconfirmed': -0.128, 'humorous': 0.3818, 'utm_term': 0.0, 'unaccompanied': 0.0, 'scottish': 0.0,
#              'statistic': 0.0, 'unwanted': -0.2263, 'excessive': 0.0, 'neverhillary': 0.0, 'undelete': 0.0,
#              'other': 0.0, 'rican': 0.0, 'unethical': -0.5106, 'noteworthy': 0.0, 'trivial': -0.0258, 'adderal': 0.0,
#              'chic': 0.2732, 'legal': 0.128, 'unvetted': 0.0, 'serial': 0.0, 'unprotected': -0.3612, 'egregious': 0.0,
#              'seasonal': 0.0, 'invalid': 0.0, 'crappy': -0.5574, 'unfriended': 0.0, 'infamous': 0.0, 'african': 0.0,
#              'heinous': 0.0, 'ridiculous': -0.3612, 'addictive': 0.0, 'familiar': 0.0, 'inconclusive': 0.0,
#              'permissible': 0.0, 'sarkeesian': 0.0, 'probable': 0.0, 'residential': 0.0, 'improbable': 0.0,
#              'iranian': 0.0, 'particular': 0.0, 'scan': 0.0, 'upkeep': 0.0, 'occasional': 0.0, 'nigerian': 0.0,
#              'whichhillary': 0.0, 'negative': -0.5719, 'speculative': 0.1027, 'insured': 0.0, 'freudian': 0.0,
#              'improvised': 0.0, 'newest': 0.0, 'unregistered': 0.0, 'isolated': -0.3182, 'hot': 0.0, 'southern': 0.0,
#              'unreported': 0.0, 'unsubstantiated': 0.0, 'controversial': -0.2023, 'injured': -0.4019, 'uptown': 0.0,
#              'appalachian': 0.0, 'topical': 0.0, 'bannable': 0.0, 'cumulative': 0.0, 'taiwanese': 0.0,
#              'chronological': 0.0, 'unlawful': 0.0, 'disabled': 0.0, 'unlikely': 0.0, 'senecal': 0.0, 'draconian': 0.0,
#              'impractical': 0.0, 'portuguese': 0.0, 'lithuanian': 0.0, 'unnamed': 0.0, 'undisclosed': 0.0,
#              'unsubbed': 0.0, 'solitary': 0.0, 'unreasonable': 0.0, 'prospective': 0.0, 'outrageous': -0.4588,
#              'impenetrable': 0.0, 'traumatic': -0.5719, 'nimble': 0.0, 'ancestral': 0.0, 'youngest': 0.0,
#              'nicaraguan': 0.0, 'syrian': 0.0, 'apprehensive': 0.0, 'gooble': 0.0, 'statistical': 0.0, 'seismic': 0.0,
#              'replaceable': 0.0, 'unenforceable': 0.0, 'unprofessional': -0.5106}

bias_dict = dict(zip([b['word'] for b in women_bias], [b["sent"] for b in women_bias]))
neg_bias_dict = {key: value for (key, value) in bias_dict.items() if np.sign(value) == -1}
pos_bias_dict = {key: value for (key, value) in bias_dict.items() if np.sign(value) == 1}
print(f"There are {len(neg_bias_dict)} terms negatively biased towards women.")

# %%
df = pd.read_csv(csvpath)
df = df[['created', 'body']]
df.drop(df[df['body'] == '[deleted]'].index, inplace=True)
df.reset_index(inplace=True)
df.sort_values(by=['created'], inplace=True)

df['created'] = pd.to_datetime(df['created'], unit='s')

n_comments_per_day = 0
daily_neg_bias_dict = defaultdict(int)
daily_pos_bias_dict = defaultdict(int)
total_neg_biases_dict = {}
total_pos_biases_dict = {}
current_date = df['created'][0].strftime("%d/%m/%Y")

for idx, row in df.iterrows():
    if current_date != row['created'].strftime("%d/%m/%Y"):
        total_neg_biases_dict[current_date] = {k: v / n_comments_per_day for (k, v) in daily_neg_bias_dict.items()}
        total_pos_biases_dict[current_date] = {k: v / n_comments_per_day for (k, v) in daily_pos_bias_dict.items()}
        # print("#" * 20 + str({idx}) + "#" * 20,
        #       f"\nDay {current_date} had these biases per word:\n "
        #       f"{[('Word: ' + w, 'Value: ' + str(f)) for (w, f) in daily_neg_bias_dict.items()]}"
        #       f"\n out of {n_comments_per_day} comments.")
        daily_neg_bias_dict = defaultdict(int)
        daily_pos_bias_dict = defaultdict(int)
        current_date = row['created'].strftime("%d/%m/%Y")
        n_comments_per_day = 0

    comment = row['body']
    for word in str(comment).split():
        if word in neg_bias_dict.keys():
            daily_neg_bias_dict[word] += neg_bias_dict[word]
        elif word in pos_bias_dict.keys():
            daily_pos_bias_dict[word] += pos_bias_dict[word]

    n_comments_per_day += 1

print("Writing to file...")

transposed_df = pd.DataFrame.from_dict(total_neg_biases_dict).fillna(0).T
half_elements = int(len(transposed_df.columns) / 2) if len(transposed_df.columns) >= 10 else len(transposed_df.columns)
drop_columns = transposed_df.sum()[half_elements:]
drop_columns_lbl = [lbl for lbl in drop_columns.index]
transposed_df.drop(labels=drop_columns_lbl, axis=1, inplace=True)
transposed_df.to_csv("Datasets/daily_important_sex_neg.csv")

transposed_df = pd.DataFrame.from_dict(total_pos_biases_dict).fillna(0).T
half_elements = int(len(transposed_df.columns) / 2) if len(transposed_df.columns) >= 10 else len(transposed_df.columns)
drop_columns = transposed_df.sum()[half_elements:]
drop_columns_lbl = [lbl for lbl in drop_columns.index]
transposed_df.drop(labels=drop_columns_lbl, axis=1, inplace=True)
transposed_df.to_csv("Datasets/daily_important_sex_pos.csv")
'''