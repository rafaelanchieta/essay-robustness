import random
from util import Corpus
from adversarial import song, cake_recipe


def get_good_essays(essay):
    """
    Get good essays
    :param essay: essay to be analyzed
    :return: essays with score greater than or equal 680
    """
    return essay.loc[essay['score'] >= 680]


def get_unrelated_essay(essay, prompt):
    """
    Get unrelated essay
    :param prompt: prompt to be analyzed
    :param essay: essay to be analyzed
    :return: unrelated essay
    """
    unrelated_essay = essay[essay['prompt'] != prompt]['essay'].sample()
    while unrelated_essay.empty or unrelated_essay.isna().values.any():
        unrelated_essay = essay[essay['prompt'] != prompt]['essay'].sample()
    return unrelated_essay.iloc[0][0]


def add_unrelated_text(essay, position):
    """
    Add unrelated text to the essay
    :param essay: essay to be modified
    :param position: position to be added (0: begin, 1: middle, 2: end)
    :return: essay with unrelated text
    """
    cont = 0
    for idx, e in essay.iterrows():
        new_essay = []
        unrelated_essay = get_unrelated_essay(essay, e['prompt'])
        new_essay = essay['essay'].iloc[cont]
        if position == 0:  # begin
            new_essay.insert(0, unrelated_essay)            
        elif position == 1: # middle
            new_essay.insert(len(new_essay)//2, unrelated_essay)    
        else:  # end
            new_essay.append(unrelated_essay)
        essay.at[idx, 'essay'] = new_essay
        cont += 1
    essay.to_csv('essay-br/splits/add_end.csv', index=False, header=True)
        
def repeat_text(essay, position):
    """
    Repeat text in the essay
    :param essay: essay to be modified
    :param position: position to be added (0: begin, 1: middle, 2: end)
    :return: essay with repeated text
    """
    cont = 0
    for idx, e in essay.iterrows():
        new_essay = []
        new_essay = essay['essay'].iloc[cont]
        if position == 0:  # begin
            new_essay.insert(0, new_essay[0])            
        elif position == 1: # middle
            new_essay.insert(len(new_essay)//2, new_essay[len(new_essay)//2])    
        else:  # end
            new_essay.append(new_essay[-1])
        essay.at[idx, 'essay'] = new_essay
        cont += 1
    essay.to_csv('essay-br/splits/repeat_end.csv', index=False, header=True)


def delete_text(essay, position):
    """
    Delete text in the essay
    :param essay: essay to be modified
    :param position: position to be added (0: begin, 1: middle, 2: end)
    :return: essay with deleted text
    """
    cont = 0
    for idx, e in essay.iterrows():
        new_essay = []
        new_essay = essay['essay'].iloc[cont]
        if position == 0:  # begin
            new_essay.pop(0)            
        elif position == 1: # middle
            new_essay.pop(len(new_essay)//2)    
        else:  # end
            new_essay.pop(-1)
        essay.at[idx, 'essay'] = new_essay
        cont += 1
    essay.to_csv('essay-br/splits/delete_end.csv', index=False, header=True)


def add_song_or_recipe(essay, position, type):
    """
    Add song or recipe to the essay
    :param essay: essay to be modified
    :param position: position to be added (0: begin, 1: middle, 2: end)
    :param type: type of text to be added (0: song, 1: recipe)
    :return: essay with song or recipe
    """
    cont = 0
    value = 0
    if type == 0:
        value = song()
    else:
        value = cake_recipe()
    for idx, e in essay.iterrows():
        new_essay = []
        new_essay = essay['essay'].iloc[cont]
        if position == 0:  # begin
            new_essay.insert(0, value.strip())            
        elif position == 1: # middle
            new_essay.insert(len(new_essay)//2, value)    
        else:  # end
            new_essay.append(value)
        essay.at[idx, 'essay'] = new_essay
        cont += 1
    essay.to_csv('essay-br/splits/add_cake.csv', index=False, header=True)


def shuffle_essay(essay):
    """
    Shuffle essay
    :param essay: essay to be shuffled
    :return: shuffled essay
    """
    cont = 0
    for idx, e in essay.iterrows():
        new_essay = []
        new_essay = essay['essay'].iloc[cont]
        random.shuffle(new_essay)
        essay.at[idx, 'essay'] = new_essay
        cont += 1
    essay.to_csv('essay-br/splits/shuffle.csv', index=False, header=True)

if __name__ == "__main__":
    c = Corpus()
    train, valid, test = c.read_splits()
    good_essays = get_good_essays(test)
    good_essays.to_csv('essay-br/splits/good_essays.csv', index=False, header=True)
    # get_unrelated_essay(good_essays, good_essays['prompt'].iloc[0])
    # add_unrelated_text(good_essays, 0)
    # repeat_text(good_essays, 2)
    # delete_text(good_essays, 2)
    # add_song_or_recipe(good_essays, 1, 1)
    # shuffle_essay(good_essays)
