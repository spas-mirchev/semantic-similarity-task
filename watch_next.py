import spacy

# We can use 'en_core_web_md' which is able to find similarities and differences better than ‘en_core_web_sm’.
nlp = spacy.load('en_core_web_md')

# We can save movies in dictionary title:description {'movie n' : 'description'}
with open('movies.txt', 'r') as f:
    data = f.readlines()
    splitted_data = [line.strip('\n').split(':') for line in data]
    data_dictionary = {line[0]: line[1] for line in splitted_data}


hulk_description = 'Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator.'


model_description = nlp(hulk_description)


def next_movie(description):
    # We compare the description of Planet Hulk with the descriptions of the other movies in the file
    # We record the results of the comparisons in a list_similarities
    # The keys in data_dictionary and list_similarities are identical
    # The function returns the key with the maximum value
    list_similarities = {}
    for sentence in data_dictionary:
        similarity = nlp(data_dictionary[sentence]).similarity(description)
        list_similarities[sentence] = similarity
        #print(sentence + " - ", similarity)

    title_most_similar_movie = max(
        list_similarities, key=list_similarities.get)
    return title_most_similar_movie


the_most_similar_movie = next_movie(model_description)
print(f'The movie most similar to Planet Hulk is: {the_most_similar_movie}')
