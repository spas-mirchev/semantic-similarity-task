import spacy

# According to the official documentation "spaCy’s small pipeline packages (all packages that end in sm) don’t ship with word vectors, and only include context-sensitive tensors. This means you can still use the similarity() methods to compare documents, spans and tokens – but the result won’t be as good, and individual tokens won’t have any vectors assigned."
# Despite the warning "the model you are using has no word vectors loaded", we get a similarity score, but these may not provide useful similarity judgments.
# nlp = spacy.load('en_core_web_sm')

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word1))
print(word3.similarity(word2))
# The cat and the monkey look alike because they are both animals.
# The cat bears no resemblance to the fruit.
# The model assumes that monkeys eat bananas and therefore there is a significant similarity.
print('\n ---------------------------------------\n')
word4 = nlp("sitcom")
word5 = nlp("comedy")
word6 = nlp("show")
word7 = nlp("sketch")
print(word4.similarity(word5))
print(word4.similarity(word6))
print(word4.similarity(word7))
print(word5.similarity(word6))
print(word5.similarity(word7))
print(word6.similarity(word7))
# Sitcom and comedy are similar because sitcom is a portmanteau of situational comedy
# Sitcom is probably analyzed as representing two basic morphemes, which is why it is less similar to "show" and "sketch" /if "situation" dominates/
# If someone searched for something with one of the words comedy, show, sketch, we may recommend things containing one of the others.


#tokens = nlp('cat apple monkey banana ')
# tokens = nlp('show sketch comedy sitcom ')

# for token1 in tokens:
#     for token2 in tokens:
#         print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)
