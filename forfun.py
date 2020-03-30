from seg.newline.segmenter import NewLineSegmenter
import spacy

nlseg = NewLineSegmenter()
nlp = spacy.load('en')
nlp.add_pipe(nlseg.set_sent_starts, name='sentence_segmenter', before='parser')
my_doc_text = "Donald John Trump is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Trump was born and raised in Queens, a borough of New York City, and received a bachelor's degree in economics from the Wharton School."
doc = nlp(my_doc_text)
print(doc)
