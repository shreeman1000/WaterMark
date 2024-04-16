from parrot import Parrot
import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")
import os
import pickle


parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

def paraphraser(texts ):
    paraphrased = []
    for i , text in enumerate(texts):
        paraphrased_text = []
        sentences = nltk.sent_tokenize(text)
        for j , sentence in enumerate(sentences):
            paraphrased_sentence = []
            k = 0.1
            while(len(paraphrased_sentence) < 1):
                paraphrased_sentence =  parrot.augment(input_phrase=sentence, 
                                    diversity_ranker="levenshtein",
                                    do_diverse=False, 
                                    max_return_phrases = 1, 
                                    max_length=2000, 
                                    adequacy_threshold = 0.9-k, 
                                    fluency_threshold = 0.9-k,
                                    use_gpu=True)
                k+=0.1
                if(paraphrased_sentence is None):
                    paraphrased_sentence = []
            paraphrased_text.append(paraphrased_sentence[0][0])
        paraphrased_text = " ".join(paraphrased_text)
        paraphrased.append(paraphrased_text)
    return paraphrased
        
    


directories = ["Dataset/Attacked/PivotTranslation/"]
flag = False
for directory in directories:
    print(directory.split('/')[-2])
    for folder in os.listdir(directory):
        print("-->" , folder)
        for _file_name in os.listdir(directory + folder):
            file_name = f"{directory}/{folder}/{_file_name}"
            
            with open(file_name , "rb") as file:
                data = pickle.load(file)
            
            translated = paraphraser(data)
            
            dest = f"Dataset/Attacked/R{directory.split('/')[-2]}/{folder}"
            
            if not os.path.exists(dest):
                os.makedirs(dest)
                print(f"Folder '{dest}' created successfully.")
            else:
                print(f"Folder '{dest}' already exists.")
            dest = f'{dest}/{_file_name}'
            with open(dest , "wb") as wfile:
                pickle.dump(translated , wfile)
            if not flag:
                print(translated)
            flag = True
    