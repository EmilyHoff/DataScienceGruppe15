from multiprocessing.dummy import active_children
from sklearn import svm
from sklearn.metrics import classification_report
import sys
import numpy as np
import wandb
from sklearn.metrics import accuracy_score
import re 

sys.path.insert(0,"../")

import main

df = None

sweep_configuration = {
    "method":"bayes",
    "metric":{"name":"accuracy","goal":"maximize"},
    "parameters":{
        "regC":{"values":[0.3,0.7,1,1.5]},
        "kernelFunc":{"values":["linear","poly","rbf","sigmoid"]},
        "iterMax":{"values":[3,15,30,150,300]},
        "maxSize":{"values":[150,250,300,500,700]}
    }
}


def supportVectorMachine(config):
    max_len = max([len(seq) for seq in df["Article encoded"]])
    max_len = config.maxSize
    print(f"First article: {df['Article encoded'][0]}")
    #padded_list = [seq + [0.0] * (max_len - len(seq)) for seq in df["Article encoded"]]
    
    articles = []
    
    for seq in df["Article encoded"]:
        toAppend = []
        if not isinstance(seq,list):
            for word in seq.split(","):
                toAppend.append(float(word.replace("[","").replace("]","")))
        else:
            for word in seq:
                toAppend.append(float(word.replace("[","").replace("]","")))
        articles.append(toAppend)
    
    for text in articles:
        for x in range(max_len - len(text)):
            text.append(0.0)
            
    X = [x[:max_len] for x in articles]    
    X = np.array(X)
    X = X.reshape((X.shape[0], -1))
    
    y = np.array(df["type"])
    
    clf = svm.SVC(C=config.regC,kernel=config.kernelFunc,max_iter=config.iterMax)
    
    clf.fit(X[:3*(len(df)//4)-1],y[:3*(len(df)//4)-1])
    
    y_pred = clf.predict(X[3*(len(df)//4)+1:])
    
    return accuracy_score(y_pred,y[3*(len(df)//4)+1:])
    

def wandbHandle():
    wandb.init(project='fake-news-SVM-tuning')
    accuracy = supportVectorMachine(wandb.config)
    wandb.log({
        'accuracy':accuracy
    })
    
    
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='fake-news-SVM-tuning')
    df = main.main() #Kræver at du laver en funktion inde i main der returnere et dataframe som vi kan bruge til rensning
    wandb.agent(sweep_id, function=wandbHandle, count=4)
    
    

