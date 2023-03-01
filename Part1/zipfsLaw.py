

def commonWords(csvFile,quantiles=[0.05,0.95],generateGraph=True):
  df = pd.read_csv(csvFile)[:5]

  for y in ["content"]:
    for x in range(0,len(df)):
      tokens = nltk.tokenize.word_tokenize(df[y][x])
      allWordsDist = nltk.FreqDist(w.lower() for w in tokens)
      words = [[word,dict(allWordsDist.most_common())[word]] for word in dict(allWordsDist.most_common())]
      words = sorted(words,key=lambda k: k[1],reverse=True)

      lower = int(math.floor(words[0][1]-words[0][1]*quantiles[0]))
      upper = int(math.floor(words[0][1]-words[0][1]*quantiles[1]))

      print(f"lower = {lower} upper = {upper}----------------")

      mostCommon,leastCommon = [],[]
      
      for word in words:
        if word[1] > upper:
          mostCommon.append(word[0])
        elif word[1] < lower:
          leastCommon.append(word[0])
      
      print(mostCommon)
      print(leastCommon)


      for removeWord in mostCommon:
        print(f"before = {len(df[y][x])}")
        df[y][x] = df[y][x].replace(re.compile(rf"\b{removeWord}\b"),"")    #fiks det her
        print(f"after = {len(df[y][x])}")

      for removeWord in leastCommon:
        print(f"before = {len(df[y][x])}")
        df[y][x] = df[y][x].replace(re.compile(rf"\b{removeWord}\b"),"")
        print(f"after = {len(df[y][x])}"
      
      return df
              
###     
              
              
              
              
