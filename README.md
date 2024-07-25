# Sentiment Analysis - Inside Out 2

# Step 1: Pre-processing & Appending Data 

#### There are datasets available for emotion analysis; however, they do not categorize all the emotions represented in Inside Out 2. The list of emotions includes: [joy, sadness, anger, fear, disgust, embarrassment, anxiety, nostalgia, envy, boredom]. 

### https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data 

#### Emotions dataset analyzes Twitter messages and assigns the predominant emotion that’s associated with the text. Dataset Covers: [joy, sadness, love, anger, fear, surprise]. 

#### Also contains upwards of 393,000 entries, so augmenting synthetically generated text for the remaining categories won’t influence the model dramatically, as it’ll have a considerably smaller example space. A possible workaround is marginalizing the space occupied by the original dataset, so that each category has roughly the same number of examples. 

### https://huggingface.co/datasets/google-research-datasets/go_emotions 

#### The GoEmotions dataset contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral. The raw data and the smaller, simplified version of the dataset are included with predefined train/val/test splits. 

#### This dataset contains examples for [disgust, embarrassment, nervousness], so it’s plausible to append these to an aggregated dataset using the initial spread of emotions from the upper dataset. It additionally covers a wide range of emotions on the spectrum, so it might be possible to classify these under one of the prospective categories. 

#### For the remaining datasets, we can perform synthetic generation in the intended format, so that it’ll be easier to manually tokenize the data. Having GPT analyze the format of the existing data for replication is the easiest path. 

# Step 2: Developing a Model for Emotion Analysis 

#### I can try developing a model using different architectures: LSTM, RNNs, transformers, etc., and determine the model with the highest accuracy, and utilize that for emotion classification. 

# Step 3: Creating a Questionnaire/Dynamic Emotional Interface 

#### Basically, the different characters from Inside Out will pop up and ask questions to the user, similar to Akinator, and depending on the information provided by the user with their speech (maybe consider tonality as well), it will ask another question to follow up (there could be some level of variance implemented here). After it’s asked enough questions to gather what character you would be, it will essentially reveal to you (similar to Buzzfeed quizzes) what character you are. 

#### This process would require creating an intelligent agent that could adapt to the propensity demonstrated by the individual. If they exhibit an inclination towards a specific emotion, the agent could try to stray them away, but if they happen to edge closer to that (perhaps defined by a confidence interval in that emotion), then the decision can be finalized. 
