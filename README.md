# WeissTranslate
### 6/6/24
Started training. I chose to do the data in JSON, and gathered some quick data to do an initial proof of concept. First issue I ran into was that the instance was running out of RAM and crashing while training. I solved this by decreasing the batch size from 16 to 8, and switching instances from default to L4(I would've preferred A100, but they were not available). There was also a weird issue with accelerate, with it being up to date but colab not recognizing that it was up to date. The solution I found for this on HF forums was just to pip update it, then restart the runtime and then not pip any other packages. Weird, but worked.
### 6/5/24
Exploring models to retrain. There weren't many specific Japanese to English models on huggingface, so I settled on using "facebook/mbart-large-50-many-to-many-mmt", which seems to be pretty popular. It uses 50 languages, so I will just take the tokenizer for Japanese and do the fine tuning on that
