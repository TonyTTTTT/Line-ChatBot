from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline

class SentimentPredictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
        self.model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese') # or other models above
        self.sentiment_classifier = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer)

    def getResult(self, x):
        res = self.sentiment_classifier(x)

        return res


if __name__ == '__main__':
    sentiment_predictor = SentimentPredictor()
    res = sentiment_predictor.getResult('好好玩')
    print(res)