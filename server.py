#app.py
import random
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
from SentimentPredictor import SentimentPredictor, LSTM_Net
from SentimentPredictorV2 import SentimentPredictor
from DataCutter import DataCutter


app = Flask(__name__)

with open('line-secret/secret', 'r') as f:
    secret = f.readline()
with open('line-secret/token', 'r') as f:
    token = f.readline()

print('secret: {}'.format(secret))
print('token: {}'.format(token))

line_bot_api = LineBotApi(token)
handler = WebhookHandler(secret)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

data_cutter = DataCutter()
sentiment_predictor = SentimentPredictor()
pos_res = ['這麼爽?', '厲害厲害', '對阿', '哈哈哈']
neg_res = ['7噗噗', '不要那麼火爆', 'QQ', '不行嗎?']

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # print('userId: {}'.format(event.source.sender_id))

    if event.source.type == 'group':
        if event.source.sender_id == 'Cd0ae944d90475fb2e6065954c20e6459':
            print('got KaoKao message!')
            sen_cut = data_cutter.cut(event.message.text)
            data_cutter.save()

    print('test: {}'.format(event.message.text))
    # sentiment_predictor.input([sen_cut])
    predict = sentiment_predictor.getResult(event.message.text)
    print(predict)
    predice_bool = predict[0]['label'][:8]
    predict_score = predict[0]['score']
    if predice_bool == 'positive':
        # res = '+'
        res = random.choice(pos_res)
    else:
        # res = '-'
        res = random.choice(neg_res)
    print('result: {}'.format(res))

    # dis = abs(res_bool[1] - res_bool[2])
    # print('dis: {}'.format(dis))
    r = random.randint(1, 100)
    print('r: {}'.format(r))
    # if dis > 1e-2 or event.source.type == 'user':
    if (predict_score > 0.9 and r > 50) or event.source.type == 'user':
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=res))

@app.route("/test", methods=['GET'])
def test():
    h1 = '<h1>Hello World!</h1>'
    return h1


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(ssl_context=('server.crt', 'server.key'), debug=True)