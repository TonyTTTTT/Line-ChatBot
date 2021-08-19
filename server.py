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


sentiment_predictor = SentimentPredictor()
pos_res = ['這麼爽?', '厲害厲害', '對阿', '哈哈哈']
neg_res = ['7噗噗', '不要那麼火爆', 'QQ', '不行嗎?']

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # print('userId: {}'.format(event.source.sender_id))
    data_cutter = DataCutter()
    sen_cut = data_cutter.cut(event.message.text)
    if event.source.type == 'group':
        if event.source.sender_id == 'Cd0ae944d90475fb2e6065954c20e6459':
            print('got KaoKao message!')
            data_cutter.save()

    # print(sen_cut)
    sentiment_predictor.input([sen_cut])
    res_bool = sentiment_predictor.getResult()
    print(res_bool)
    if res_bool[0]:
        # res = '+'
        res = random.choice(pos_res)
    else:
        # res = '-'
        res = random.choice(neg_res)
    print('result: {}'.format(res))

    dis = abs(res_bool[1] - res_bool[2])
    print('dis: {}'.format(dis))
    r = random.randint(1, 100)
    print('r: {}'.format(r))
    # if dis > 1e-2 or event.source.type == 'user':
    if (dis > 0.5 and r > 90) or event.source.type == 'user':
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