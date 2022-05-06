import contextlib
import wave
import requests
import time
from xml.etree import ElementTree
import argparse

context = ''

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-people', type=str)
    parser.add_argument('-style', type=str)
    parser.add_argument('-context', type=str)
    args = parser.parse_args().context
    style = parser.parse_args().style
    people = parser.parse_args().people
except NameError:
    pass

len(context)


class TextToSpeech(object):
    def __init__(self, subscription_key):
        self.subscription_key = subscription_key
        self.tts = args
        self.timestr = time.strftime("%Y%m%d-%H%M")
        self.access_token = None

    def get_token(self):
        fetch_token_url = "https://eastasia.api.cognitive.microsoft.com/sts/v1.0/issueToken"  # 终结点
        headers = {'Ocp-Apim-Subscription-Key': self.subscription_key}
        response = requests.post(fetch_token_url, headers=headers)
        self.access_token = str(response.text)

    def save_audio(self):
        if people is None:
            ShortName = 'zh-CN-YunxiNeural'
        else:
            ShortName = people

        base_url = 'https://eastasia.tts.speech.microsoft.com/'
        path = 'cognitiveservices/v1'
        constructed_url = base_url + path
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'riff-24khz-16bit-mono-pcm',
            'User-Agent': 'YOUR_RESOURCE_NAME'
        }
        xml_body = ElementTree.Element('speak', version='1.0')
        xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', 'en-us')
        voice = ElementTree.SubElement(xml_body, 'voice')
        voice.set('{http://www.w3.org/XML/1998/namespace}lang', 'en-US')
        voice.set(
            'name', ShortName
        )  # Short name for 'Microsoft Server Speech Text to Speech Voice (en-US, Guy24KRUS)'
        if style == None:
            pass
        else:
            mstts = ElementTree.SubElement(voice, 'mstts', nsmap='express-as')
            mstts.set('style', style)
        # A.set('role','Boy')
        voice.text = self.tts
        body = ElementTree.tostring(xml_body)

        response = requests.post(constructed_url, headers=headers, data=body)

        if response.status_code == 200:
            with open('sample-' + self.timestr + '.wav', 'wb') as audio:
                audio.write(response.content)
                print("\nStatus code: " + str(response.status_code) +
                      "\nYour TTS is ready for playback.\n")
        else:
            print(
                "\nStatus code: " + str(response.status_code) +
                "\nSomething went wrong. Check your subscription key and headers.\n"
            )
            print("Reason: " + str(response.reason) + "\n")

    def get_time(self):
        with contextlib.closing(
                wave.open('sample-' + self.timestr + '.wav', 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(duration)
        return duration


if __name__ == "__main__":
    subscription_key = 'f63905c334f24e9f884f17c37a545b0d'
    app = TextToSpeech(subscription_key)
    app.get_token()
    app.save_audio()
    time.sleep(1)
    app.get_time()