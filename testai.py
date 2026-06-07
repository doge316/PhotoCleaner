import dashscope
from dashscope import MultiModalConversation
import requests

dashscope.api_key = ""

img_path = r"D:\PhotoCleaner\消除路人\训练集\test4.png"

messages = [
    {
        "role": "user",
        "content": [
            {"image": f"file://{img_path}"},
            {"text": "去除图中所有人物，保留背景，自然修复，返回修改后的图片"}
        ]
    }
]

rsp = MultiModalConversation.call(
    model="qwen-vl-plus",
    messages=messages
)

if rsp.code == 200:
    print("返回内容：", rsp.output.choices[0].message.content)
else:
    print("错误：", rsp.message)