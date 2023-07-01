import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

# 加载inception_v3模型
model = models.inception_v3(pretrained=True)

# 设置模型为评估模式
model.eval()

# 设置图像转换操作
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

def predict(image):
    # 将numpy图像转换为PIL图像
    image = Image.fromarray(image)

    # 将图像转为张量
    img = transform(image).unsqueeze(0)

    # 检查是否支持MPS，如果支持则使用GPU进行运算
    if torch.backends.mps.is_available():
        img = img.to('mps')
        model.to('mps')
        
    # CUDA加速
    # if torch.cuda.is_available():
    #     img = img.to('cuda')
    #     model.to('cuda')

    # 在模型上进行前向传递
    with torch.no_grad():
        output = model(img)

    # 获取最可能的标签
    _, predicted = torch.max(output.data, 1)
    return idx2label[predicted.item()]

iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(299, 299)),
    outputs="text",
    title="Inception v3 Classifier",
    description="Upload an image, and this interface will classify it using the Inception v3 model trained on the ImageNet dataset.",
    theme="huggingface", # 设置主题为huggingface
    allow_flagging=False, # 不显示“标记”
    layout="vertical", # 将输入和输出组件垂直排列
)
iface.launch(share=True)
