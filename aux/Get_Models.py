from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")


model.save_pretrained("models/donut-base")
processor.save_pretrained("processors/donut-base")
