from modeling_dimbart import DiMBartForCausalLM
from transformers import DonutSwinModel, VisionEncoderDecoderModel

class TabeleiroModel(VisionEncoderDecoderModel):
    
    def from_pretrained(path, from_donut = False, **kwargs):
        if from_donut:
            donut_model = super().from_pretrained(path)
            dimBart_config = DiMBartConfig(**donut_model.decoder.config.__dict__)
            state_dic = donut_model.state_dict()
            
            final_model = TabeleiroModel(econder = donut_model.encoder, decoder = DiMBartForCausalLM(dimbart_config))
            final_model.load_state_dict(state_dic, strict = False)
            return final_model
        else:
            encoder = DonutSwinModel.from_pretrained(path+'/donut_encoder')
            decoder = DiMBartForCausalLM.from_pretrained(path+'/dimbart_decoder')
            return TabeleiroModel(encoder = encoder, decoder= decoder)