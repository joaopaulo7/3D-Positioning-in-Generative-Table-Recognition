from configuration_dimbart import DiMBartConfig
from modeling_dimbart import DiMBartForCausalLM
from transformers import DonutSwinModel, VisionEncoderDecoderModel

class TabeleiroModel(VisionEncoderDecoderModel):
    
    def from_pretrained(path, from_donut = False, decoder_extra_config = dict(), donut_config = None, **kwargs):
        if from_donut:
            donut_model = VisionEncoderDecoderModel.from_pretrained(path, config = donut_config)
            donut_model.decoder.config.__dict__.update(decoder_extra_config)
            dimbart_config = DiMBartConfig(**donut_model.decoder.config.__dict__)
            state_dic = donut_model.state_dict()
            
            decoder = DiMBartForCausalLM(dimbart_config)
            donut_model.config.decoder = dimbart_config
            final_model = TabeleiroModel(config= donut_model.config, encoder = donut_model.encoder, decoder = decoder)
            final_model.load_state_dict(state_dic, strict = False)
            return final_model
        else:
            encoder = DonutSwinModel.from_pretrained(path+'/donut_encoder')
            decoder = DiMBartForCausalLM.from_pretrained(path+'/dimbart_decoder', ignore_mismatched_sizes=True)
            return TabeleiroModel(encoder = encoder, decoder= decoder)
    
    
    def save_pretrained(self, path):
        self.encoder.save_pretrained(path+'/donut_encoder')
        self.decoder.save_pretrained(path+'/dimbart_decoder')