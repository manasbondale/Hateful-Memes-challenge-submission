import torch
 
# registry is need to register our new model so as to be MMF discoverable
from mmf.common.registry import registry
# All model using MMF need to inherit BaseModel
from mmf.models.base_model import BaseModel
# ProjectionEmbedding will act as proxy encoder for FastText Sentence Vector
from mmf.modules.embeddings import ProjectionEmbedding
# Builder methods for image encoder and classifier
from mmf.utils.build import build_classifier_layer, build_image_encoder
 
# Register the model for MMF, "concat_vl" key would be used to find the model
@registry.register_model("concat_vl")
class LanguageAndVisionConcat(BaseModel):
   # All models in MMF get first argument as config which contains all
   # of the information you stored in this model's config (hyperparameters)
   def __init__(self, config, *args, **kwargs):
       # This is not needed in most cases as it just calling parent's init
       # with same parameters. But to explain how config is initialized we
       # have kept this
       super().__init__(config, *args, **kwargs)
  
   # This classmethod tells MMF where to look for default config of this model
   @classmethod
   def config_path(cls):
       # Relative to user dir root
       return "/content/hm_example_mmf/configs/models/concat_vl.yaml"
  
   # Each method need to define a build method where the model's modules
   # are actually build and assigned to the model
   def build(self):
       """
       Config's image_encoder attribute will used to build an MMF image
       encoder. This config in yaml will look like:
 
       # "type" parameter specifies the type of encoder we are using here.
       # In this particular case, we are using resnet152
       type: resnet152
    
       # Parameters are passed to underlying encoder class by
       # build_image_encoder
       params:
         # Specifies whether to use a pretrained version
         pretrained: true
         # Pooling type, use max to use AdaptiveMaxPool2D
         pool_type: avg
    
         # Number of output features from the encoder, -1 for original
         # otherwise, supports between 1 to 9
         num_output_features: 1
       """
       self.vision_module = build_image_encoder(self.config.image_encoder)
 
       """
       For classifer, configuration would look like:
       # Specifies the type of the classifier, in this case mlp
       type: mlp
       # Parameter to the classifier passed through build_classifier_layer
       params:
         # Dimension of the tensor coming into the classifier
         in_dim: 512
         # Dimension of the tensor going out of the classifier
         out_dim: 2
         # Number of MLP layers in the classifier
         num_layers: 0
       """
       self.classifier = build_classifier_layer(self.config.classifier)
      
       # ProjectionEmbeddings takes in params directly as it is module
       # So, pass in kwargs, which are in_dim, out_dim and module
       # whose value would be "linear" as we want linear layer
       self.language_module = ProjectionEmbedding(
           **self.config.text_encoder.params
       )
       # Dropout value will come from config now
       self.dropout = torch.nn.Dropout(self.config.dropout)
       # Same as Projection Embedding, fusion's layer params (which are param
       # for linear layer) will come from config now
       self.fusion = torch.nn.Linear(**self.config.fusion.params)
       self.relu = torch.nn.ReLU()
 
   # Each model in MMF gets a dict called sample_list which contains
   # all of the necessary information returned from the image
   def forward(self, sample_list):
       # Text input features will be in "text" key
       text = sample_list["text"]
       # Similarly, image input will be in "image" key
       image = sample_list["image"]
 
       text_features = self.relu(self.language_module(text))
       image_features = self.relu(self.vision_module(image))
 
       # Concatenate the features returned from two modality encoders
       combined = torch.cat(
[text_features, image_features.squeeze()], dim=1
 )
 
       # Pass through the fusion layer, relu and dropout
       fused = self.dropout(self.relu(self.fusion(combined)))
 
       # Pass final tensor from classifier to get scores
       logits = self.classifier(fused)
 
       # For loss calculations (automatically done by MMF
       # as per the loss defined in the config),
       # we need to return a dict with "scores" key as logits
       output = {"scores": logits}
 
       # MMF will automatically calculate loss
       return output
