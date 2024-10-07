from TTS.utils.manage import ModelManager

# Initialize the model manager
model_manager = ModelManager()

# List available models and print them
models = model_manager.list_models()
print(models)
