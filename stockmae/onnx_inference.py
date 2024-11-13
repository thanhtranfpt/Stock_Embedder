import onnxruntime as rt
import numpy as np



class StockInference:
    def __init__(self, model_name_or_path):
        self.model_path = model_name_or_path
        self.load_model()
    
    def load_model(self, providers=[("CUDAExecutionProvider", 
                                    {"cudnn_conv_algo_search": "DEFAULT"}),
                                    "CPUExecutionProvider"]):

        sess_options = rt.SessionOptions()
        # sess_options.enable_profiling = True
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = rt.InferenceSession(self.model_path, sess_options, providers)
        self.input_names = [input.name for input in self.sess.get_inputs()] # get all input define
        self.output_names = [output.name for output in self.sess.get_outputs()] # get all output define
        
        print("Model load sucessesfully")
        print("Input Stock Model name: ", self.input_names)
        print("Output Stock Model name: ", self.output_names)
        
    def inference(self, x):
        assert hasattr(self, "sess"), "Model not loaded"
        assert len(x) == len(self.input_names), "Input shape mismatch"
        return self.sess.run(self.output_names, dict(zip(self.input_names, x)))
    
    def preprocessing(self, x):
        # the onnx inference support numpy 
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        
        return x
    
    def postprocessing(self, x):
        raise NotImplemented
        
    def predict(self, x):
        x = self.preprocessing(x)
        result = self.inference(x)
        # result = self.postprocessing(result)
        
        return result
    

if __name__ == '__main__':
    import torch 
    
    batch_size = 10
    t_size = 24
    n_feature = 20
    stock_data = torch.randn(batch_size, t_size, n_feature)
    
    model = StockInference('weights/best.onnx')

    encoder_embd, decoder_embd = model.predict(stock_data)
    
    print("Encoder embd: ", encoder_embd.shape)
    print("Decoder embd: ", decoder_embd.shape)