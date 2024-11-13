import time
import onnx
import torch
import argparse
import importlib.util

import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models import StockEmbedding, attempt_load


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='weights', help='metricnet weights path')
    parser.add_argument('--batch_size', type=int, default=1, help='saved logging and dir path')
    parser.add_argument("--simplify", action="store_true", help="simplify onnx model")
    parser.add_argument("--dynamic", action="store_true", help="dynamic ONNX axes")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="True for using onnx_graphsurgeon to sort and remove unused",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="dynamic batch onnx for tensorrt and onnx-runtime",
    )
    
    args = parser.parse_args()
    
    t = time.time()    
    # Load Model
    tempt = torch.load(args.checkpoint_path)
    stock_embedder = StockEmbedding(tempt['hyper_parameters']['cfg']['model'], mode='ae')
    stock_embedder = attempt_load(stock_embedder, args.checkpoint_path)
    stock_embedder = stock_embedder.to(DEVICE)
    
    t_size = 24
    n_feature = 20
    stock_data = torch.randn(args.batch_size, t_size, n_feature)

    with torch.no_grad():
        stock_data = stock_data.to(DEVICE)
        encoder_output, decoder_output = stock_embedder(stock_data)
        
    # output saved path
    saved_path = args.checkpoint_path.replace(".ckpt", ".onnx")  # filename
    
    dynamic_axes = None
    if args.dynamic_batch:
        args.batch_size = "batch"
        dynamic_axes = {
            "input": {0: "batch"},
        }
        output_axes = {
            "encoder" : {0: "batch"},
            "decoder" : {0: "batch"},
            # "interpolate" : {0 : "batch"}
        }
        dynamic_axes.update(output_axes)
        
    torch.onnx.export(
        stock_embedder,  # model
        stock_data, # inputs
        saved_path,
        verbose=False,
        opset_version=12, 
        input_names = ['input'],
        output_names = ['encoder', 'decoder'],
        dynamic_axes=dynamic_axes
    )
    
    # Checks
    onnx_model = onnx.load(saved_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx mode
    
    if args.simplify:
        try:
            if importlib.util.find_spec('onnxsim') is None:
                import os
            
                os.system('pip install onnxsim')
                print("Import ONNXRuntime sucessfully")
            
            import onnxsim
            
            print("\nStarting to simplify ONNX...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")
            
    if args.cleanup:
        try:
            if importlib.util.find_spec('onnx_graphsurgeon') is None:
                import os
            
                os.system('pip install onnx_graphsurgeon')
                print("Import ONNXRuntime sucessfully")
            
            import onnx_graphsurgeon as gs

            print("\nStarting to cleanup ONNX using onnx_graphsurgeon...")
            graph = gs.import_onnx(onnx_model)
            graph = graph.cleanup().toposort()
            onnx_model = gs.export_onnx(graph)
        except Exception as e:
            print(f"Cleanup failure: {e}")
            

    input = [node for node in onnx_model.graph.input]
    output =[node for node in onnx_model.graph.output]
    print('\nInput: ', input)
    print('\nOutputs: ', output)
    
    onnx.save(onnx_model, saved_path)
    print("ONNX export success, saved as %s" % saved_path)

    # Finish
    print(
        "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
        % (time.time() - t)
    )