# Arguments
import argparse, json, os

home = os.getcwd()
def load_arguments(home):
    '''
    Args:
        [Model related] model_cfg: predict_config.json
        [Stock related] config_dir: stock_config.json
        stock_dir: stock_data.csv
        [stock related] ts_size, mask_size, num_masks, total_mask_size: from stock_config.json
        [Metrics related] metric_iteration: from stock_config.json
        
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cfg', default=r'\Stock_Embedder\config\predict_config.json')
    args_dict = vars(parser.parse_args())

    # Load the stock_config.json
    model_dir = args_dict['model_cfg']
        
    with open(model_dir, 'r') as f:
        model_dict = json.load(fp=f)

    model_dict['home'] = home

    # Stock related and Metric related
    total_dict = {**model_dict, **args_dict}
    
    total_dict['data_name'] = 'stock'
    total_dict['stock_dir'] = 'YOUR TEST DATA PATH'
    total_dict['model_cfg'] = r'\Stock_Embedder\config\predict_config.json'
    
    total_dict['ra_dir'] = r'\Stock_Embedder\storage\synthesis\random_average'
    total_dict['cc_dir'] = r'\Stock_Embedder\storage\synthesis\cross_concat'
    total_dict['ca_dir'] = r'\Stock_Embedder\storage\synthesis\cross_average'  
    
    if not os.path.exists(total_dict['ra_dir']):
        os.makedirs(total_dict['ra_dir'])
    if not os.path.exists(total_dict['cc_dir']):
        os.makedirs(total_dict['cc_dir'])
    if not os.path.exists(total_dict['ca_dir']):
        os.makedirs(total_dict['ca_dir'])
    
    args = argparse.Namespace(**total_dict)
    
    return args
    
    

if __name__ == '__main__':
    home = os.getcwd()
    args = load_arguments(home)
    print(args)
