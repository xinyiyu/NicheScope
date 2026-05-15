import os, argparse
import pandas as pd
import scanpy as sc

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory of simulated data', required=True)
    parser.add_argument('--data-name', type=str, help='name of simulated data', required=True)
    parser.add_argument('--out-dir', type=str, help='directory for saving results', required=True)
    args = parser.parse_args()

    ## load processed ST data
    adata = sc.read(f"{args.data_dir}/{args.data_name}.h5ad")
    print(f'{len(adata)} cells.')

    ## extract coordinates and cell types
    input_dir = f'{args.out_dir}/{args.data_name}/Input'
    os.makedirs(input_dir, exist_ok=True)
    coords = adata.obs[['x', 'y']].copy()
    ct_labels = adata.obs['cell_type'].copy()
    coords.to_csv(f'{input_dir}/simu_Coordinates.txt', sep='\t', index=False, header=False)
    ct_labels.to_csv(f'{input_dir}/simu_CellTypeLabel.txt', sep='\t', index=False, header=False)
    with open(f'{input_dir}/ImageNameList.txt', 'w') as file:
        file.write('simu\n')
